"""
Implements learning-related composite actions; by learning, we are primarily
referring to belief updates that modify long-term memory
"""
import re
import math
import torch
from itertools import permutations
from collections import defaultdict, Counter

import open3d as o3d
import numpy as np
import networkx as nx
from numpy.linalg import norm, inv
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from .interact import (
    _goal_selection, _tabulate_goal_selection_result,
    _match_existing_subassemblies
)
from ..vision.utils import (
    blur_and_grayscale, visual_prompt_by_mask, rmat2quat, xyzw2wxyz,
    flip_position_y, flip_quaternion_y, transformation_matrix
)
from ..lpmln import Literal
from ..lpmln.utils import flatten_ante_cons, wrap_args


EPS = 1e-10                 # Value used for numerical stabilization
SR_THRES = 0.8              # Mismatch surprisal threshold
U_IN_PR = 0.99              # How much the agent values information provided by the user

# Connectivity graph that represents pairs of 3D inspection images to be cross-referenced
CON_GRAPH = nx.Graph()
for i in range(8):
    CON_GRAPH.add_edge(i, i+8)
    CON_GRAPH.add_edge(i+8, i+16)
    CON_GRAPH.add_edge(i+16, i+24)
    CON_GRAPH.add_edge(i+24, i+32)
    if i < 7:
        CON_GRAPH.add_edge(i, i+1); CON_GRAPH.add_edge(i+8, i+9)
        CON_GRAPH.add_edge(i+16, i+17); CON_GRAPH.add_edge(i+24, i+25)
        CON_GRAPH.add_edge(i+32, i+33)
    else:
        CON_GRAPH.add_edge(i, i-7); CON_GRAPH.add_edge(i+8, i+1)
        CON_GRAPH.add_edge(i+16, i+9); CON_GRAPH.add_edge(i+24, i+17)
        CON_GRAPH.add_edge(i+32, i+25)
# Index of viewpoints whose data (camera pose, visible points and their descriptors)
# will be stored in long-term memory; storing for all consumes too much time (for
# examining possible initial poses during pose estimation) and space (storing all
# descriptors---even in reduced dimensionalities---would take too much)
STORE_VP_INDS = [
    0, 2, 4, 6, 8, 10, 12, 14, 24, 26, 28, 30, 32, 34, 36, 38
]

# Recursive helper methods for checking whether rule cons/ante is grounded (variable-
# free), lifted (all variables), contains any predicate referent as argument, or uses
# a reserved (pred type *) predicate 
is_grounded = lambda cnjt: all(not is_var for _, is_var in cnjt.args) \
    if isinstance(cnjt, Literal) else all(is_grounded(nc) for nc in cnjt)
is_lifted = lambda cnjt: all(is_var for _, is_var in cnjt.args) \
    if isinstance(cnjt, Literal) else all(is_lifted(nc) for nc in cnjt)
has_reserved_pred = \
    lambda cnjt: cnjt.name.startswith("sp_") \
        if isinstance(cnjt, Literal) else any(has_reserved_pred(nc) for nc in cnjt)


def identify_mismatch(agent, statement):
    """
    Test against vision-only sensemaking result to identify any mismatch btw.
    agent's & user's perception of world state. If any significant mismatch
    (as determined by surprisal) is identified, handle it by updating agent's
    exemplar base. Return bool indicating whether mismatch is identified and
    exemplar base is accordingly updated.
    """
    xb_updated = False          # Return value

    ante, cons = statement
    stm_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))

    # Proceed only when we have a grounded event & vision-only sensemaking
    # result has been obtained
    if not stm_is_grounded or agent.symbolic.concl_vis is None:
        return False

    # Make a yes/no query to obtain the likelihood of content
    reg_gr_v, _ = agent.symbolic.concl_vis
    q_response, _ = agent.symbolic.query(reg_gr_v, None, statement)
    ev_prob = q_response[()]

    # Proceed only if surprisal is above threshold
    surprisal = -math.log(ev_prob + EPS)
    if surprisal < -math.log(SR_THRES):
        return False

    for ante, cons in flatten_ante_cons(*statement):
        if len(ante+cons) != 1:
            # Not going to happen in our scope, but setting up a flag...
            continue

        if len(cons) == 1 and len(ante) == 0:
            # Positive grounded fact
            atom = cons[0]
            pol = "pos"
        else:
            # Negative grounded fact
            atom = ante[0]
            pol = "neg"

        conc_type, conc_ind = atom.name.split("_")
        conc_ind = int(conc_ind)
        ex_obj = atom.args[0][0]

        match conc_type:
            case "pcls":
                if agent.vision.scene[ex_obj]["exemplar_ind"] is None:
                    # New exemplar, mask & vector of the object should be added
                    pointer = ((conc_type, conc_ind, pol), (True, 0))
                else:
                    # Exemplar present in storage, only add pointer
                    ex_ind = agent.vision.scene[ex_obj]["exemplar_ind"]
                    pointer = ((conc_type, conc_ind, pol), (False, ex_ind))

                added_xi = _add_scene_and_exemplar_2d(
                    pointer,
                    agent.vision.scene[ex_obj]["scene_img"],
                    agent.vision.scene[ex_obj]["pred_mask"],
                    agent.vision.scene[ex_obj]["vis_emb"],
                    agent.lt_mem.exemplars
                )
                if agent.vision.scene[ex_obj]["exemplar_ind"] is None:
                    agent.vision.scene[ex_obj]["exemplar_ind"] = added_xi
                xb_updated = True

            case "prel":
                raise NotImplementedError   # Step back for relation prediction...

            case _:
                raise ValueError("Invalid concept type")

    return xb_updated


def identify_acknowledgement(agent, rule, prev_statements, prev_context):
    """
    Check whether the agent reported its estimate of something by its utterance
    and whether `rule` effectively acknowledges any of them positively or negatively.
    We may treat "Correct.", silence or explicit repetition as positive acknowledgement.
    Any conflicting statement from user would count as negative acknowledgement, as
    well as "Incorrect." or "No.".
    """
    cons, ante = rule
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))
    rule_has_pred_referent = (cons is None or has_pred_referent(cons)) and \
        (ante is None or has_pred_referent(ante))

    if rule_is_grounded and not rule_has_pred_referent:
        # Grounded event without constant predicate referents

        if len(agent.vision.latest_inputs) == 1:
            # Discussion about irrelevant contexts, can return early without doing anything
            return

        for (ti, ci), (speaker, (statement, _)) in prev_statements:
            # Only interested in whether an agent's statement is acknowledged
            if speaker != "Student": continue

            # Entailment checks; cons vs. statement, and cons vs. ~statement
            pos_entail_check, _ = Literal.entailing_mapping_btw(cons, statement)
            neg_entail_check, _ = Literal.entailing_mapping_btw(cons, (list(statement),))
                # Recall that a list of literals stands for the negation of the conjunction;
                # wrap it in a tuple again to make it an iterable

            pos_ack = pos_entail_check is not None and pos_entail_check >= 0
            neg_ack = neg_entail_check is not None and neg_entail_check >= 0

            if not (pos_ack or neg_ack): continue       # Nothing to do here

            # Determine polarity of the acknowledgement, collect relevant visual embeddings
            # from prev_scene, and then record
            polarity = pos_ack
            acknowledgement_data = (statement, polarity, prev_context)
            agent.lang.dialogue.acknowledged_stms[("curr", ti, ci)] = acknowledgement_data
                # "curr" indicates the acknowledgement is relevant to a statement in the
                # ongoing dialogue record


def identify_generics(agent, statement, prev_Qs, provenance):
    """
    For symbolic knowledge base expansion. Integrate the rule into KB by adding
    as new entry if it is not already included. For now we won't worry about
    intra-KB consistency, belief revision, etc. Return bool indicating whether
    the statement is identified as a novel generic rule, and KB is accordingly
    updated.
    """
    kb_updated = False          # Return value

    ante, cons = statement
    rule_is_lifted = (cons is None or is_lifted(cons)) and \
        (ante is None or is_lifted(ante))
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))

    # List of generic rules that can be extracted from the statement
    generics = []

    if rule_is_lifted:
        # Lifted generic rule statement, without any grounded term arguments

        # Assume default knowledge type here
        knowledge_type = "property"

        # First add the face-value semantics of the explicitly stated rule
        generics.append((rule, U_IN_PR, provenance, knowledge_type))

    # if rule_is_grounded and ante is None:
    #     # Grounded fact without constant predicate referents

    #     # For corrective feedback "This is Y" following the agent's incorrect answer
    #     # to the probing question "What kind of X is this?", extract 'Y entails X'
    #     # (e.g., What kind of truck is this? This is a fire truck => All fire trucks
    #     # are trucks). More of a universal statement rather than a generic one.

    #     # Collect concept entailment & constraints in questions made by the user
    #     # during this dialogue
    #     entail_consts = defaultdict(list)       # Map from pred var to set of pred consts
    #     instance_consts = {}                    # Map from ent const to pred var
    #     context_Qs = {}                         # Pointers to user's original question

    #     # Disregard all questions except the last one from user
    #     relevant_Qs = [
    #         (q_vars, q_cons, presup, raw)
    #         for _, (spk, (q_vars, (q_cons, _)), presup, raw) in prev_Qs
    #         if spk=="Teacher"
    #     ][-1:]

    #     for q_vars, q_cons, presup, raw in relevant_Qs:

    #         if presup is None:
    #             p_cons = []
    #         else:
    #             p_cons, _ = presup

    #         for qv, is_pred in q_vars:
    #             # Consider only predicate variables
    #             if not is_pred: continue

    #             for ql in q_cons:
    #                 # Constraint: P should entail conjunction {p1 and p2 and ...}
    #                 if ql.name=="sp_subtype" and ql.args[0][0]==qv:
    #                     entail_consts[qv] += [pl.name for pl in p_cons]

    #                 # Constraint: x should be an instance of P
    #                 if ql.name=="sp_isinstance" and ql.args[0][0]==qv:
    #                     instance_consts[ql.args[1][0]] = qv
            
    #             context_Qs[qv] = raw

    #     # Synthesize into a rule encoding the appropriate entailment
    #     for ent, pred_var in instance_consts.items():
    #         if pred_var not in entail_consts: continue
    #         entailed_preds = entail_consts[pred_var]

    #         # (Temporary) Only consider 1-place predicates, so match the first and
    #         # only entity from the arg list. Disregard negated conjunctions.
    #         entailing_preds = tuple(
    #             lit.name for lit in cons
    #             if isinstance(lit, Literal) and len(lit.args)==1 and lit.args[0][0]==ent
    #         )
    #         if len(entailing_preds) == 0: continue

    #         entailment_rule = (
    #             tuple(Literal(pred, [("X", True)]) for pred in entailed_preds),
    #             tuple(Literal(pred, [("X", True)]) for pred in entailing_preds)
    #         )
    #         knowledge_source = f"{context_Qs[pred_var]} => {provenance}"
    #         knowledge_type = "taxonomy"
    #         generics.append(
    #             (entailment_rule, U_IN_PR, knowledge_source, knowledge_type)
    #         )

    # Update knowledge base with obtained generic statements
    for rule, w_pr, knowledge_source, knowledge_type in generics:
        kb_updated |= agent.lt_mem.kb.add(
            rule, w_pr, knowledge_source, knowledge_type
        )

    return kb_updated


def handle_acknowledgement(agent, acknowledgement_info):
    """
    Handle (positive) acknowledgements to an utterance by the agent reporting its
    estimation on some state of affairs. The learning process differs based on the
    agent's strategy regarding how to extract learning signals from user's assents.
    """
    if agent.strat_assent == "doNotLearn": return       # Nothing further to do here

    ack_ind, ack_data = acknowledgement_info
    if ack_data is None: return         # Marked 'already processed'

    statement, polarity, context = ack_data
    vis_scene, pr_prog, kb_snap = context
    if ack_ind[0] == "prev":
        vis_raw = agent.vision.previous_input
    else:
        vis_raw = agent.vision.latest_inputs[-1]

    if polarity != True:
        # Nothing to do with negative acknowledgements
        agent.lang.dialogue.acknowledged_stms[ack_ind] = None
        return

    # Vision-only (neural) estimations recognized
    lits_from_vis = {
        rule.body[0].as_atom(): r_pr[0] for rule, r_pr, _ in pr_prog.rules
        if len(rule.head)==0 and len(rule.body)==2 and rule.body[0].naf
    }

    # Find literals to be considered as confirmed via user's assent; always include
    # each literal in `statement`, and others if relevant via some rule in KB
    lits_to_learn = set()
    for main_lit in statement:
        if main_lit.name.startswith("sp_"): continue

        # Find relevant KB entries (except taxonomy entries) containing the main literal
        relevant_kb_entries = [
            kb_snap.entries[ei][0]
            for ei in kb_snap.entries_by_pred[main_lit.name]
            if kb_snap.entries[ei][3] != "taxonomy"
        ]

        # Into groups of literals to be recognized as learning signals
        literal_groups = []
        for cons, ante in relevant_kb_entries:
            # Only consider rules with antecedents that don't have negated conjunction
            # as conjunct, when learning from assents
            if not all(isinstance(cnjt, Literal) for cnjt in ante): continue

            # Only consider unnegated conjuncts in consequent
            cons_pos = tuple(cnjt for cnjt in cons if isinstance(cnjt, Literal))

            for conjunct in ante:
                assert isinstance(conjunct, Literal)
                if conjunct.name==main_lit.name:
                    vc_map = {v: c for v, c in zip(conjunct.args, main_lit.args)}
                    literal_groups.append((cons_pos, ante, vc_map))

        for cons, ante, vc_map in literal_groups:
            # Substitute and identify the set of unique args occurring
            cons_subs = [l.substitute(terms=vc_map) for l in cons]
            ante_subs = [l.substitute(terms=vc_map) for l in ante]
            occurring_args = set.union(*[set(l.args) for l in cons_subs+ante_subs])

            # Consider all assignments of constants to variables & skolem functions
            # possible, and add viable cases to the set of acknowledged literals
            possible_assignments = [
                tuple(zip(occurring_args, prm))
                for prm in permutations(vis_scene, len(occurring_args))
            ]
            for prm in possible_assignments:
                # Skip this assignment if constant mismatch occurs
                const_mismatch = any(
                    arg_name!=cons and not (is_var or isinstance(arg_name, tuple))
                    for (arg_name, is_var), cons in prm
                )
                if const_mismatch: continue

                assig = {arg: (cons, False) for arg, cons in prm if arg[0]!=cons}
                cons_subs_full = {l.substitute(terms=assig) for l in cons_subs}
                ante_subs_full = {l.substitute(terms=assig) for l in ante_subs}

                # Union the fully substituted consequent to `lits_to_learn` only if
                # all the literals in the substituted antecedent can be found in visual
                # scene; apply some value threshold according to the strategy choice.
                if all(lit in lits_from_vis for lit in ante_subs_full):
                    for lit in cons_subs_full | ante_subs_full:
                        easy_positive = lit in lits_from_vis and lits_from_vis[lit] > 0.75
                        if agent.strat_assent == "threshold" and easy_positive:
                            # Adopting thresholding strategy, and estimated probability
                            # is already high enough; opt out of adding this as exemplar
                            continue

                        lits_to_learn.add(lit)

    # Add the instances represented by the literals as concept exemplars
    objs_to_add = set(); pointers = defaultdict(set)
    for lit in lits_to_learn:
        conc_type, conc_ind = lit.name.split("_")
        conc_ind = int(conc_ind)
        if conc_type == "prel": continue         # Relations are not neurally predicted

        pol = "pos" if not lit.naf else "neg"

        if vis_scene[lit.args[0][0]]["exemplar_ind"] is None:
            # New exemplar, mask & vector of the object should be added
            objs_to_add.add(lit.args[0][0])
            pointers[(conc_type, conc_ind, pol)].add(lit.args[0][0])
        else:
            # Exemplar present in storage, only add pointer
            ex_ind = agent.vision.scene[lit.args[0][0]]["exemplar_ind"]
            pointers[(conc_type, conc_ind, pol)].add(ex_ind)

    objs_to_add = list(objs_to_add)         # Assign arbitrary ordering
    _add_scene_and_exemplars_2d(
        objs_to_add, pointers,
        vis_scene, vis_raw, agent.lt_mem.exemplars
    )

    # Replace value with None to mark as 'already processed'
    agent.lang.dialogue.acknowledged_stms[ack_ind] = None


def handle_neologism(agent, novel_concepts, dialogue_state):
    """
    Identify neologisms (that the agent doesn't know which concepts they refer to)
    to be handled, attempt resolving from information available so far if possible,
    or record as unresolved neologisms for later addressing otherwise
    """
    # Return value, boolean flag whether there has been any change to the
    # exemplar base
    xb_updated = False

    neologisms = {
        tok: sym for tok, (sym, den) in agent.lang.dialogue.word_senses.items()
        if den is None
    }

    if len(neologisms) == 0: return False       # Nothing to do

    objs_to_add = set(); pointers = defaultdict(set)
    for tok, sym in neologisms.items():
        neologism_in_cons = tok[2].startswith("pc")
        neologisms_in_same_clause_ante = [
            n for n in neologisms
            if tok[:2]==n[:2] and n[2]==tok[2].replace("pc", "pa")
        ]
        if neologism_in_cons and len(neologisms_in_same_clause_ante)==0:
            # Occurrence in rule cons implies either definition or exemplar is
            # provided by the utterance containing this token... only if the
            # source clause is in indicative mood. If that's the case, register
            # new visual concept, and perform few-shot learning if appropriate
            pos, name = sym
            match pos:
                case "n" | "a":
                    conc_type = "pcls"
                case "v" | "p":
                    conc_type = "prel"
                case _:
                    raise ValueError("Invalid POS type")

            # Expand corresponding visual concept inventory
            conc_ind = agent.vision.add_concept(conc_type)
            novel_concept = (conc_type, conc_ind)
            novel_concepts.add(novel_concept)

            # Acquire novel concept by updating lexicon
            agent.lt_mem.lexicon.add((pos, name), novel_concept)

            ti = int(tok[0].strip("t"))
            ci = int(tok[1].strip("c"))
            (_, _, ante, cons), _ = dialogue_state.record[ti][1][ci]

            mood = dialogue_state.clause_info[f"t{ti}c{ci}"]["mood"]
            if len(ante) == 0 and mood == ".":
                # Labelled exemplar provided; add new concept exemplars to
                # memory, as feature vectors obtained from vision module backbone
                raise NotImplementedError       # Update as necessary
                args = [
                    agent.symbolic.value_assignment[arg] for arg in cons[0][2]
                ]

                match conc_type:
                    case "pcls":
                        if agent.vision.scene[args[0]]["exemplar_ind"] is None:
                            # New exemplar, mask & vector of the object should be added
                            objs_to_add.add(args[0])
                            pointers[(conc_type, conc_ind, "pos")].add(args[0])
                        else:
                            # Exemplar present in storage, only add pointer
                            ex_ind = agent.vision.scene[args[0]]["exemplar_ind"]
                            pointers[(conc_type, conc_ind, "pos")].add(ex_ind)
                    case "prel":
                        raise NotImplementedError   # Step back for relation prediction...
                    case _:
                        raise ValueError("Invalid concept type")

                # Register this instance as a handled mismatch, so that add_exs_2d()
                # won't be called upon this one during this loop again by handle_mismatch()
                stm = ((Literal(f"{conc_type}_{conc_ind}", wrap_args(*args)),), None)
                agent.symbolic.mismatches.append([stm, None, True])

                # Set flag that XB is updated
                xb_updated |= True

            elif len(ante) == 0 and mood == "!":
                # Neologism is an argument of a command; agent will report whichever
                # inability associated with the concept, and teacher will provide
                # some demonstration that involves an instance of the concept.
                # Lexicon will be appropriately expanded while analyzing the demo,
                # and the neologism will be duly considered as 'resolved' then.
                agent.lang.unresolved_neologisms.add(sym)

            else:
                # Otherwise not immediately resolvable
                agent.lang.unresolved_neologisms.add(sym)

        else:
            # Otherwise not immediately resolvable
            agent.lang.unresolved_neologisms.add(sym)

    if len(objs_to_add) > 0 or len(pointers) > 0:
        objs_to_add = list(objs_to_add)         # Assign arbitrary ordering
        _add_scene_and_exemplars_2d(
            objs_to_add, pointers,
            agent.vision.scene, agent.vision.latest_inputs[-1], agent.lt_mem.exemplars
        )

    return xb_updated


def report_neologism(agent, neologism):
    """
    Some neologism was identified and couldn't be resolved with available information
    in ongoing context; report lack of knowledge, so that user can provide appropriate
    information that characterize the concept denoted by the neologism (e.g., definition,
    exemplar)
    """
    # Update cognitive state w.r.t. value assignment and word sense
    
    # NL surface form and corresponding logical form
    surface_form = f"I don't know what '{neologism[1]}' means."
    gq = None; bvars = set(); ante = []
    cons = [
        ("sp", "unknown", ["x0", "x1"]),
        ("sp", "pronoun1", ["x0"]),
        neologism + (["x1"],)
    ]
    logical_form = (gq, bvars, ante, cons)

    # Referents & predicates info
    referents = {
        "e": { "mood": "." },       # Indicative
        "x0": { "entity": "_self", "rf_info": {} }
    }
    predicates = { "pc0": (("sp", "unknown"), "sp_unknown") }

    agent.lang.dialogue.to_generate.append(
        (logical_form, surface_form, referents, predicates, {})
    )


def analyze_demonstration(agent, demo_data):
    """
    Learn from demonstration; agent is provided with (full) demonstration represented
    as a pre-segmented & annotated sequence of atomic actions along with time-aligned
    visual observations. Learning outcomes to be extracted include: 2D, 3D features
    of novel part concepts, structure (constituent parts and their arrangements and
    contant constraints) of desired target goal concept, and any symbolic constraints
    that apply among different concepts.
    """
    # Shortcuts
    exemplars = agent.lt_mem.exemplars
    referents = agent.lang.dialogue.referents
    value_assignment = agent.lang.dialogue.value_assignment

    prev_img = None         # Stores previous steps' visual observations
    inspect_data = { "img": {}, "msk": {}, "pose": {} }
            # Buffer of 3d object instance views from inspect_~ actions

    # Sequentially process each demonstration step
    current_held = [None, None]; current_assembly_info = None
    nonatomic_subassemblies = set()
    part_labeling = {}; sa_labeling = {}
    vision_2d_data = defaultdict(list); vision_3d_data = {}; assembly_sequence = []
    for img, annotations, env_refs in demo_data:
        # Appropriately handle each annotation
        for (_, _, ante, cons), raw, clause_info in annotations:
            # Nothing to do with non-indicative annotations
            if clause_info["mood"] != ".": continue
            # Nothing to do with initial declaration of demonstration
            if raw.startswith("I will demonstrate"): continue

            if ante is None:
                # Non-quantified statement without antecedent
                if raw.startswith("# Action:"):
                    # Non-NL annotation of action intent specifying atomic action type
                    # and parameters
                    for lit in cons:
                        conc_type, conc_ind = lit.name.split("_")

                        # Only address action concepts
                        if conc_type != "arel": continue
                        conc_ind = int(conc_ind)

                        # Handle each action accordingly
                        act_name = agent.lt_mem.lexicon.d2s[(conc_type, conc_ind)][0][1]
                        act_type = act_name.split("_")[0]
                        left_or_right = 0 if act_name.endswith("left") else 1
                        match act_type:
                            case "pick":
                                # pick_up_~ action; track the held object and record 2d
                                # visual data (image + mask)
                                target_info = [
                                    rf_dis for rf_dis, rf_env in value_assignment.items()
                                    if rf_dis.startswith(lit.args[0][0]) and rf_env == lit.args[2][0]
                                ][0]
                                target_info = referents["dis"][target_info]
                                current_held[left_or_right] = (target_info["name"], {})

                                if target_info["name"] not in nonatomic_subassemblies:
                                    # Atomic part type to be remembered, record (image, mask)
                                    # pair by the name index
                                    data = (prev_img, env_refs["o0"]["mask"])
                                    vision_2d_data[target_info["name"]].append(data)

                            case "drop":
                                # drop_~ action
                                lastly_dropped, _ = current_held[left_or_right]
                                current_held[left_or_right] = None

                            case "assemble":
                                # assemble_~ action; record current step's assembly action info,
                                # then remember non-atomic subassemblies (distinguished by string
                                # name handle)

                                # Annotation about this join
                                current_assembly_info = [
                                    (current_held[0], referents["dis"][lit.args[3][0]]["name"]),
                                    (current_held[1], referents["dis"][lit.args[4][0]]["name"]),
                                    referents["dis"][lit.args[2][0]]["name"]
                                ]

                                # Update hand states
                                current_held[left_or_right] = (
                                    referents["dis"][lit.args[2][0]]["name"], {}
                                )
                                current_held[(left_or_right+1) % 2] = None

                                nonatomic_subassemblies.add(current_held[left_or_right][0])

                            case "inspect":
                                # inspect_~ action; collect all views for 3D reconstruction
                                viewed_obj = lit.args[2][0]
                                view_ind = int(referents["dis"][lit.args[3][0]]["name"])
                                if view_ind < 40:
                                    inspect_data["img"][view_ind] = img
                                if view_ind > 0:
                                    inspect_data["msk"][view_ind-1] = env_refs[viewed_obj]["mask"]

                elif raw.startswith("# Effect:"):
                    # Non-NL annotation of action effect specifying atomic action type
                    # and parameters
                    for lit in cons:
                        conc_type, conc_ind = lit.name.split("_")

                        # Only address action concepts
                        if conc_type != "arel": continue
                        conc_ind = int(conc_ind)

                        # Utility method for parsing serialized float lists passed
                        # as action effect
                        parse_floats = lambda ai: tuple(
                            float(v)
                            for v in referents["dis"][lit.args[ai][0]]["name"].split("/")
                        )

                        # Handle each action accordingly (actions 2~9 to be handled
                        # in our scope)
                        act_name = agent.lt_mem.lexicon.d2s[(conc_type, conc_ind)][0][1]
                        act_type = act_name.split("_")[0]
                        left_or_right = 0 if act_name.endswith("left") else 1
                        match act_type:
                            case "pick":
                                # pick_up_~ action; record (ground-truth) poses of individual
                                # atomic parts contained in the object picked up; beginning
                                # from the sixth argument, each given in the order of string
                                # identifier, rotation, position)
                                num_parts = int(referents["dis"][lit.args[5][0]]["name"])

                                # Update poses of individual parts
                                part_poses = current_held[left_or_right][1]
                                for i in range(num_parts):
                                    part_name = referents["dis"][lit.args[6+3*i][0]]["name"]
                                    part_poses[part_name] = (
                                        flip_quaternion_y(xyzw2wxyz(parse_floats(6+3*i+1))),
                                        flip_position_y(parse_floats(6+3*i+2))
                                    )

                            case "assemble":
                                # assemble_~ action; record poses of moved manipulator before
                                # & after movement, and poses of part instances contained in
                                # the assembly product
                                mnp_pose_before = (
                                    flip_quaternion_y(xyzw2wxyz(parse_floats(5))),
                                    flip_position_y(parse_floats(6))
                                )
                                mnp_pose_after = (
                                    flip_quaternion_y(xyzw2wxyz(parse_floats(7))),
                                    flip_position_y(parse_floats(8))
                                )
                                current_assembly_info += [
                                    "RToL" if left_or_right==0 else "LToR",
                                    mnp_pose_before, mnp_pose_after
                                ]
                                assembly_sequence.append(tuple(current_assembly_info))

                                num_parts = int(referents["dis"][lit.args[11][0]]["name"])
                                part_poses = current_held[left_or_right][1]
                                for i in range(num_parts):
                                    part_name = referents["dis"][lit.args[12+3*i][0]]["name"]
                                    part_poses[part_name] = (
                                        flip_quaternion_y(xyzw2wxyz(parse_floats(12+3*i+1))),
                                        flip_position_y(parse_floats(12+3*i+2))
                                    )

                                # Making way for a new one (not necessary, just signposting)
                                current_assembly_info = None

                            case "inspect":
                                # inspect_~ action; 3d poses of the object being inspected
                                # in camera coordinate, where the camera is put at different
                                # vantage points. Needed for appropriately adjusting ground
                                # truth poses passed from Unity environment.
                                if view_ind < 40:
                                    inspect_data["pose"][view_ind] = (
                                        flip_quaternion_y(xyzw2wxyz(parse_floats(2))),
                                        flip_position_y(parse_floats(3))
                                    )

                elif raw.startswith("Pick up a"):
                    # NL description providing labeling of the atomic part just
                    # picked up with a pick_up~ action
                    part_labeling[current_held[left_or_right][0]] = \
                        re.findall(r"Pick up a (.*)\.$", raw)[0]

                elif raw.startswith("This is a"):
                    # NL description providing labeling of the subassembly just
                    # placed on the desk with a drop~ action
                    sa_labeling[lastly_dropped] = \
                        re.findall(r"This is a (.*)\.$", raw)[0]

            else:
                # Quantified statement with antecedent; expecting a generic rule
                # expressed in natural language in the scope of our experiment
                raise NotImplementedError

        if len(inspect_data["img"]) == 40 and len(inspect_data["msk"]) == 40:
            inst_inspected = current_held[left_or_right][0]

            # Reconstruct 3D structure of the inspected object instance
            reconstruction = agent.vision.reconstruct_3d_structure(
                inspect_data["img"], inspect_data["msk"], inspect_data["pose"],
                CON_GRAPH, STORE_VP_INDS,
                # resolution_multiplier=1.25
            )
            vision_3d_data[inst_inspected] = reconstruction

            # Add select examples to 2D classification data as well
            vision_2d_data[inst_inspected] += [
                (inspect_data["img"][view_ind], inspect_data["msk"][view_ind])
                for view_ind in STORE_VP_INDS[4:8]
            ]

            # Make way for new data
            inspect_data = { "img": {}, "msk": {}, "pose": {} }

        prev_img = img

    # Tag each part instance with their visual concept index, registering any
    # new visual concepts & neologisms; we assume here all neologisms are nouns
    # (corresponding to 'pcls')
    inst2conc_map = {}
    if agent.cfg.exp.player_type in ["bool", "demo"]:
        # No access to any NL labeling; first assign new concept indices for
        # part instances with vision_3d_data available (obtained from multi-
        # view inspection), as they are understood to have all distinct types.
        # Instances without vision_3d_data will later be classified into one
        # of the newly assigned concepts.
        for part_inst in vision_3d_data:
            new_conc_ind = agent.vision.add_concept("pcls")
            inst2conc_map[part_inst] = new_conc_ind
            # Store 'identification code strings' so that they can be passed
            # to Unity environment when executing pick-up actions, which will
            # be compared against the list of licensed labels to simulate
            # pick-up actions with correct/perturbed poses. This is needed
            # for language-less player types only for their lack of access
            # to NL labels.
            type_code = next(
                obj["type_code"] for obj in agent.vision.scene.values()
                if obj["env_handle"] == part_inst
            )
            agent.lt_mem.lexicon.codesheet[new_conc_ind] = type_code
    else:
        # Has access to NL labeling of part & subassembly instances, use them
        assert agent.cfg.exp.player_type in ["label", "full"]
        for part_inst, part_type_name in part_labeling.items():
            sym = ("n", part_type_name)
            if sym in agent.lt_mem.lexicon:
                # Already registered
                _, conc_ind = agent.lt_mem.lexicon.s2d[sym][0]
                inst2conc_map[part_inst] = conc_ind
            else:
                # Neologism
                new_conc_ind = agent.vision.add_concept("pcls")
                agent.lt_mem.lexicon.add(sym, ("pcls", new_conc_ind))
                inst2conc_map[part_inst] = new_conc_ind

            # In whichever case, the symbol shouldn't be a 'unresolved neologism'
            if sym in agent.lang.unresolved_neologisms:
                agent.lang.unresolved_neologisms.remove(sym)
        # Also process any new subassembly concepts/neologisms
        for sa_inst, sa_type_name in sa_labeling.items():
            sym = ("n", sa_type_name)
            if sym not in agent.lt_mem.lexicon:
                new_conc_ind = agent.vision.add_concept("pcls")
                agent.lt_mem.lexicon.add(sym, ("pcls", new_conc_ind))
    

    # Process 3D vision data, storing reconstructed structure data in XB    
    for part_inst, reconstruction in vision_3d_data.items():
        point_cloud, views, descriptors = reconstruction

        # Store the reconstructed structure info in XB
        exemplars.add_exs_3d(
            inst2conc_map[part_inst],
            np.asarray(point_cloud.points), views, descriptors
        )

    # Process vision 2D data to obtain instance-level embeddings
    vision_2d_data = {
        part_inst: [
            (
                image, mask, bg_image := blur_and_grayscale(image),
                visual_prompt_by_mask(image, bg_image, [mask])
            )
            for image, mask in examples
        ]
        for part_inst, examples in vision_2d_data.items()
    }
    vis_model = agent.vision.model; vis_model.eval()
    with torch.no_grad():
        for part_inst, examples in vision_2d_data.items():
            examples_with_embs = []
            for data in examples:
                image, mask, _, vis_prompt = data
                vp_processed = vis_model.dino_processor(images=vis_prompt, return_tensors="pt")
                vp_pixel_values = vp_processed.pixel_values.to(vis_model.dino.device)
                vp_dino_out = vis_model.dino(pixel_values=vp_pixel_values, return_dict=True)
                f_vec = vp_dino_out.pooler_output.cpu().numpy()[0]
                examples_with_embs.append((image, mask, f_vec))
            vision_2d_data[part_inst] = examples_with_embs
    # Add 2D vision data in XB, based on the newly assigned pcls concept indices
    for part_inst, examples in vision_2d_data.items():
        if part_inst not in inst2conc_map:
            # Concept label info was not available (which happens for language-less
            # player types), 
            continue

        for image, mask, f_vec in examples:
            exs_2d = [{ "scene_id": None, "mask": mask, "f_vec": f_vec }]
            pointers = {
                ("pcls", inst2conc_map[inst], "pos" if inst==part_inst else "neg"): {
                    # (Whether object is newly added to XB, index 0 as only one is newly
                    # added each time)
                    (True, 0)
                }
                for inst in vision_2d_data if inst in inst2conc_map
            }
            exemplars.add_exs_2d(scene_img=image, exemplars=exs_2d, pointers=pointers)
    # Concept labels must be assigned to unlabeled part instances. An orthodox approach
    # would be to classify them to the closest example by visual features. However,
    # our abstraction approach of using a 'collision table cheat sheet' as oracle
    # complicates things when parts are misclassified at this stage. Since it is not
    # our primary focus to achieve perfect classification here (and it should be
    # relatively straightforward to achieve, in real scenarios; e.g., active perception)
    # we will just assume unlabeled instances are perfectly classified, using the
    # licensed label code data received from Unity.
    for part_inst in vision_2d_data:
        if part_inst in inst2conc_map: continue

        type_code = next(
            obj["type_code"] for obj in agent.vision.scene.values()
            if obj["env_handle"] == part_inst
        )
        inst2conc_map[part_inst] = next(
            conc_ind
            for conc_ind, label in agent.lt_mem.lexicon.codesheet.items()
            if label == type_code
        )

    # Finally process assembly data; estimate pose of assembled parts in hand,
    # infer 3D locations of 'contact points' based on manipulator pose difference,
    # and topological structure of (sub)assemblies
    assembly_trees = {}          # Tracking progress as structure trees
    cp2conc_map = {}
    for assembly_step in assembly_sequence:
        # Unpacking assembly step information
        (obj_l, part_poses_l), contact_l = assembly_step[0]
        (obj_r, part_poses_r), contact_r = assembly_step[1]
        resulting_subassembly = assembly_step[2]
        direction, pose_src_manip_before, pose_src_manip_after = assembly_step[3:6]

        # Find in each subassembly the parts that are directly joined by the
        # assemble action
        part_inst_l = contact_l.split("/")[0]
        part_inst_r = contact_r.split("/")[0]
        part_conc_ind_l = inst2conc_map[part_inst_l]
        part_conc_ind_r = inst2conc_map[part_inst_r]

        # Obtain oracle cp identifier strings by reducing part instance field
        # values into part (super)type values
        part_supertype_l = re.findall(r"t_(.*)_\d+$", part_inst_l)[0]
        part_supertype_r = re.findall(r"t_(.*)_\d+$", part_inst_r)[0]
        cp_name_l = f"{part_supertype_l}/" + contact_l.split("/")[1]
        cp_name_r = f"{part_supertype_r}/" + contact_r.split("/")[1]

        # Contact point to corresponding concept, assigning new concepts as
        # needed on the fly
        if cp_name_l in cp2conc_map:
            cp_conc_ind_l = cp2conc_map[cp_name_l]
        else:
            cp_conc_ind_l = agent.vision.add_concept("pcls")
            cp2conc_map[cp_name_l] = cp_conc_ind_l
        if cp_name_r in cp2conc_map:
            cp_conc_ind_r = cp2conc_map[cp_name_r]
        else:
            cp_conc_ind_r = agent.vision.add_concept("pcls")
            cp2conc_map[cp_name_r] = cp_conc_ind_r

        # Inferring contact point poses. Each contact point in a 3D part
        # structure has different pose according to which contact point
        # it is joining to.
        part_pose_l = part_poses_l[part_inst_l]
        part_pose_r = part_poses_r[part_inst_r]
        tmat_l = transformation_matrix(*part_pose_l)
        tmat_r = transformation_matrix(*part_pose_r)

        if direction == "RToL":
            tmat_tgt_obj = tmat_l; tmat_src_obj_before = tmat_r
            pcl_tgt = exemplars.object_3d[part_conc_ind_l][0]
            pcl_src = exemplars.object_3d[part_conc_ind_r][0]
        else:
            tmat_tgt_obj = tmat_r; tmat_src_obj_before = tmat_l
            pcl_tgt = exemplars.object_3d[part_conc_ind_r][0]
            pcl_src = exemplars.object_3d[part_conc_ind_l][0]
        pcl_tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl_tgt))
        pcl_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl_src))

        # Relation among manipulator & object transformations:
        # [Tr. of src object after movement]
        # = [Tr. of src manipulator after movement] *
        #   inv([Tr. of src manipulator before movement]) *
        #   [Tr. of src object before movement]
        tmat_src_manip_before = transformation_matrix(*pose_src_manip_before)
        tmat_src_manip_after = transformation_matrix(*pose_src_manip_after)
        tmat_src_obj_after = tmat_src_manip_after @ \
            inv(tmat_src_manip_before) @ tmat_src_obj_before

        # Align the point clouds of the involved parts according to the
        # obtained transformations to (arbitrarily) define a contact point
        # to be aligned when assembled. Define both position and rotation.
        pcl_tgt.transform(tmat_tgt_obj); pcl_src.transform(tmat_src_obj_after)

        # Contact point poses are inferred to be weighted centroid of the
        # associated two point clouds (position), whose orientation are determined
        # by PCA in the 3D coordinate
        points_cat = np.concatenate([pcl_tgt.points, pcl_src.points])

        # Position of contact point: weighted centroid of points in both clouds,
        # weighted by distances (gaussian kernel) to nearest points in each other
        pdists = pairwise_distances(pcl_tgt.points, pcl_src.points)
        min_dists_cat = np.concatenate([pdists.min(axis=1), pdists.min(axis=0)])
        weights = np.exp(-min_dists_cat / min_dists_cat.mean())
        weights /= weights.sum()
        cp_position = weights[:,None] * points_cat
        cp_position = cp_position.sum(axis=0)

        # Rotation of contact point: run PCA, select the first two principal
        # components as x and y axis (with z axis automatically determined by
        # cross product), obtain rotation matrix from the xyz axes
        x_axis, y_axis = PCA(n_components=2).fit(points_cat).components_
        x_axis /= norm(x_axis); y_axis /= norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= norm(z_axis)
        cp_rmat = np.stack([x_axis, y_axis, z_axis], axis=1)

        # Finally obtain the 3D poses on source & target parts; for position,
        # apply inverse of the transformations applied to the point clouds;
        # for rotation, apply the rotation matrix first, then inverse of the
        # rotation part of the transformations
        pose_cp_tgt = (
            rmat2quat(tmat_tgt_obj[:3,:3].T @ cp_rmat),
            tuple((inv(tmat_tgt_obj) @ np.append(cp_position, 1))[:3].tolist())
        )
        pose_cp_src = (
            rmat2quat(tmat_src_obj_after[:3,:3].T @ cp_rmat),
            tuple((inv(tmat_src_obj_after) @ np.append(cp_position, 1))[:3].tolist())
        )
        pose_cp_l = pose_cp_tgt if direction == "RToL" else pose_cp_src
        pose_cp_r = pose_cp_src if direction == "RToL" else pose_cp_tgt

        # Register contact point type and pose info as needed. Also providing
        # the oracle string name in this work, so that we can 'cheat' when
        # we run 'collision checks' for planning
        exemplars.add_exs_3d(
            part_conc_ind_l, None, None, None,
            { cp_conc_ind_l: ({ cp_conc_ind_r: pose_cp_l }, cp_name_l) }
        )
        exemplars.add_exs_3d(
            part_conc_ind_r, None, None, None,
            { cp_conc_ind_r: ({ cp_conc_ind_l: pose_cp_r } , cp_name_r) }
        )

        # Add new subassembly tree by adding concept-annotated nodes and
        # connecting them with contact-annotated edges
        def flatten_paths(sa_graph):
            # Recursive helper method for obtaining the map from individual
            # atomic part name handle in the provided subassembly graph to
            # their flattened paths
            flattened_paths = {}
            for n in sa_graph:
                match sa_graph.nodes[n]["node_type"]:
                    case "atomic":
                        flattened_paths[n] = n
                    case "sa":
                        sa_node_paths = flatten_paths(assembly_trees[n])
                        for hdl, path in sa_node_paths.items():
                            flattened_paths[hdl] = f"{n}/{path}"
            return flattened_paths
        subassemblies = []; connect_nodes = []; part_paths = []
        lr_bundle = zip(
            [obj_l, obj_r], [part_conc_ind_l, part_conc_ind_r],
            [part_inst_l, part_inst_r]
        )
        for obj, part_conc_ind, part_inst in lr_bundle:
            if obj in part_labeling or obj in vision_2d_data:
                # Atomic part if listed in part_labeling (for languageful
                # player types) or vision_2d_data (fallback for languageless
                # player types); create a new node
                sa_graph = nx.Graph()
                sa_graph.add_node(
                    obj, node_type="atomic",
                    conc=part_conc_ind, part_paths={ part_inst: part_inst }
                )
                subassemblies.append(sa_graph)
                connect_nodes.append(obj)
                part_paths.append(obj)
            elif obj in sa_labeling:
                # Meaningful subassembly if listed in sa_labeling; create
                # a new node
                sa_graph = nx.Graph()
                sa_graph.add_node(
                    obj, node_type="sa",
                    conc=agent.lt_mem.lexicon.s2d[("n", sa_labeling[obj])][0][1],
                    part_paths=flatten_paths(assembly_trees[obj])
                )
                subassemblies.append(sa_graph)
                connect_nodes.append(obj)
                part_paths.append(
                    obj + "/" + sa_graph.nodes[obj]["part_paths"][part_inst]
                )
            else:
                # Subassembly but not annotated as meaningful, pop from progress
                assert obj in assembly_trees
                sa_graph = assembly_trees.pop(obj)
                all_paths = {
                    hdl: (n, f"{n}/{path}") 
                        if sa_graph.nodes[n]["node_type"] == "sa" else (n, n)
                    for n in sa_graph
                    for hdl, path in sa_graph.nodes[n]["part_paths"].items()
                }
                subassemblies.append(sa_graph)
                connect_nodes.append(all_paths[part_inst][0])
                part_paths.append(all_paths[part_inst][1])

        assembly_trees[resulting_subassembly] = nx.union(*subassemblies)
        assembly_trees[resulting_subassembly].add_edge(
            connect_nodes[0], connect_nodes[1],
            contact={
                connect_nodes[0]: (part_paths[0], cp_conc_ind_l),
                connect_nodes[1]: (part_paths[1], cp_conc_ind_r)
            }
        )

    # Remaining assembly trees are those of meaningful substructures, process
    # and remember them in the long-term memory
    node_reindex = {
        subassembly_name: { n: i for i, n in enumerate(tree.nodes) }
        for subassembly_name, tree in assembly_trees.items()
    }           # Assign arbitrary integer ordering to nodes in each tree
    for subassembly_name, tree in assembly_trees.items():
        # Create a new assembly graph, relabeling node names into case-neutral
        # indices
        neutral_tree = nx.Graph()

        # Accordingly neutralize node and edge name & data
        for n, n_data in tree.nodes(data=True):
            n_ind = node_reindex[subassembly_name][n]
            n_data_new = {
                "node_type": n_data["node_type"],
                "conc": n_data["conc"] if n_data["node_type"] == "atomic"
                    else (n_data["conc"], 0)
                    # This assumes all involved subassembly concepts are the
                    # first entry in the list of all of its possible structures
            }
            neutral_tree.add_node(n_ind, **n_data_new)
        for n1, n2, e_data in tree.edges(data=True):
            n1_ind, n2_ind = (
                node_reindex[subassembly_name][n1],
                node_reindex[subassembly_name][n2]
            )
            part_path_n1, cp_conc_ind_n1 = e_data["contact"][n1]
            part_path_n2, cp_conc_ind_n2 = e_data["contact"][n2]
            # Translating instance-based paths to lifted, node-based paths
            part_path_n1 = [subassembly_name] + part_path_n1.split("/")
            part_path_n2 = [subassembly_name] + part_path_n2.split("/")
            part_path_n1 = tuple(
                node_reindex[part_path_n1[i]][part_path_n1[i+1]]
                for i in range(len(part_path_n1)-1)
            )
            part_path_n2 = tuple(
                node_reindex[part_path_n2[i]][part_path_n2[i+1]]
                for i in range(len(part_path_n2)-1)
            )
            e_data_new = {
                "contact": {
                    n1_ind: (part_path_n1, cp_conc_ind_n1),
                    n2_ind: (part_path_n2, cp_conc_ind_n2)
                }
                # An assumption under work here is that each part cannot
                # have more than one instances of the same contact point
                # type; i.e., each contact point within a part can be
                # uniquely determined by contact point type concept
            }
            neutral_tree.add_edge(n1_ind, n2_ind, **e_data_new)

        # Parse subassembly concept type & index
        sa_conc = agent.lt_mem.lexicon.s2d[("n", sa_labeling[subassembly_name])][0]
        sa_conc = (sa_conc[0], int(sa_conc[1]))

        # Store the structure in KB
        agent.lt_mem.kb.add_structure(sa_conc, neutral_tree)


def posthoc_episode_analysis(agent):
    """
    After the task for the current episode is fulfilled, compare the final
    estimated environemnt state---namely estimated part types of each object
    used in the finished assembly---against agent's current knowledge, then
    resolve any mismatch by updating knowledge state. In our scope, primary
    learning signals to be extracted are positive/negative exemplars.
    """
    exec_state = agent.planner.execution_state      # Shortcut var

    # Purge agent's current committed recognitions before final state
    # estimation, leaving only label info from teacher

    # Direct positive labels (languageful agents) and pairwise
    # negative labels (languageless agents)
    resolved_record = agent.lang.dialogue.export_resolved_record()
    labeling_feedback = [
        (lit.args[0][0], int(lit.name.strip("pcls_")))
        for spk, turn_clauses in resolved_record
        for ((_, _, _, cons), _, clause_info) in turn_clauses
        if cons is not None
        for lit in cons
        if spk == "Teacher" and clause_info["mood"] == "." \
            and lit.name.startswith("pcls")
    ]
    exec_state["recognitions"] = dict(labeling_feedback) | {
        objs: labels
        for objs, labels in exec_state["recognitions"].items()
        if isinstance(objs, tuple)
    }

    # Obtain agent's final estimate on how each atomic part in the completed
    # subassembly is contributing to the finished structure, by running the
    # goal selection ASP program for the last time. Namely, obtain the
    # node unification options for each object, and 'determine' their types
    # among possible options with predicted confidence values obtained from
    # few-shot classifiers. Update the exemplar base accordingly.
    build_target = agent.planner.execution_state["plan_goal"][1]
    best_model = _goal_selection(agent, build_target)
    tabulated_results = _tabulate_goal_selection_result(best_model)
    connection_graph, node_unifications, atomic_node_concs = \
        tabulated_results[3:]

    # As done in `.interact._plan_assembly` procedure, we could try to further
    # narrow down possible unification choices for the unbounded nodes
    final_sa = next(
        sa for sa, sa_graph in exec_state["connection_graphs"].items()
        if len(sa_graph) > 1
    )
    possible_mappings = _match_existing_subassemblies(
        connection_graph, node_unifications, atomic_node_concs, exec_state
    )
    for ism in possible_mappings["anchored"] + possible_mappings["lifted"]:
        for n, ex_obj in ism.items():
            node_unifications[f"{final_sa}_{ex_obj}"].add(n)

    # Collect possible atomic part types for each existing object
    obj_possible_concs = defaultdict(set)
    for oi, nodes in node_unifications.items():
        oi = re.findall(r"s\d+_(o\d+)", oi)[0]
        obj_possible_concs[oi] |= {atomic_node_concs[n] for n in nodes}

    # Part type selection must comply with the total atomic part type counts
    # for the selected target structure
    required_type_counts = Counter(atomic_node_concs.values())

    # Algorithm for enumerating every such category commitment, because
    # naively testing every cartesian product may take exponentiallt long
    # (Props to ChatGPT for saving my time)

    # Start with objects with smaller numbers of type choices in order
    # to minimize branching factor
    objs_sorted = sorted(
        obj_possible_concs, key=lambda obj: len(obj_possible_concs[obj])
    )
    # Object assignment and type counts
    assignment = {}; type_counts = defaultdict(int)
    def recursive_assign(idx):
        if idx == len(obj_possible_concs):
            # All objects are assigned a type, yield only if assignment
            # satisfies the required type counts
            if type_counts == required_type_counts:
                yield assignment.copy()
            return
        
        # Forward check: for each type, count how many of the remaining
        # objects can cover that type
        remaining_objs = objs_sorted[idx:]
        for c, req_cnt in required_type_counts.items():
            needed = req_cnt - type_counts[c]
            if needed < 0:
                # Already exceeded the required count for the type, return
                return
            # Counting remaining objects that can cover the type
            available_objs = [
                obj for obj in remaining_objs
                if c in obj_possible_concs[obj]
            ]
            if len(available_objs) < needed:
                # The remainder of the assignment can never satisfy the
                # requirement further on
                return

        # Choosing next object to allocate
        obj = objs_sorted[idx]
        for c in obj_possible_concs[obj]:
            if type_counts[c] < required_type_counts[c]:
                # Update assignment & type counts for the choice accordingly
                assignment[obj] = c
                type_counts[c] += 1

                # Recurse for the remaining ones
                yield from recursive_assign(idx + 1)

                # Backtrack after done
                del assignment[obj]
                type_counts[c] -= 1

    # Examine each possibility for each object and commit to the best one
    # based on existing few-shot classifier for each candidate concept,
    # while accounting for any pairwise negative label info
    scenarios_scored = []
    for scn in recursive_assign(0):
        # All pairwise negative label info should be observed
        if any(
            scn[objs[0]] == -labels[0] and scn[objs[1]] == -labels[1]
            for objs, labels in exec_state["recognitions"].items()
            if isinstance(objs, tuple)
        ): continue

        prob_score = sum(
            agent.vision.scene[oi]["pred_cls"][c] for oi, c in scn.items()
        )
        scenarios_scored.append((scn, prob_score.item()))

    # Select the best allocation of objects to nodes by score sum    
    final_recognitions = sorted(
        scenarios_scored, reverse=True, key=lambda x: x[1]
    )[0][0]

    pr_thres = 0.8                  # Threshold for adding exemplars
    exemplars = {}; pointers = defaultdict(set)
    for oi, best_conc in final_recognitions.items():
        # Add to the list of positive exemplars if the predicted probability
        # for the best type concept is lower than the threshold
        pred_prob = agent.vision.scene[oi]["pred_cls"][best_conc].item()
        if pred_prob < pr_thres:
            exemplars[oi] = {
                "scene_id": None,
                "mask": agent.vision.scene[oi]["pred_mask"],
                "f_vec": agent.vision.scene[oi]["vis_emb"]
            }
            pointers[("pcls", best_conc, "pos")].add(oi)

    # Extract any inferrable negative exemplars from pairwise negative
    # label info
    for objs, labels in exec_state["recognitions"].items():
        if isinstance(objs, str): continue
        assert isinstance(objs, tuple)
        obj1, obj2 = objs
        label1, label2 = -labels[0], -labels[1]

        obj1_is_label1 = final_recognitions[obj1] == label1
        obj2_is_label2 = final_recognitions[obj2] == label2
        assert not (obj1_is_label1 and obj2_is_label2)
        if obj1_is_label1:
            # obj1 is label1, so obj2 shouldn't be label2
            pred_prob = agent.vision.scene[obj2]["pred_cls"][label2].item()
            if pred_prob > 1 - pr_thres:
                exemplars[obj2] = {
                    "scene_id": None,
                    "mask": agent.vision.scene[obj2]["pred_mask"],
                    "f_vec": agent.vision.scene[obj2]["vis_emb"]
                }
                pointers[("pcls", label2, "neg")].add(obj2)
        if obj2_is_label2:
            # obj2 is label2, so obj1 shouldn't be label1
            pred_prob = agent.vision.scene[obj1]["pred_cls"][label1].item()
            if pred_prob > 1 - pr_thres:
                exemplars[obj1] = {
                    "scene_id": None,
                    "mask": agent.vision.scene[obj1]["pred_mask"],
                    "f_vec": agent.vision.scene[obj1]["vis_emb"]
                }
                pointers[("pcls", label1, "neg")].add(obj1)

    # Get the very first scene image before any sort of manipulation took
    # place
    scene_img = next(
        vis_inp for vis_inp in agent.vision.latest_inputs
        if vis_inp is not None
    )
    # Reformat `exemplars` and `pointers`
    obj_ordering = list(exemplars)
    exemplars = [exemplars[oi] for oi in obj_ordering]
    pointers = {
        conc_dscr: [(True, obj_ordering.index(oi)) for oi in objs]
        for conc_dscr, objs in pointers.items()
        if len(objs) > 0
    }

    # Exemplar base update in batch
    agent.lt_mem.exemplars.add_exs_2d(
        scene_img=scene_img, exemplars=exemplars, pointers=pointers
    )

def _add_scene_and_exemplar_2d(pointer, scene_img, mask, f_vec, ex_mem):
    """
    Helper method factored out for adding a scene, object and/or concept exemplar
    pointer
    """
    exemplar_spec, (is_new_obj, exemplar_id) = pointer

    if is_new_obj:
        # New scene img, new ID should be assigend (flagged as None). Second
        # entry of the tuple `exemplar_id` is an integer index pointing to
        # an item in the `exemplars` list defined below.
        assert isinstance(exemplar_id, int)
        scene_id = None
    else:
        # Scene is already stored in memory, fetch the scene ID
        assert isinstance(exemplar_id, tuple) and len(exemplar_id) == 2
        scene_id = exemplar_id[0]

    # Add concept exemplar to memory
    scene_img = scene_img if scene_id is None else None
        # Need to pass the scene image if not already stored in memory
    exemplars = [{ "scene_id": scene_id, "mask": mask, "f_vec": f_vec }]
    pointers = { exemplar_spec: [(is_new_obj, exemplar_id)] }
    added_inds = ex_mem.add_exs_2d(
        scene_img=scene_img, exemplars=exemplars, pointers=pointers
    )

    return added_inds[0]
