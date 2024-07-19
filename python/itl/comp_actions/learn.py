"""
Implements learning-related composite actions; by learning, we are primarily
referring to belief updates that modify long-term memory
"""
import re
import math
import torch
from math import sin, cos, pi
from itertools import permutations
from collections import defaultdict

import inflect
import cv2 as cv
import open3d as o3d
import numpy as np
import networkx as nx
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..lpmln import Literal
from ..lpmln.utils import flatten_cons_ante, wrap_args
from ..vision.utils import (
    blur_and_grayscale, visual_prompt_by_mask, crop_images_by_masks,
    quaternion_to_rotation_matrix, xyzw2wxyz, flip_position_y, flip_quaternion_y
)


EPS = 1e-10                 # Value used for numerical stabilization
SR_THRES = 0.8              # Mismatch surprisal threshold
U_IN_PR = 0.99              # How much the agent values information provided by the user

# Pre-specified poses (quaternion & position pair) of 3D inspection viewpoints (total 16)
R = 0.3; Tr = (0, 0, R); theta_H = pi/4; theta_L = pi/8
H1 = cos(theta_H/2); H2 = sin(theta_H/2); L1 = cos(theta_L/2); L2 = sin(theta_L/2)
VP_POSES = [
    ## (qw/qx/qy/qz quaternion, tx/ty/tz position); rotation first, then translation
    # 'Lower-high' loop
    ((H1, H2, 0, 0), Tr),
    ((H1*cos(pi/4), H2*cos(pi/4), H1*sin(pi/4), H2*sin(pi/4)), Tr),
    ((0, 0, H1, H2), Tr),
    ((H1*cos(3*pi/4), H2*cos(3*pi/4), H1*sin(3*pi/4), H2*sin(3*pi/4)), Tr),
    # 'Lower-low' loop
    ((L1, L2, 0, 0), Tr),
    ((L1*cos(pi/4), L2*cos(pi/4), L1*sin(pi/4), L2*sin(pi/4)), Tr),
    ((0, 0, L1, L2), Tr),
    ((L1*cos(3*pi/4), L2*cos(3*pi/4), L1*sin(3*pi/4), L2*sin(3*pi/4)), Tr),
    # 'Upper-low' loop
    ((L1, -L2, 0, 0), Tr),
    ((L1*cos(pi/4), -L2*cos(pi/4), L1*sin(pi/4), -L2*sin(pi/4)), Tr),
    ((0, 0, L1, -L2), Tr),
    ((L1*cos(3*pi/4), -L2*cos(3*pi/4), L1*sin(3*pi/4), -L2*sin(3*pi/4)), Tr),
    # 'Upper-high' loop
    ((H1, -H2, 0, 0), Tr),
    ((H1*cos(pi/4), -H2*cos(pi/4), H1*sin(pi/4), -H2*sin(pi/4)), Tr),
    ((0, 0, H1, -H2), Tr),
    ((H1*cos(3*pi/4), -H2*cos(3*pi/4), H1*sin(3*pi/4), -H2*sin(3*pi/4)), Tr)
]
# Connectivity graph that represents pairs of 3D inspection images to be cross-referenced
CON_GRAPH = nx.Graph()
for i in range(4):
    CON_GRAPH.add_edge(i, i+4); CON_GRAPH.add_edge(i+4, i+8); CON_GRAPH.add_edge(i+8, i+12)
CON_GRAPH.add_edge(12, 13); CON_GRAPH.add_edge(1, 2)
CON_GRAPH.add_edge(14, 15); CON_GRAPH.add_edge(3, 0)

# Recursive helper methods for checking whether rule cons/ante is grounded (variable-
# free), lifted (all variables), contains any predicate referent as argument, or uses
# a reserved (pred type *) predicate 
is_grounded = lambda cnjt: all(not is_var for _, is_var in cnjt.args) \
    if isinstance(cnjt, Literal) else all(is_grounded(nc) for nc in cnjt)
is_lifted = lambda cnjt: all(is_var for _, is_var in cnjt.args) \
    if isinstance(cnjt, Literal) else all(is_lifted(nc) for nc in cnjt)
has_pred_referent = \
    lambda cnjt: any(isinstance(a,str) and a[0].lower()=="p" for a, _ in cnjt.args) \
        if isinstance(cnjt, Literal) else any(has_pred_referent(nc) for nc in cnjt)
has_reserved_pred = \
    lambda cnjt: cnjt.name.startswith("sp_") \
        if isinstance(cnjt, Literal) else any(has_reserved_pred(nc) for nc in cnjt)


def identify_mismatch(agent, rule):
    """
    Test against vision-only sensemaking result to identify any mismatch btw.
    agent's & user's perception of world state
    """
    cons, ante = rule
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))
    rule_has_pred_referent = (cons is None or has_pred_referent(cons)) and \
        (ante is None or has_pred_referent(ante))

    # Skip handling duplicate cases in the context
    if rule in [r for r, _, _ in agent.symbolic.mismatches]: return

    if (rule_is_grounded and not rule_has_pred_referent and 
        agent.symbolic.concl_vis is not None):
        # Grounded event without constant predicate referents, only if vision-only
        # sensemaking result has been obtained

        # Make a yes/no query to obtain the likelihood of content
        reg_gr_v, _ = agent.symbolic.concl_vis
        q_response, _ = agent.symbolic.query(reg_gr_v, None, rule)
        ev_prob = q_response[()]

        surprisal = -math.log(ev_prob + EPS)
        if surprisal >= -math.log(SR_THRES):
            agent.symbolic.mismatches.append([rule, surprisal, False])


def identify_confusion(agent, rule, prev_statements, novel_concepts):
    """
    Test against vision module output to identify any 'concept overlap' -- i.e.
    whenever the agent confuses two concepts difficult to distinguish visually
    and mistakes one for another.
    """
    cons, ante = rule
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))
    rule_has_pred_referent = (cons is None or has_pred_referent(cons)) and \
        (ante is None or has_pred_referent(ante))

    if (
        agent.cfg.exp.strat_feedback.startswith("maxHelp") and
        rule_is_grounded and ante is None and not rule_has_pred_referent
    ):
        # Positive grounded fact with non-reserved predicates that don't have constant
        # predicate referent args; only if the user adopts maxHelp* strategy and provides
        # generic NL feedback

        # Fetch agent's last answer; this assumes the last factual statement
        # by agent is provided as answer to the last question from user. This
        # might change in the future when we adopt a more sophisticated formalism
        # for representing discourse to relax the assumption.
        prev_statements_A = [
            stm for _, (spk, stm) in prev_statements
            if spk=="A" and not any(has_reserved_pred(cnjt) for cnjt in stm[0])
        ]
        if len(prev_statements_A) == 0:
            # Hasn't given an answer (i.e., "I am not sure.")
            return

        agent_last_ans = prev_statements_A[-1][0][0]
        ans_conc_type, ans_conc_ind = agent_last_ans.name.split("_")
        ans_conc_ind = int(ans_conc_ind)

        for lit in cons:
            # Disregard negated conjunctions
            if not isinstance(lit, Literal): continue

            # (Temporary) Only consider 1-place predicates, so retrieve the single
            # and first entity from the arg list
            assert len(lit.args) == 1

            conc_type, conc_ind = lit.name.split("_")
            conc_ind = int(conc_ind)

            if (conc_ind, conc_type) not in novel_concepts:
                # Disregard if the correct answer concept is novel to the agent and
                # has to be newly registered in the visual concept inventory
                continue

            if (lit.args == agent_last_ans.args and conc_type == ans_conc_type
                and conc_ind != ans_conc_ind):
                # Potential confusion case, as unordered label pair
                confusion_pair = frozenset([conc_ind, ans_conc_ind])

                if (conc_type, confusion_pair) not in agent.confused_no_more:
                    # Agent's best guess disagrees with the user-provided
                    # information
                    agent.vision.confusions.add((conc_type, confusion_pair))


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
            if speaker != "A": continue

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


def identify_generics(agent, rule, provenance, prev_Qs, generics):
    """
    For symbolic knowledge base expansion. Integrate the rule into KB by adding
    (for now we won't worry about intra-KB consistency, belief revision, etc.).
    Identified generics will be added to the provided `generics` list.
    """
    cons, ante = rule
    rule_is_lifted = (cons is None or is_lifted(cons)) and \
        (ante is None or is_lifted(ante))
    rule_is_grounded = (cons is None or is_grounded(cons)) and \
        (ante is None or is_grounded(ante))
    rule_has_pred_referent = (cons is None or has_pred_referent(cons)) and \
        (ante is None or has_pred_referent(ante))

    if rule_is_lifted:
        # Lifted generic rule statement, without any grounded term arguments

        # Assume default knowledge type here
        knowledge_type = "property"

        # First add the face-value semantics of the explicitly stated rule
        generics.append((rule, U_IN_PR, provenance, knowledge_type))

    if (rule_is_grounded and ante is None and not rule_has_pred_referent):
        # Grounded fact without constant predicate referents

        # For corrective feedback "This is Y" following the agent's incorrect answer
        # to the probing question "What kind of X is this?", extract 'Y entails X'
        # (e.g., What kind of truck is this? This is a fire truck => All fire trucks
        # are trucks). More of a universal statement rather than a generic one.

        # Collect concept entailment & constraints in questions made
        # by the user during this dialogue
        entail_consts = defaultdict(list)       # Map from pred var to set of pred consts
        instance_consts = {}                    # Map from ent const to pred var
        context_Qs = {}                         # Pointers to user's original question

        # Disregard all questions except the last one from user
        relevant_Qs = [
            (q_vars, q_cons, presup, raw)
            for _, (spk, (q_vars, (q_cons, _)), presup, raw) in prev_Qs if spk=="U"
        ][-1:]

        for q_vars, q_cons, presup, raw in relevant_Qs:

            if presup is None:
                p_cons = []
            else:
                p_cons, _ = presup

            for qv, is_pred in q_vars:
                # Consider only predicate variables
                if not is_pred: continue

                for ql in q_cons:
                    # Constraint: P should entail conjunction {p1 and p2 and ...}
                    if ql.name=="sp_subtype" and ql.args[0][0]==qv:
                        entail_consts[qv] += [pl.name for pl in p_cons]

                    # Constraint: x should be an instance of P
                    if ql.name=="sp_isinstance" and ql.args[0][0]==qv:
                        instance_consts[ql.args[1][0]] = qv
            
                context_Qs[qv] = raw

        # Synthesize into a rule encoding the appropriate entailment
        for ent, pred_var in instance_consts.items():
            if pred_var not in entail_consts: continue
            entailed_preds = entail_consts[pred_var]

            # (Temporary) Only consider 1-place predicates, so match the first and
            # only entity from the arg list. Disregard negated conjunctions.
            entailing_preds = tuple(
                lit.name for lit in cons
                if isinstance(lit, Literal) and len(lit.args)==1 and lit.args[0][0]==ent
            )
            if len(entailing_preds) == 0: continue

            entailment_rule = (
                tuple(Literal(pred, [("X", True)]) for pred in entailed_preds),
                tuple(Literal(pred, [("X", True)]) for pred in entailing_preds)
            )
            knowledge_source = f"{context_Qs[pred_var]} => {provenance}"
            knowledge_type = "taxonomy"
            generics.append(
                (entailment_rule, U_IN_PR, knowledge_source, knowledge_type)
            )


def handle_mismatch(agent, mismatch):
    """
    Handle cognition gap following some specified strategy. Note that we now
    assume the user (teacher) is an infallible oracle, and the agent doesn't
    question info provided from user.
    """
    rule, _, handled = mismatch

    if handled: return 

    objs_to_add = set(); pointers = defaultdict(set)
    for cons, ante in flatten_cons_ante(*rule):
        is_grounded = all(not is_var for l in cons+ante for _, is_var in l.args)

        if is_grounded and len(cons+ante)==1:
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
            args = [a for a, _ in atom.args]

            match conc_type:
                case "pcls":
                    if agent.vision.scene[args[0]]["exemplar_ind"] is None:
                        # New exemplar, mask & vector of the object should be added
                        objs_to_add.add(args[0])
                        pointers[(conc_type, conc_ind, pol)].add(args[0])
                    else:
                        # Exemplar present in storage, only add pointer
                        ex_ind = agent.vision.scene[args[0]]["exemplar_ind"]
                        pointers[(conc_type, conc_ind, pol)].add(ex_ind)
                case "prel":
                    raise NotImplementedError   # Step back for relation prediction...
                case _:
                    raise ValueError("Invalid concept type")

    objs_to_add = list(objs_to_add)         # Assign arbitrary ordering
    _add_scene_and_exemplars_2d(
        objs_to_add, pointers,
        agent.vision.scene, agent.vision.latest_inputs[-1], agent.lt_mem.exemplars
    )

    # Mark as handled
    mismatch[2] = True


def handle_confusion(agent, confusion):
    """
    Handle 'concept overlap' between two similar visual concepts. Two (fine-grained)
    concepts can be disambiguated by some symbolically represented generic rules,
    request such differences by generating an appropriate question. 
    """
    # This confusion is about to be handled
    agent.vision.confusions.remove(confusion)

    if agent.cfg.exp.strat_feedback.startswith("maxHelpExpl"):
        # When interacting with teachers with this strategy, generic KB rules are
        # elicited not by the difference questions
        return

    # New dialogue turn & clause index for the question to be asked
    ti_new = len(agent.lang.dialogue.record)
    ci_new = 0

    conc_type, conc_inds = confusion
    conc_inds = list(conc_inds)

    # For now we are only interested in disambiguating class (noun) concepts
    assert conc_type == "pcls"

    # Prepare logical form of the concept-diff question to ask
    q_vars = ((f"X2t{ti_new}c{ci_new}", False),)
    q_rules = (
        (("diff", "sp", tuple(f"{ri}t{ti_new}c{ci_new}" for ri in ["x0", "x1", "X2"]), False),),
        ()
    )
    ques_logical_form = (q_vars, q_rules)

    # Prepare surface form of the concept-diff question to ask
    pluralize = inflect.engine().plural
    conc_names = [
        agent.lt_mem.lexicon.d2s[(ci, conc_type)][0][0]
        for ci in conc_inds
    ]       # Fetch string name for concepts from the lexicon
    conc_names = [
        re.findall(r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", cn)
        for cn in conc_names
    ]       # Unpack camelCased names
    conc_names = [
        pluralize(" ".join(tok.lower() for tok in cn))
        for cn in conc_names
    ]       # Lowercase tokens and pluralize

    # Update cognitive state w.r.t. value assignment and word sense
    agent.symbolic.value_assignment.update({
        f"x0t{ti_new}c{ci_new}": f"{conc_type}_{conc_inds[0]}",
        f"x1t{ti_new}c{ci_new}": f"{conc_type}_{conc_inds[1]}"
    })

    ques_translated = f"How are {conc_names[0]} and {conc_names[1]} different?"

    agent.lang.dialogue.to_generate.append(
        ((None, ques_logical_form), ques_translated, {})
    )

    # No need to request concept differences again for this particular case
    # for the rest of the interaction episode sequence
    agent.confused_no_more.add(confusion)


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
        tok: sym for tok, (sym, den) in agent.symbolic.word_senses.items()
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
            (_, _, ante, cons), _, mood = dialogue_state.record[ti][1][ci]

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
    # New dialogue turn & clause index for the answer to be provided
    ti_new = len(agent.lang.dialogue.record)
    ci_new = len(agent.lang.dialogue.to_generate)
    ri_ignorance = f"t{ti_new}c{ci_new}"            # Denotes the ignorance state event
    ri_self = f"t{ti_new}c{ci_new}x0"               # Denotes self (i.e., agent)
    ri_neologism = f"t{ti_new}c{ci_new}x1"          # Denotes the neologism
    tok_ind = (f"t{ti_new}", f"c{ci_new}", "pc0")   # Denotes 'unknown' predicate

    # Update cognitive state w.r.t. value assignment and word sense
    agent.symbolic.value_assignment[ri_self] = "_self"
    agent.symbolic.word_senses[tok_ind] = (("sp", "unable"), "sp_unable")

    # Corresponding logical form
    gq = None; bvars = None; ante = []
    cons = [
        ("sp", "unknown", [ri_self, ri_neologism]),
        ("sp", "pronoun1", [ri_self]),
        neologism + ([ri_neologism],)
    ]

    agent.lang.dialogue.to_generate.append(
        (
            (gq, bvars, ante, cons),
            f"I don't know what '{neologism[1]}' means.",
            ".",
            {
                ri_self: { "source_evt": ri_ignorance },
                ri_neologism: { "source_evt": ri_ignorance, "is_pred": True }
            },
            {}
        )
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
    referents = agent.lang.dialogue.referents
    value_assignment = agent.symbolic.value_assignment

    prev_img = None         # Stores previous steps' visual observations
    inspect_data = { "img": {}, "msk": {} }
            # Buffer of 3d object instance views from inspect_~ actions

    # Sequentially process each demonstration step
    current_held = [None, None]; current_assembly_info = None
    nonatomic_subassemblies = set(); subassembly_labeling = {}
    vision_2d_data = {}; vision_3d_data = {}; assembly_sequence = []
    for img, annotations, env_refs in demo_data:
        # Appropriately handle each annotation
        for (_, _, ante, cons), raw, mood in annotations:
            # Nothing to do with non-indicative annotations
            if mood != ".": continue
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

                        # Handle each action accordingly (actions 1~8 to be handled
                        # in our scope)
                        left_or_right = (conc_ind-1) % 2
                        match conc_ind:
                            case 1 | 2:
                                # pick_up_~ action; track the held object and record 2d
                                # visual data (image + mask)
                                target_info = [
                                    rf_dis for rf_dis, rf_env in value_assignment.items()
                                    if rf_dis.startswith(lit.args[0][0]) and rf_env == lit.args[2][0]
                                ][0]
                                target_info = referents["dis"][target_info]
                                current_held[left_or_right] = target_info["name"]

                                if target_info["name"] not in nonatomic_subassemblies:
                                    # Atomic part type to be remembered, record (image, mask)
                                    # pair by the name index
                                    data = (prev_img, env_refs["o0"]["mask"])
                                    vision_2d_data[target_info["name"]] = data

                            case 3 | 4:
                                # drop_~ action
                                lastly_dropped = current_held[left_or_right]
                                current_held[left_or_right] = None

                            case 5 | 6:
                                # assemble_~ action; record 2d visual data (image + masks)
                                # and assembly action info, then remember non-atomic subassemblies
                                # (distinguished by string name handle)
                                current_assembly_info = [
                                    current_held[left_or_right], current_held[left_or_right+1 % 2],
                                    prev_img, env_refs["o0"]["mask"], env_refs["o1"]["mask"]
                                ]
                                current_held[left_or_right] = referents["dis"][lit.args[4][0]]["name"]
                                current_held[left_or_right+1 % 2] = None

                                nonatomic_subassemblies.add(current_held[left_or_right])

                            case 7 | 8:
                                # inspect_~ action; collect all views for 3D reconstruction
                                viewed_obj = lit.args[2][0]
                                view_ind = int(referents["dis"][lit.args[3][0]]["name"])
                                if view_ind < 16:
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

                        # Handle each action accordingly (actions 1~8 to be handled
                        # in our scope)
                        left_or_right = (conc_ind-1) % 2
                        match conc_ind:
                            case 5 | 6:
                                # assemble_~ action; record manipulator poses
                                parse_values = lambda ai: tuple(
                                    float(v) for v in referents["dis"][lit.args[ai][0]]["name"].split("/")
                                )
                                mnp_pose_left = (
                                    parse_values(2), xyzw2wxyz(parse_values(3))
                                )
                                mnp_pose_right = (
                                    parse_values(4), xyzw2wxyz(parse_values(5))
                                )
                                mnp_pose_moved = (
                                    parse_values(6), xyzw2wxyz(parse_values(7))
                                )
                                current_assembly_info += [
                                    "RToL" if left_or_right==0 else "LToR",
                                    mnp_pose_left, mnp_pose_right, mnp_pose_moved
                                ]
                                assembly_sequence.append(tuple(current_assembly_info))

                                current_assembly_info = None

                else:
                    # Additional natural language annotation, providing labeling of
                    # object instances of interest
                    subassembly_labeling[lastly_dropped] = cons[0].name

            else:
                # Quantified statement with antecedent; expecting a generic rule
                # expressed in natural language in the scope of our experiment
                raise NotImplementedError

        if len(inspect_data["img"]) == 16 and len(inspect_data["msk"]) == 16:
            # Reconstruct 3D structure of the inspected object instance
            # for mpl in [1, 1.5, 2]:
            for mpl in [1, 1.5]:
                # Try at most three times, increasing the 'resolution multiplier'
                # value each time the obtained point cloud doesn't have enough points
                reconstruction = agent.vision.reconstruct_3d_structure(
                    inspect_data["img"], inspect_data["msk"], VP_POSES, CON_GRAPH,
                    resolution_multiplier=mpl
                )
                point_cloud = reconstruction[0]

                # Break if enough points obtained
                if len(point_cloud.points) >= 1000: break

            vision_3d_data[current_held[left_or_right]] = reconstruction

            # Make way for new data
            inspect_data = { "img": {}, "msk": {} }

        prev_img = img

    # Process gathered data, 2D/3D vision and assembly contact points

    # Process 3D vision data first, registering new pcls concepts and storing
    # reconstructed structure data in XB
    inst2conc_map = {}
    for instance_name, reconstruction in vision_3d_data.items():
        point_cloud, views, descriptors = reconstruction

        # Assumption: All concepts for which 3D data is acquired are novel to
        # the learner agent and should be newly registered)
        new_conc_ind = agent.vision.add_concept("pcls")

        # Store the reconstructed structure info in XB
        agent.lt_mem.exemplars.add_exs_3d(new_conc_ind, point_cloud, views, descriptors)

        # Track corresponding concepts
        inst2conc_map[instance_name] = new_conc_ind

    # Further process vision 2D data to obtain instance-level embeddings
    vision_2d_data = {
        instance_name: (
            image, mask, bg_image := blur_and_grayscale(image),
            visual_prompt_by_mask(image, bg_image, [mask])
        )
        for instance_name, (image, mask) in vision_2d_data.items()
    }
    vis_model = agent.vision.model; vis_model.eval()
    with torch.no_grad():
        for instance_name, (image, mask, bg_image, vis_prompt) in vision_2d_data.items():
            vp_processed = vis_model.dino_processor(images=vis_prompt, return_tensors="pt")
            vp_pixel_values = vp_processed.pixel_values.to(vis_model.dino.device)
            vp_dino_out = vis_model.dino(pixel_values=vp_pixel_values, return_dict=True)
            f_vec = vp_dino_out.pooler_output.cpu().numpy()[0]
            vision_2d_data[instance_name] = (image, mask, f_vec)
    # Add 2D vision data in XB, based on the newly assigned pcls concept indices
    for instance_name, (image, mask, f_vec) in vision_2d_data.items():
        if instance_name not in inst2conc_map:
            # Cannot exactly specify which concept this instance classifies as, skip
            continue

        exemplars = [{ "scene_id": None, "mask": mask, "f_vec": f_vec }]
        pointers = {
            ("pcls", inst2conc_map[inst], "pos" if inst==instance_name else "neg"): {
                # (Whether object is newly added to XB, index 0 as only one is newly
                # added each time)
                (True, 0)
            }
            for inst in vision_2d_data if inst in inst2conc_map
        }
        agent.lt_mem.exemplars.add_exs_2d(
            scene_img=image, exemplars=exemplars, pointers=pointers
        )

    # Finally process assembly data; estimate pose of assembled parts in hand,
    # infer 3D locations of 'contact points' based on manipulator pose difference,
    # and topological structure of (sub)assemblies
    cam_K, distortion_coeffs = agent.vision.camera_intrinsics
    with torch.no_grad():
        for assembly_step in assembly_sequence:
            # Unpacking assembly step information
            targets = assembly_step[:2]
            image, *masks = assembly_step[2:5]
            direction, *poses, pose_moved = assembly_step[5:9]
            foo = assembly_step[9:]

            # Extract patch-level features as guided by masks
            zoomed_images, zoomed_masks, crop_dims = crop_images_by_masks(
                { 0: image, 1: image }, masks       # 0: Left, 1: Right
            )
            patch_features, lr_masks, lr_dims = vis_model.lr_features_from_masks(
                zoomed_images, zoomed_masks, 900, 2
            )
            D = patch_features[0].shape[-1]

            # Run below once for left target and right target
            lr_zipped = zip(
                targets, patch_features, lr_masks, lr_dims,
                zoomed_images, crop_dims, poses
            )       # Zipping data for left side and right side
            for zip_data in lr_zipped:
                # Length unpacking split into multiple lines...
                tgt, pth_ft = zip_data[:2]
                lr_msk, (lr_w, lr_h) = zip_data[2:4]
                z_img, (cr_x, _, cr_y, _), mnp_pose = zip_data[4:]

                # Unpack manipulator pose info, while accounting for Unity vs. vision
                # libraries y-axis difference
                mnp_position = flip_position_y(mnp_pose[0])
                mnp_rotation = flip_quaternion_y(mnp_pose[1])

                # Need to know which concept's instance the target object is
                if tgt in inst2conc_map:
                    conc_ind = inst2conc_map[tgt]
                else:
                    # TEMPORARY TEST
                    conc_ind = 1

                msk_flattened = lr_msk.reshape(-1)
                nonzero_inds = msk_flattened.nonzero()[0]

                # Scale ratio between zoomed image vs. low-res feature map
                x_ratio = z_img.width / lr_w
                y_ratio = z_img.height / lr_h

                # Compare against keypoint descriptors stored in XB, per each view
                point_cloud, views, descriptors = agent.lt_mem.exemplars.object_3d[conc_ind]
                pose_estimation_results = []
                for vi, view_info in views.items():
                    # Fetch keypoint descriptors
                    keypoints = sorted(view_info["visible_keypoints"])
                    kp_features = torch.tensor([descriptors[pi][vi] for pi in keypoints])
                    kp_features = kp_features.to(pth_ft.device)

                    # Compute cosine similarities between patches
                    features_nrm_1 = F.normalize(pth_ft.reshape(-1, D))
                    features_nrm_2 = F.normalize(kp_features)
                    S = (features_nrm_1 @ features_nrm_2.t()).cpu()

                    # (u,v)-coordinates of keypoints at the initial extrinsic guess
                    rmat_guess = quaternion_to_rotation_matrix(view_info["cam_quaternion"])
                    tvec_guess = np.array(mnp_position)
                    tr_mat = np.concatenate(
                        [
                            np.concatenate([rmat_guess, np.array([tvec_guess]).T], axis=1),
                            np.array([[0.0, 0.0, 0.0, 1.0]])
                        ]
                    )
                    tr_pcl = o3d.geometry.PointCloud(point_cloud).transform(tr_mat)
                    kp_guess_projections = cv.projectPoints(
                        np.array([tr_pcl.points[pi] for pi in keypoints]),
                        np.array([0.0] * 3), np.array([0.0] * 3),
                        cam_K, distortion_coeffs
                    )[0][:,0,:]

                    # Proximity scores (w.r.t. to guessed keypoint projection) computed
                    # with RBF kernel; for giving slight advantages to pixels close to
                    # initial guess projections
                    uv_coords = np.stack([
                        np.tile((np.arange(lr_w)*x_ratio + cr_x)[None], [lr_h, 1]),
                        np.tile((np.arange(lr_h)*y_ratio + cr_y)[:,None], [1, lr_w])
                    ], axis=-1)
                    sigma = 10
                    proximity = np.linalg.norm(
                        uv_coords[:,:,None] - kp_guess_projections[None,None], axis=-1
                    )
                    proximity = np.exp(-np.square(proximity) / (2 * (sigma ** 2)))

                    # Forward matching
                    agg_scores = S + 0.5 * proximity.reshape(-1, len(keypoints))
                    match_forward = linear_sum_assignment(
                        agg_scores[msk_flattened], maximize=True
                    )

                    # Extract 2D-3D correspondence based on the match
                    points_2d = [
                        (i % lr_w, i // lr_w) for i in nonzero_inds[match_forward[0]]
                    ]
                    points_2d = np.array([
                        (cr_x + lr_x * x_ratio, cr_y + lr_y * y_ratio)
                        for lr_x, lr_y in points_2d
                    ])
                    points_3d = np.array([
                        point_cloud.points[keypoints[i]] for i in match_forward[1]
                    ])

                    # Pose estimation by PnP with USAC (MAGSAC)
                    output_valid, rvec, tvec, _ = cv.solvePnPRansac(
                        points_3d, points_2d, cam_K, distortion_coeffs,
                        flags=cv.USAC_MAGSAC
                    )
                    if output_valid:
                        # Evaluate estimated pose by obtaining mean similarity
                        # scores at reprojected keypoints
                        reprojections = cv.projectPoints(
                            points_3d, rvec, tvec, cam_K, distortion_coeffs
                        )[0][:,0,:]
                        dists_to_reprojs = np.linalg.norm(
                            uv_coords[:,:,None] - reprojections[None,None], axis=-1
                        )
                        dists_to_reprojs = dists_to_reprojs.reshape(-1, len(keypoints))
                        reproj_coords = np.unravel_index(
                            dists_to_reprojs.argmin(axis=0), (lr_h, lr_w)
                        )
                        reproj_score_inds = reproj_coords + (np.arange(len(keypoints)),)
                        reproj_scores = S.reshape(lr_h, lr_w, -1)[reproj_score_inds]
                        # Only consider keypoints whose reprojections fall into the
                        # low-res mask; accounts for occlusions
                        within_mask_inds = [
                            lr_msk[(reproj_coords[0][i], reproj_coords[1][i])]
                            for i in range(len(keypoints))
                        ]
                        reproj_scores = reproj_scores[within_mask_inds]

                        # Store pose estimation results along with pose evaluation score
                        pose_estimation_results.append(
                            (rvec, tvec, reproj_scores.mean().item())
                        )

                # Select the best pose estimation result with highest score
                print(0)

            print(0)


def _add_scene_and_exemplars_2d(
        objs_to_add, pointers, current_scene, current_raw_img, ex_mem
    ):
    """
    Helper method factored out for adding a scene, objects and/or concept exemplar
    pointers
    """
    # Check if this scene is already stored in memory and if so fetch the ID
    scene_ids = [
        obj_info["exemplar_ind"][0] for obj_info in current_scene.values()
        if obj_info["exemplar_ind"] is not None
    ]
    assert len(set(scene_ids)) <= 1         # All same or none
    scene_id = scene_ids[0] if len(scene_ids) > 0 else None

    # Add concept exemplars to memory
    objs_to_add = list(objs_to_add)         # Assign arbitrary ordering
    scene_img = current_raw_img if scene_id is None else None
        # Need to pass the scene image if not already stored in memory
    exemplars = [
        {
            "scene_id": scene_id,
            "mask": current_scene[oi]["pred_mask"],
            "f_vec": current_scene[oi]["vis_emb"]
        }
        for oi in objs_to_add
    ]
    pointers = {
        conc_spec: {
            # Pair (whether object is newly added, index within objs_to_add if True,
            # (scene_id, obj_id) otherwise)
            (True, objs_to_add.index(oi)) if isinstance(oi, str) else (False, oi)
            for oi in objs
        }
        for conc_spec, objs in pointers.items()
    }
    added_inds = ex_mem.add_exs_2d(
        scene_img=scene_img, exemplars=exemplars, pointers=pointers
    )

    for oi, xi in zip(objs_to_add, added_inds):
        # Record exemplar storage index for corresponding object in visual scene,
        # so that any potential redundant additions can be avoided
        current_scene[oi]["exemplar_ind"] = xi
