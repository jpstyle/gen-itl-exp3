"""
Implements learning-related composite actions; by learning, we are primarily
referring to belief updates that modify long-term memory
"""
import re
import math
import torch
from math import sin, cos, radians
from itertools import product, permutations
from collections import defaultdict

import inflect
import cv2 as cv
import open3d as o3d
import numpy as np
import networkx as nx
from numpy.linalg import norm, inv
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from skimage.morphology import dilation, disk

from ..lpmln import Literal
from ..lpmln.utils import flatten_cons_ante, wrap_args
from ..vision.utils import (
    blur_and_grayscale, visual_prompt_by_mask, crop_images_by_masks,
    masks_bounding_boxes, mask_iou, quat2rmat, rmat2quat, xyzw2wxyz,
    flip_position_y, flip_quaternion_y, transformation_matrix
)


EPS = 1e-10                 # Value used for numerical stabilization
SR_THRES = 0.8              # Mismatch surprisal threshold
U_IN_PR = 0.99              # How much the agent values information provided by the user

# Pre-specified poses (quaternion & position pair) of 3D inspection viewpoints
R = 0.3; Tr = (0, 0, R); theta_H = radians(70); theta_L = radians(50)
H1 = cos(theta_H/2); H2 = sin(theta_H/2); L1 = cos(theta_L/2); L2 = sin(theta_L/2)
## (qw/qx/qy/qz quaternion, tx/ty/tz position); rotation first, then translation
Y_ROTATIONS = [th1+th2 for th1, th2 in product([0, 90, 180, 270], [0, 15])]
VP_POSES = [
    # 'Southern-high' loop
    ((H1*cos(th := radians(th_y / 2)), H2*cos(th), H1*sin(th), H2*sin(th)), Tr)
    for th_y in Y_ROTATIONS
] + [
    # 'Southern-low' loop
    ((L1*cos(th := radians(th_y / 2)), L2*cos(th), L1*sin(th), L2*sin(th)), Tr)
    for th_y in Y_ROTATIONS
] + [
    # 'Equator' loop
    ((cos(th := radians(th_y / 2)), 0, sin(th), 0), Tr)
    for th_y in Y_ROTATIONS
] + [
    # 'Northern-low' loop
    ((L1*cos(th := radians(th_y / 2)), -L2*cos(th), L1*sin(th), -L2*sin(th)), Tr)
    for th_y in Y_ROTATIONS
] + [
    # 'Northern-high' loop
    ((H1*cos(th := radians(th_y / 2)), -H2*cos(th), H1*sin(th), -H2*sin(th)), Tr)
    for th_y in Y_ROTATIONS
]
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
    nonatomic_subassemblies = set(); nl_labeling = {}
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
                                current_held[left_or_right] = target_info["name"]

                                if target_info["name"] not in nonatomic_subassemblies:
                                    # Atomic part type to be remembered, record (image, mask)
                                    # pair by the name index
                                    data = (prev_img, env_refs["o0"]["mask"])
                                    vision_2d_data[target_info["name"]] = [data]

                            case "drop":
                                # drop_~ action
                                lastly_dropped = current_held[left_or_right]
                                current_held[left_or_right] = None

                            case "assemble":
                                # assemble_~ action; record 2d visual data (image + masks)
                                # and assembly action info, then remember non-atomic subassemblies
                                # (distinguished by string name handle)
                                component_masks = {
                                    # Provided ground-truth masks for potentially involved components
                                    referents["dis"][rf_dis]["name"]: env_refs[rf_env]["mask"]
                                    for msk_arg, _ in lit.args[6:]
                                    for rf_dis, rf_env in value_assignment.items()
                                    if rf_dis.startswith(lit.args[0][0]) and msk_arg == rf_env
                                }
                                current_assembly_info = [
                                    (current_held[0], referents["dis"][lit.args[3][0]]["name"]),
                                    (current_held[1], referents["dis"][lit.args[4][0]]["name"]),
                                    prev_img, component_masks, referents["dis"][lit.args[2][0]]["name"]
                                ]
                                # Add visuals of any atomic parts to 2d exemplar lists
                                for obj in current_held:
                                    # Only atomic parts are listed in 2d vision data storage
                                    if obj in vision_2d_data:
                                        vision_2d_data[obj].append((prev_img, component_masks[obj]))
                                # Update hand states
                                current_held[left_or_right] = referents["dis"][lit.args[2][0]]["name"]
                                current_held[(left_or_right+1) % 2] = None

                                nonatomic_subassemblies.add(current_held[left_or_right])

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

                        # Handle each action accordingly (actions 2~9 to be handled
                        # in our scope)
                        act_name = agent.lt_mem.lexicon.d2s[(conc_type, conc_ind)][0][1]
                        act_type = act_name.split("_")[0]
                        left_or_right = 0 if act_name.endswith("left") else 1
                        match act_type:
                            case "assemble":
                                # assemble_~ action; record manipulator poses
                                parse_values = lambda ai: tuple(
                                    float(v) for v in referents["dis"][lit.args[ai][0]]["name"].split("/")
                                )
                                mnp_pose_before = (
                                    xyzw2wxyz(parse_values(2)), parse_values(3)
                                )
                                mnp_pose_after = (
                                    xyzw2wxyz(parse_values(4)), parse_values(5)
                                )
                                current_assembly_info += [
                                    "RToL" if left_or_right==0 else "LToR",
                                    mnp_pose_before, mnp_pose_after
                                ]
                                assembly_sequence.append(tuple(current_assembly_info))

                                current_assembly_info = None

                elif raw.startswith("Pick up a"):
                    # NL description providing labeling of the atomic part just
                    # picked up with a PickUp~ action
                    nl_labeling[current_held[left_or_right]] = \
                        re.findall(r"Pick up a (.*)\.$", raw)[0]

                elif raw.startswith("This is a"):
                    # NL description providing labeling of the subassembly just
                    # placed on the desk with a Drop~ action
                    nl_labeling[lastly_dropped] = \
                        re.findall(r"This is a (.*)\.$", raw)[0]

            else:
                # Quantified statement with antecedent; expecting a generic rule
                # expressed in natural language in the scope of our experiment
                raise NotImplementedError

        if len(inspect_data["img"]) == 40 and len(inspect_data["msk"]) == 40:
            # Reconstruct 3D structure of the inspected object instance
            reconstruction = agent.vision.reconstruct_3d_structure(
                inspect_data["img"], inspect_data["msk"],
                VP_POSES, CON_GRAPH, STORE_VP_INDS,
                # resolution_multiplier=1.25
            )
            vision_3d_data[current_held[left_or_right]] = reconstruction

            # Make way for new data
            inspect_data = { "img": {}, "msk": {} }

        prev_img = img

    # Tag each part instance with their visual concept index, registering any
    # new visual concepts & neologisms; we assume here all neologisms are nouns
    # (corresponding to 'pcls')
    inst2conc_map = {}
    for part_inst, part_type_name in nl_labeling.items():
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

    # Process 3D vision data, storing reconstructed structure data in XB    
    for part_inst, reconstruction in vision_3d_data.items():
        point_cloud, views, descriptors = reconstruction

        # Store the reconstructed structure info in XB
        agent.lt_mem.exemplars.add_exs_3d(
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
            for image, mask in data
        ]
        for part_inst, data in vision_2d_data.items()
    }
    vis_model = agent.vision.model; vis_model.eval()
    with torch.no_grad():
        for part_inst, data in vision_2d_data.items():
            processed_data = []
            for image, mask, bg_image, vis_prompt in data:
                vp_processed = vis_model.dino_processor(images=vis_prompt, return_tensors="pt")
                vp_pixel_values = vp_processed.pixel_values.to(vis_model.dino.device)
                vp_dino_out = vis_model.dino(pixel_values=vp_pixel_values, return_dict=True)
                f_vec = vp_dino_out.pooler_output.cpu().numpy()[0]
                processed_data.append((image, mask, f_vec))
            vision_2d_data[part_inst] = processed_data
    # Add 2D vision data in XB, based on the newly assigned pcls concept indices
    for part_inst, data in vision_2d_data.items():
        if part_inst not in inst2conc_map:
            # Cannot exactly specify which concept this instance classifies as, skip
            continue

        for image, mask, f_vec in data:
            exemplars = [{ "scene_id": None, "mask": mask, "f_vec": f_vec }]
            pointers = {
                ("pcls", inst2conc_map[inst], "pos" if inst==part_inst else "neg"): {
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
    assembly_trees = {}             # Trees representing (sub)assembly structures

    # Shortcut helper for normalizing set of features by L2 norm, so that cosine
    # similarities can be computed
    normalize = lambda fts: fts / norm(fts, axis=1, keepdims=True)

    for assembly_step in assembly_sequence:
        with torch.no_grad():
            # Unpacking assembly step information
            involved_objs = assembly_step[:2]
            image, component_masks, resulting_subassembly = assembly_step[2:5]
            direction, *manip_poses = assembly_step[5:8]
            foo = assembly_step[8:]

            # Select reference masks by which the target components' poses
            # will be estimated; may be the very components themselves, or
            # some other assembly-connected proxy components when the targets
            # are heavily occluded (hence very small mask area)
            selected_components = []; selected_masks = []
            for obj, cp in involved_objs:
                if obj in assembly_trees:
                    # Involved object is a subassembly built in a previous step,
                    # *may* need to select a mask if heavily occluded
                    subassembly = assembly_trees[obj]
                    mask_criteria = {
                        # Tuple of (whether type agrees, mask area); prioritize
                        # type agreement, fall back to next best only if and only if
                        # mask area is too small
                        n: (hdl==cp.split("/")[0], component_masks[n].sum())
                        for n, hdl in subassembly.nodes(data="part_handle")
                    }
                    components_sorted = sorted(
                        subassembly.nodes, reverse=True, key=lambda n: mask_criteria[n]
                    )
                    # Select the top two: The explicitly mentioned part (which may
                    # be heavily occluded), and another part with the largest mask
                    # area among the rest as a fallback in case of heavy occlusion
                    selected = components_sorted[:2]
                else:
                    # Only atomic component part, use its mask
                    selected = [obj]

                selected_components.append(selected)
                selected_masks.append([component_masks[obj] for obj in selected])

            # For each of the lefthand & righthand side
            involved_parts_info = []            # For storing part concepts and poses
            for parts, masks in zip(selected_components, selected_masks):
                pose_estimation_results = []
                for part, msk in zip(parts, masks):
                    # Skip on obvious occlusion, i.e., if mask area of part in
                    # a subassembly is too small
                    if len(masks) > 1 and msk.sum() < 3000:
                        continue

                    # Extract patch-level features as guided by masks
                    zoomed_image, zoomed_msk, crop_dim = crop_images_by_masks(
                        { 0: image }, [msk]
                    )
                    patch_features, lr_msk, lr_dim = vis_model.lr_features_from_masks(
                        zoomed_image, zoomed_msk, 750, 3
                    )
                    D = patch_features[0].shape[-1]
                    (cr_x, _, cr_y, _), (lr_w, lr_h) = crop_dim[0], lr_dim[0]
                    zoomed_image = zoomed_image[0]; lr_msk = lr_msk[0]

                    msk_flattened = lr_msk.reshape(-1)
                    nonzero_inds = msk_flattened.nonzero()[0]

                    # Scale ratio between zoomed image vs. low-res feature map
                    x_ratio = zoomed_image.width / lr_w
                    y_ratio = zoomed_image.height / lr_h

                    # Compare against downsampled point descriptors stored in XB, per each view
                    conc_ind = inst2conc_map[part]
                    points, views, descriptors, _ = agent.lt_mem.exemplars.object_3d[conc_ind]
                    pth_fh_flattened = patch_features[0].cpu().numpy().reshape(-1, D)
                    for vi, view_info in views.items():
                        # Fetch descriptors of visible points
                        visible_pts_sorted = sorted(view_info["visible_points"])
                        visible_pts_features = np.stack(
                            [descriptors[pi][vi] for pi in visible_pts_sorted]
                        )

                        # Compute cosine similarities between patches
                        features_nrm_1 = normalize(view_info["pca"].transform(pth_fh_flattened))
                        features_nrm_2 = normalize(visible_pts_features)
                        S = features_nrm_1 @ features_nrm_2.T

                        # Obtaining (u,v)-coordinates of projected (downsampled) points at the
                        # viewpoint pose, needed for proximity score computation
                        rmat_view = quat2rmat(view_info["cam_quaternion"])
                        tvec_view = view_info["cam_position"]
                        proj_at_view = cv.projectPoints(
                            points,
                            cv.Rodrigues(rmat_view)[0], tvec_view,
                            cam_K, distortion_coeffs
                        )[0][:,0,:]
                        u_min, u_max = proj_at_view[:,0].min(), proj_at_view[:,0].max()
                        v_min, v_max = proj_at_view[:,1].min(), proj_at_view[:,1].max()
                        proj_w = u_max - u_min; proj_h = v_max - v_min
                        # Scale and align point coordinates
                        proj_aligned = proj_at_view[visible_pts_sorted]
                        proj_aligned[:,0] -= u_min; proj_aligned[:,1] -= v_min
                        # Scale so that the bounding box would encase the provided object
                        # mask (object may be occluded while projection is never occluded)
                        obj_box = masks_bounding_boxes([msk])[0]
                        obj_w = obj_box[2] - obj_box[0]; obj_h = obj_box[3] - obj_box[1]
                        obj_cu = (obj_box[0]+obj_box[2]) / 2; obj_cv = (obj_box[3]+obj_box[1]) / 2
                        scale_ratio = min(proj_w / obj_w, proj_h / obj_h)
                        proj_aligned /= scale_ratio
                        # Align by box center coordinates
                        proj_aligned[:,0] += obj_cu - (proj_w / 2) / scale_ratio
                        proj_aligned[:,1] += obj_cv - (proj_h / 2) / scale_ratio

                        # Proximity scores (w.r.t. to downsampled point projection) computed
                        # with RBF kernel; for giving slight advantages to pixels close to
                        # initial guess projections
                        uv_coords = np.stack([
                            np.tile((np.arange(lr_w)*x_ratio + cr_x)[None], [lr_h, 1]),
                            np.tile((np.arange(lr_h)*y_ratio + cr_y)[:,None], [1, lr_w])
                        ], axis=-1)
                        sigma = min(zoomed_image.width, zoomed_image.height)
                        proximity = norm(
                            uv_coords[:,:,None] - proj_aligned[None,None], axis=-1
                        )
                        proximity = np.exp(-np.square(proximity) / (2 * (sigma ** 2)))

                        # Forward matching
                        agg_scores = S + 0.4 * proximity.reshape(-1, len(visible_pts_sorted))
                        match_forward = linear_sum_assignment(
                            agg_scores[msk_flattened], maximize=True
                        )

                        points_2d = [
                            (i % lr_w, i // lr_w)
                            for i in nonzero_inds[match_forward[0]]
                        ]
                        points_2d = np.array([
                            (cr_x + lr_x * x_ratio, cr_y + lr_y * y_ratio)
                            for lr_x, lr_y in points_2d
                        ])
                        points_3d = np.array([
                            points[visible_pts_sorted[i]]
                            for i in match_forward[1]
                        ])

                        # Pose estimation by PnP with USAC (MAGSAC)
                        output_valid, rvec, tvec, _ = cv.solvePnPRansac(
                            points_3d, points_2d, cam_K, distortion_coeffs,
                            flags=cv.USAC_MAGSAC
                        )
                        assert output_valid

                        # Evaluate estimated pose by obtaining mean similarity scores at
                        # reprojected downsampled points
                        estim_reprojections = cv.projectPoints(
                            points, rvec, tvec, cam_K, distortion_coeffs
                        )[0][:,0,:]
                        visible_reprojections = np.array([
                            estim_reprojections[visible_pts_sorted[i]]
                            for i in match_forward[1]
                        ])
                        dists_to_reprojs = norm(
                            uv_coords[:,:,None] - visible_reprojections[None,None], axis=-1
                        )
                        dists_to_reprojs = dists_to_reprojs.reshape(-1, len(visible_reprojections))
                        reproj_coords = np.unravel_index(
                            dists_to_reprojs.argmin(axis=0), (lr_h, lr_w)
                        )
                        reproj_score_inds = reproj_coords + (np.arange(len(visible_pts_sorted)),)
                        reproj_scores = S.reshape(lr_h, lr_w, -1)[reproj_score_inds]
                        # Only consider points that belong to the primary cluster; accounts
                        # for occlusions
                        within_mask_inds = [
                            lr_msk[(reproj_coords[0][i], reproj_coords[1][i])]
                            for i in range(len(visible_pts_sorted))
                        ]
                        reproj_scores = reproj_scores[within_mask_inds]

                        # Also evaluate by overlap between provided object mask vs. mask made
                        # from reprojected visible points dilated with disk-shaped footprints.
                        # Size of disk determined by mean nearest distances between reprojected
                        # visible points.
                        vps_prm_reproj = estim_reprojections[visible_pts_sorted]
                        pdists = pairwise_distances(vps_prm_reproj, vps_prm_reproj)
                        pdists[np.diag_indices(len(pdists))] = float("inf")
                        median_nn_dist = np.median(pdists.min(axis=0))
                            # Occasionally some reprojected points are placed very far
                            # from the rest... Using median instead of mean to eliminate
                            # effects by outliers
                        proj_x, proj_y = vps_prm_reproj.round().astype(np.int64).T
                        proj_msk = np.zeros_like(msk)
                        valid_inds = valid_inds = np.logical_and(
                            np.logical_and(0 <= proj_y, proj_y < msk.shape[0]),
                            np.logical_and(0 <= proj_x, proj_x < msk.shape[1])
                        )
                        proj_msk[proj_y[valid_inds], proj_x[valid_inds]] = True
                        disk_size = min(
                            max(2*median_nn_dist, 1), min(obj_w, obj_h) / 10
                        )       # Max cap disk size with 0.1 * min(mask_w, mask_h)
                        proj_msk = dilation(proj_msk, footprint=disk(disk_size))
                        mask_overlap = mask_iou([msk], [proj_msk])[0][0]

                        # Store pose estimation results along with pose evaluation score
                        pose_estimation_results.append(
                            (
                                (rvec, tvec), conc_ind, part,
                                (reproj_scores.mean().item()+1) / 2, mask_overlap
                            )
                        )

                # Select the best pose estimation result with highest score
                best_estimation = sorted(
                    pose_estimation_results, key=lambda x: x[3]+x[4], reverse=True
                )[0]
                involved_parts_info.append(
                    (transformation_matrix(*best_estimation[0]),)+best_estimation[1:3]
                )

        # Resolve any (chain of) indirect pose estimation via connected parts
        assert len(involved_parts_info) == 2
        for side in [0, 1]:         # Left (0) or right (1)
            obj, cp = involved_objs[side]
            if obj in assembly_trees:
                subassembly = assembly_trees[obj]
                part_estimated = involved_parts_info[side][2]
                part_goal = [
                    n for n, hdl in subassembly.nodes(data="part_handle")
                    if hdl == cp.split("/")[0]
                ][0]
                if part_estimated != part_goal:
                    # Get a shortest connection path
                    connection_path = next(nx.shortest_simple_paths(
                        subassembly, part_estimated, part_goal
                    ))
                    u = connection_path.pop(0); v = connection_path.pop(0)
                    tmat_u = involved_parts_info[side][0]
                    while True:
                        part_u = subassembly.nodes[u]["part_conc"]
                        part_v = subassembly.nodes[v]["part_conc"]
                        _, cp_u, i_u = subassembly.edges[(u, v)]["contact"][u]
                        _, cp_v, i_v = subassembly.edges[(u, v)]["contact"][v]
                        pose_cp_u = agent.lt_mem.exemplars.object_3d[part_u][3][cp_u][i_u]
                        pose_cp_v = agent.lt_mem.exemplars.object_3d[part_v][3][cp_v][i_v]
                        tmat_cp_u = transformation_matrix(*pose_cp_u)
                        tmat_cp_v = transformation_matrix(*pose_cp_v)
                        tmat_v = tmat_u @ tmat_cp_u @ inv(tmat_cp_v)

                        # Update component concept and transformation info
                        involved_parts_info[side] = \
                            (tmat_v, part_v) + involved_parts_info[side][2:]

                        # Break if reached end of chain
                        if len(connection_path) == 0: break

                        # Next chain link
                        u = v; tmat_u = tmat_v; v = connection_path.pop(0)

        tmat_l, part_l_conc_ind, _ = involved_parts_info[0]
        tmat_r, part_r_conc_ind, _ = involved_parts_info[1]

        # Infer contact point from the estimated object poses and manipulator
        # movement
        if direction == "RToL":
            tmat_tgt_obj = tmat_l; tmat_src_obj_before = tmat_r
            tgt_conc_ind = part_l_conc_ind; src_conc_ind = part_r_conc_ind
            pcl_tgt = agent.lt_mem.exemplars.object_3d[part_l_conc_ind][0]
            pcl_src = agent.lt_mem.exemplars.object_3d[part_r_conc_ind][0]
        else:
            tmat_tgt_obj = tmat_r; tmat_src_obj_before = tmat_l
            tgt_conc_ind = part_r_conc_ind; src_conc_ind = part_l_conc_ind
            pcl_tgt = agent.lt_mem.exemplars.object_3d[part_r_conc_ind][0]
            pcl_src = agent.lt_mem.exemplars.object_3d[part_l_conc_ind][0]
        pcl_tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl_tgt))
        pcl_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl_src))
        pose_src_manip_before = (
            flip_quaternion_y(manip_poses[0][0]),
            flip_position_y(manip_poses[0][1])
        )
        pose_src_manip_after = (
            flip_quaternion_y(manip_poses[1][0]),
            flip_position_y(manip_poses[1][1])
        )

        # Relation among manipulator & object transformations:
        # [Tr. of src object after movement] = 
        #   [Tr. of src manipulator after movement] *
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
        points_cat = np.concatenate([pcl_tgt.points, pcl_src.points])

        # Position of contact point: weighted centroid of points in both clouds,
        # weighted by distances to nearest points in each other
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

        # Assigning new concept indices as needed
        cp_l_conc_ind = agent.vision.add_concept("pcls")
        cp_r_conc_ind = agent.vision.add_concept("pcls")
        cp_tgt_conc_ind = cp_l_conc_ind if direction == "RToL" else cp_r_conc_ind
        cp_src_conc_ind = cp_r_conc_ind if direction == "RToL" else cp_l_conc_ind

        # Register contact point type and pose info
        agent.lt_mem.exemplars.add_exs_3d(
            tgt_conc_ind, None, None, None, { cp_tgt_conc_ind: [pose_cp_tgt] }
        )
        agent.lt_mem.exemplars.add_exs_3d(
            src_conc_ind, None, None, None, { cp_src_conc_ind: [pose_cp_src] }
        )

        # Add new subassembly tree by adding concept-annotated part nodes and
        # connecting them with contact-annotated edges
        # (One ugly assumption made here is that contact points can be uniquely
        # specified by string name handle used in Unity)
        (obj_l, cp_l), (obj_r, cp_r) = involved_objs
        if obj_l in assembly_trees:
            subassembly_l = assembly_trees.pop(obj_l)
        else:
            subassembly_l = nx.Graph()
            subassembly_l.add_node(
                obj_l, part_conc=part_l_conc_ind, part_handle=cp_l.split("/")[0]
            )
        if obj_r in assembly_trees:
            subassembly_r = assembly_trees.pop(obj_r)
        else:
            subassembly_r = nx.Graph()
            subassembly_r.add_node(
                obj_r, part_conc=part_r_conc_ind, part_handle=cp_r.split("/")[0]
            )

        assembly_trees[resulting_subassembly] = nx.union(subassembly_l, subassembly_r)
        part_node_l = [
            n for n in subassembly_l.nodes
            if subassembly_l.nodes[n]["part_handle"] in cp_l
        ][0]
        part_node_r = [
            n for n in subassembly_r.nodes
            if subassembly_r.nodes[n]["part_handle"] in cp_r
        ][0]
        assembly_trees[resulting_subassembly].add_edge(
            part_node_l, part_node_r,
            contact={
                part_node_l: (None, cp_l_conc_ind, 0),
                part_node_r: (None, cp_r_conc_ind, 0)
            }
        )

    for subassembly_name, tree in assembly_trees.items():
        # Create a new assembly graph, relabeling node names into case-neutral
        # indices
        neutral_tree = nx.Graph()

        # Assign arbitrary integer ordering to atomic parts in subassemblies
        node_reindex = list(tree.nodes)

        # Accordingly neutralize node and edge name & data
        for n, n_data in tree.nodes(data=True):
            n_ind = node_reindex.index(n)
            n_data_new = {
                "node_type": "atomic_part",
                "parts": {n_data["part_conc"]}
            }
            neutral_tree.add_node(n_ind, **n_data_new)
        for n1, n2, e_data in tree.edges(data=True):
            n1_ind, n2_ind = (node_reindex.index(n1), node_reindex.index(n2))
            e_data_new = {
                "contact": {
                    n1_ind: [e_data["contact"][n1]],
                    n2_ind: [e_data["contact"][n2]]
                }
            }
            neutral_tree.add_edge(n1_ind, n2_ind, **e_data_new)

        # Parse subassembly concept type & index
        sa_conc = agent.lt_mem.lexicon.s2d[("n", nl_labeling[subassembly_name])][0]
        sa_conc = (sa_conc[0], int(sa_conc[1]))

        # Store the structure in KB
        agent.lt_mem.kb.add_structure(sa_conc, neutral_tree)


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
