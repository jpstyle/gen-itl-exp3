"""
Implements learning-related composite actions; by learning, we are primarily
referring to belief updates that modify long-term memory
"""
import re
import math
import torch
from collections import defaultdict, Counter, deque

import open3d as o3d
import numpy as np
import networkx as nx
from tqdm import tqdm
from numpy.linalg import norm, inv
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from .constants import *
from .interact import (
    _goal_selection, _tabulate_goal_selection_result,
    _match_existing_subassemblies
)
from .utils import rle_decode
from ..vision.utils import (
    blur_and_grayscale, visual_prompt_by_mask, rmat2quat, xyzw2wxyz,
    flip_position_y, flip_quaternion_y, transformation_matrix
)
from ..lpmln import Literal
from ..lpmln.utils import flatten_ante_cons, wrap_args


def identify_mismatch(agent, statement):
    """
    Test against vision-only sensemaking result to identify any mismatch btw.
    agent's & user's perception of world state. If any significant mismatch
    (as determined by surprisal) is identified, handle it by updating agent's
    exemplar base. Return bool indicating whether mismatch is identified and
    exemplar base is accordingly updated.
    """
    xb_updated = False          # Return value

    gq, bvars, ante, cons = statement

    # The code snippet commented out below is the legacy from the first and
    # second chapter where inference of whole from parts played important roles.
    # However, we don't do any part-based visual reference in this chapter,
    # and it is pointless to undergo all those belief propagation, region graph
    # querying kind of deals only to get to the same conclusion obtainable from
    # just simply reading the corresponding probability scores off the chart
    # in scene info. I will just take the simple shortcut for this chapter.
    ####### OBSOLETE #######
    # # Proceed only when we have a grounded event & vision-only sensemaking
    # # result has been obtained
    # stm_is_grounded = (len(ante) == 0 or len(cons) == 0) and \
    #     not any(lit.name=="sp_subtype" for lit in cons)
    # if not stm_is_grounded or agent.symbolic.concl_vis is None:
    #     return False
    #
    # # Make a yes/no query to obtain the likelihood of content
    # reg_gr_v, _ = agent.symbolic.concl_vis
    # q_response, _ = agent.symbolic.query(reg_gr_v, None, statement)
    # ev_prob = q_response[()]
    ####### OBSOLETE #######

    if len(ante) == 0 and len(cons) == 1:
        # Positive labeling of an instance
        atom = cons[0]
        conc_type, conc_ind = atom.name.split("_")
        conc_ind = int(conc_ind)
        obj = atom.args[0][0]
        if conc_ind in range(len(agent.vision.scene[obj]["pred_cls"])):
            ev_prob = agent.vision.scene[obj]["pred_cls"][conc_ind]
        else:
            ev_prob = 0
    elif len(ante) == 1 and len(cons) == 0:
        # Negative labeling of an instance
        atom = ante[0]
        conc_type, conc_ind = atom.name.split("_")
        conc_ind = int(conc_ind)
        obj = atom.args[0][0]
        if conc_ind in range(len(agent.vision.scene[obj]["pred_cls"])):
            ev_prob = 1 - agent.vision.scene[obj]["pred_cls"][conc_ind]
        else:
            ev_prob = 1
    else:
        # Not interested in any other types of statements for now
        return False

    # Proceed only if surprisal is above threshold
    surprisal = -math.log(ev_prob + EPS)
    if surprisal < -math.log(SR_THRES):
        return False

    for ante, cons in flatten_ante_cons(*statement[2:]):
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
                if all(obj["exemplar_ind"] is None for obj in agent.vision.scene.values()):
                    # Totally new visual scene; both scene and object should be added
                    scene_id = None
                    scene_img = agent.vision.scene[ex_obj]["scene_img"]
                    pointer = ((conc_type, conc_ind, pol), (True, 0))
                else:
                    # Scene already registered in XB, fetch scene id
                    scene_id = next(
                        obj["exemplar_ind"][0] for obj in agent.vision.scene.values()
                        if obj["exemplar_ind"] is not None
                    )
                    scene_img = None
                    if agent.vision.scene[ex_obj]["exemplar_ind"] is None:
                        # New object for existing scene, object should be added
                        pointer = ((conc_type, conc_ind, pol), (True, 0))
                    else:
                        # Exemplar present in storage, only add pointer
                        ex_ind = agent.vision.scene[ex_obj]["exemplar_ind"]

                        # First check if this exemplar info is already present in XB
                        # for this particular concept-polarity descriptor and thus
                        # redundant
                        ex_mem = agent.lt_mem.exemplars
                        redundant = False
                        match pol:
                            case "pos":
                                ex_exs = ex_mem.object_2d_pos[conc_type][conc_ind]
                                redundant = ex_ind in ex_exs
                            case "neg":
                                ex_exs = ex_mem.object_2d_neg[conc_type][conc_ind]
                                redundant = ex_ind in ex_exs
                        if redundant:
                            # No need to update XB, continue to next iteration
                            continue
                        else:
                            pointer = ((conc_type, conc_ind, pol), (False, ex_ind))

                added_xi = _add_scene_and_exemplar_2d(
                    scene_id, scene_img, pointer,
                    agent.vision.scene[ex_obj]["pred_mask"],
                    agent.vision.scene[ex_obj]["vis_emb"],
                    agent.lt_mem.exemplars
                )
                if agent.vision.scene[ex_obj]["exemplar_ind"] is None:
                    assert added_xi is not None
                    agent.vision.scene[ex_obj]["exemplar_ind"] = added_xi
                xb_updated = True

            case "prel":
                raise NotImplementedError   # Step back for relation prediction...

            case _:
                raise ValueError("Invalid concept type")

    return xb_updated


def identify_generics(agent, statement, provenance):
    """
    For symbolic knowledge base expansion. Integrate the rule into KB by adding
    as new entry if it is not already included. For now we won't worry about
    intra-KB consistency, belief revision, etc. Return bool indicating whether
    the statement is identified as a novel generic rule, and KB is accordingly
    updated.
    """
    gq, bvars, ante, cons = statement

    # Only process statements that could be interpreted as a generic rule;
    # in our scope, taxonomy relations (identified by "sp_subtype" literal)
    # or general properties (identified by "forall" quantifiers)
    taxonomy_rel = any(lit.name=="sp_subtype" for lit in cons)
    general_prop = "forall" in gq
    if not (taxonomy_rel or general_prop): return False

    kb_updated = False          # Return value

    # Translate the provided statement into a logical form to be stored in KB;
    # simple pattern matching for the scope of our study
    if taxonomy_rel:
        (subtype, _), (supertype, _) = next(
            lit for lit in cons if lit.name=="sp_subtype"
        ).args[1:]
        subtype = next(lit.name for lit in cons if lit.args[0][0] == subtype)
        supertype = next(lit.name for lit in cons if lit.args[0][0] == supertype)
        kb_updated |= agent.lt_mem.kb.add(
            (
                [Literal(subtype, wrap_args("X"))],
                [Literal(supertype, wrap_args("X"))]
            ),
            U_IN_PR, provenance, "taxonomy"
        )

    if general_prop:
        univ_q_vars = [v for q, v in zip(gq, bvars) if q == "forall"]
        exst_q_vars = [v for q, v in zip(gq, bvars) if q == "exists"]
        # Re-assign variable names to universally quantified vars first
        variable_renaming = { v: f"X{i}" for i, v in enumerate(univ_q_vars) }
        # Then assign skolem function terms to existentially quantified ones,
        # where function args are determined by any binary predicate involving
        # such var together with some universally quantified (hence already
        # re-named) var
        variable_renaming |= {
            v: (f"f{i}", (next(
                variable_renaming[list(args - {v})[0]]
                for lit in cons if v in (args := {a for a, _ in lit.args}) \
                    and len(args - {v}) == 1
            ),))
            for i, v in enumerate(exst_q_vars)
        }
        variable_renaming = {
            (v, False): (v_rn, True) for v, v_rn in variable_renaming.items()
        }
        kb_updated |= agent.lt_mem.kb.add(
            (
                [lit.substitute(variable_renaming) for lit in ante],
                [lit.substitute(variable_renaming) for lit in cons]
            ),
            U_IN_PR, provenance, "property"
        )

    return kb_updated


def handle_neologism(agent, dialogue_state):
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
        # Skip if already processed somehow
        if sym in agent.lt_mem.lexicon: continue

        # Flag whether this neologism can be immediately resolved within
        # this method call
        resolved = False

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

        # Acquire novel concept by updating lexicon
        agent.lt_mem.lexicon.add((pos, name), novel_concept)

        # Update word sense resolution accordingly
        agent.lang.dialogue.word_senses[tok] = (
            sym, f"{novel_concept[0]}_{novel_concept[1]}"
        )

        neologism_in_cons = tok[2].startswith("pc")
        neologisms_in_same_clause_ante = [
            n for n in neologisms
            if tok[:2]==n[:2] and n[2]==tok[2].replace("pc", "pa")
        ]
        if neologism_in_cons and len(neologisms_in_same_clause_ante) == 0:
            # Occurrence in rule cons implies either definition or exemplar is
            # provided by the utterance containing this token... only if the
            # source clause is in indicative mood. If that's the case, register
            # new visual concept, and perform few-shot learning if appropriate
            ti = int(tok[0].strip("t"))
            ci = int(tok[1].strip("c"))
            (_, _, ante, cons), _ = dialogue_state.record[ti][1][ci]

            mood = dialogue_state.clause_info[f"t{ti}c{ci}"]["mood"]
            if len(cons) == 0 and len(ante) == 0 and mood == ".":
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

                resolved = True

        if not resolved:
            agent.lang.unresolved_neologisms[sym] = {
                "dependency": None,     # Dependency not specified yet
                "reported": False       # Not reported to user yet
            }

    if len(objs_to_add) > 0 or len(pointers) > 0:
        objs_to_add = list(objs_to_add)         # Assign arbitrary ordering
        _add_scene_and_exemplars_2d(
            objs_to_add, pointers,
            agent.vision.scene, agent.vision.latest_inputs[-1], agent.lt_mem.exemplars
        )

    return xb_updated


def resolve_neologisms(agent):
    """
    Inspect the current set of unresolved neologisms to see if they can be
    treated as resolved at the moment. In particular, this method checks for
    each currently unresolved neologism if the 'depedency of resolution' is
    specified yet, and if so, whether the dependency is cleared. The dependency
    of an unresolved neologism is specified when the denoted concept is properly
    defined either extensionally (via exemplars) or intensionally (via relational
    definitions).
    """
    # Needed for potential registration of 3D structure inspection actions
    resolved_record = agent.lang.dialogue.export_resolved_record()
    exec_state = agent.planner.execution_state

    while True:
        resolved = set()
        for sym, info in agent.lang.unresolved_neologisms.items():
            den = agent.lt_mem.lexicon.s2d[sym][0]
            den_str = f"{den[0]}_{den[1]}"

            # See if any positive exemplar of the denoted concept is provided.
            # If so, this neologism can be immediately considered as resolved.
            pos_exs = agent.lt_mem.exemplars.object_2d_pos[den[0]].get(den[1], set())
            if len(pos_exs) > 0:
                resolved.add(sym)

                # Whenever a neologism is resolved by means of exemplification,
                # need to obtain and register its 3D structure. Add a series of
                # 3D inspection actions to the agenda deque, with the labeled
                # instance as inspection target.
                labeling_lit = next(
                    cons[0]
                    for speaker, turn_clauses in resolved_record
                    for (_, _, ante, cons), _, _ in turn_clauses
                    if len(ante) == 0 and len(cons) == 1 and \
                        speaker == "Teacher" and cons[0].name == den_str
                )
                inspection_target = labeling_lit.args[0][0]
                for side, mnp_state in enumerate(exec_state["manipulator_states"]):
                    if mnp_state[0] is None:
                        empty_side = side
                        break
                else:
                    # At least one manipulator should be free by design...
                    raise ValueError
                empty_side = "left" if empty_side == 0 else "right"
                pick_up_action = agent.lt_mem.lexicon.s2d[("va", f"pick_up_{empty_side}")][0][1]
                inspect_action = agent.lt_mem.lexicon.s2d[("va", f"inspect_{empty_side}")][0][1]
                drop_action = agent.lt_mem.lexicon.s2d[("va", f"drop_{empty_side}")][0][1]
                inspection_plan = [(pick_up_action, (inspection_target,))] + [
                    (inspect_action, (inspection_target, i, den))
                    for i in range(24+1)
                ] + [(drop_action, ())]
                inspection_plan = [
                    ("execute_command", (action_type, action_params))
                    for action_type, action_params in inspection_plan
                ]
                agent.planner.agenda = deque(inspection_plan) + agent.planner.agenda
                continue

            if info["dependency"] is None:
                # Extract resolution dependency from current knowledge state

                # See if any definitions---characterizing properties---are stored
                # in KB. If any referenced concepts are denoted by other unresolved
                # neologisms, add those to the dependency.
                relevant_kb_entries = agent.lt_mem.kb.entries_by_pred[den_str]
                if len(relevant_kb_entries) > 0:
                    dependent_neologisms = set()
                    for ei in agent.lt_mem.kb.entries_by_pred[den_str]:
                        (ante, cons), _, _, knowledge_type = agent.lt_mem.kb.entries[ei]
                        if knowledge_type != "property": continue
                        relevant_concs = {
                            ((pred_spl := lit.name.split("_"))[0], int(pred_spl[1]))
                            for lit in ante+cons
                        }
                        dependent_neologisms |= {
                            rel_sym for conc in relevant_concs
                            if (rel_sym := agent.lt_mem.lexicon.d2s[conc][0]) != sym \
                                and rel_sym in agent.lang.unresolved_neologisms
                        }
                    info["dependency"] = dependent_neologisms
            else:
                # Check if any dependent neologism in the specified set is
                # resolved; if so, remove them from set. If the set becomes
                # empty, the neologism can be marked as resolved.
                if len(info["dependency"]) == 0:
                    resolved.add(sym)

        if len(resolved) == 0:
            # No more neologisms to resolve
            break
        else:
            # Update set accordingly, deleting the resolved neologism itself
            # and updating any other relevant resolution dependencies
            for sym in resolved:
                del agent.lang.unresolved_neologisms[sym]
                for info in agent.lang.unresolved_neologisms.values():
                    if not isinstance(info["dependency"], set): continue
                    info["dependency"] -= {sym}


def report_neologism(agent, neologism):
    """
    Some neologism was identified and couldn't be resolved with available information
    in ongoing context; report lack of knowledge, so that user can provide appropriate
    information that characterize the concept denoted by the neologism (e.g., definition,
    exemplar)
    """
    if agent.lang.unresolved_neologisms[neologism]["reported"]:
        # No-op if already reported
        return

    # Update cognitive state w.r.t. value assignment and word sense

    # NL surface form and corresponding logical form
    surface_form = f"I don't know what '{neologism[1]}' means."
    gq = (); bvars = (); ante = []
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

    # Mark as reported
    agent.lang.unresolved_neologisms[neologism]["reported"] = True


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

    inspect_data = { "img": {}, "msk": {}, "pose": {} }
        # Buffer of 3d object instance views from inspect_~ actions

    # Sequentially process each demonstration step
    current_held = [None, None]; current_assembly_info = None
    nonatomic_subassemblies = set()
    part_supertype_labeling = {}; part_subtype_labeling = {}
    sa_labeling = {}; hyp_rels = {}
    vision_2d_data = defaultdict(list); vision_3d_data = {}; assembly_sequence = []
    # Collecting scene image and masks at the initial setting, fecthed and organized
    # by env_handle fields in agent.vision.scene
    for oi, obj in agent.vision.scene.items():
        vision_2d_data[obj["env_handle"]].append((
            obj["scene_img"], obj["pred_mask"]
        ))

    for img, annotations in demo_data:
        # Appropriately handle each annotation
        for (_, _, ante, cons), raw, clause_info in annotations:
            # Nothing to do with non-indicative annotations
            if clause_info["mood"] != ".": continue
            # Nothing to do with initial declaration of demonstration
            if raw.startswith("I will demonstrate"): continue

            if len(ante) == 0:
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
                                # pick_up_~ action; track the held object
                                target_info = [
                                    rf_dis for rf_dis, rf_env in value_assignment.items()
                                    if rf_dis.startswith(lit.args[0][0]) and rf_env == lit.args[2][0]
                                ][0]
                                target_info = referents["dis"][target_info]
                                current_held[left_or_right] = (target_info["name"], {})

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
                                # inspect_~ action; collect all views for 3D reconstruction.
                                # Image at "# Action", mask and pose at "# Effect" (below)
                                view_ind = int(referents["dis"][lit.args[3][0]]["name"])
                                if view_ind < 24:
                                    inspect_data["img"][view_ind] = img

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
                                if view_ind > 0:
                                    raw_mask = np.array(rle_decode(
                                        [int(v) for v in referents["dis"][lit.args[4][0]]["name"].split("/")]
                                    ))
                                    raw_mask = raw_mask.reshape(
                                        inspect_data["img"][0].height, inspect_data["img"][0].width
                                    ).astype(bool)
                                    inspect_data["msk"][view_ind-1] = raw_mask
                                if view_ind < 24:
                                    inspect_data["pose"][view_ind] = (
                                        flip_quaternion_y(xyzw2wxyz(parse_floats(2))),
                                        flip_position_y(parse_floats(3))
                                    )

                elif raw.startswith("Pick up this"):
                    # NL description providing labeling of the atomic (sub)type
                    # just picked up with a pick_up~ action
                    part_subtype_labeling[current_held[left_or_right][0]] = \
                        re.findall(r"Pick up this (.*)\.$", raw)[0]

                elif raw.startswith("Join "):
                    # NL description providing (super)type requirement for an
                    # atomic part slot in target structure, just partially
                    # fulfilled with an assemble~ action
                    _, target_l, _, target_r = \
                        re.findall(r"Join (a|the) (.*) and (a|the) (.*)\.$", raw)[0]
                    held_l = current_assembly_info[0][0][0]
                    held_r = current_assembly_info[1][0][0]
                    if held_l not in nonatomic_subassemblies:
                        part_supertype_labeling[held_l] = target_l
                    if held_r not in nonatomic_subassemblies:
                        part_supertype_labeling[held_r] = target_r

                elif raw.startswith("This is a"):
                    # NL description providing labeling of the subassembly just
                    # placed on the desk with a drop~ action
                    sa_labeling[lastly_dropped] = \
                        re.findall(r"This is a (.*)\.$", raw)[0]

                elif any(lit.name == "sp_subtype" for lit in cons):
                    # NL description of a hypernymy-hyponymy relation
                    part_subtype, part_supertype = \
                        re.findall(r"(.*) is a type of (.*)\.$", raw)[0]
                    hyp_rels[part_subtype] = (part_supertype, raw)

            else:
                # Quantified statement with antecedent; expecting a generic rule
                # expressed in natural language in the scope of our experiment
                raise NotImplementedError

        if len(inspect_data["img"]) == 24 and len(inspect_data["msk"]) == 24:
            inst_inspected = current_held[left_or_right][0]

            # Collect 3D & 2D visual data for later processing
            vision_3d_data[inst_inspected] = (
                inspect_data["img"], inspect_data["msk"], inspect_data["pose"]
            )
            # Add select examples to 2D classification data as well
            vision_2d_data[inst_inspected] += [
                (inspect_data["img"][view_ind], inspect_data["msk"][view_ind])
                for view_ind in STORE_VP_INDS[4:8]
            ]

            # Make way for new data
            inspect_data = { "img": {}, "msk": {}, "pose": {} }

    # Reconstruct 3D structure of the inspected object instances
    v3d_it = tqdm(
        vision_3d_data.items(), desc="Extracting 3D structure", leave=False
    )
    for inst, (data_img, data_msk, data_pose) in v3d_it:
        vision_3d_data[inst] = agent.vision.reconstruct_3d_structure(
            data_img, data_msk, data_pose, CON_GRAPH, STORE_VP_INDS
        )

    # Tag each part instance with their visual concept index, registering any
    # new visual concepts & neologisms; we assume here all neologisms are nouns
    # (corresponding to 'pcls')
    inst2conc_map = {}; conc_supertypes = {}
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
        for part_inst, part_subtype_name in part_subtype_labeling.items():
            sym = ("n", part_subtype_name)
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
                del agent.lang.unresolved_neologisms[sym]
        # Also process any remaining neologisms denoting hypernyms and holonyms
        for part_supertype_name in part_supertype_labeling.values():
            sym = ("n", part_supertype_name)
            if sym not in agent.lt_mem.lexicon:
                new_conc_ind = agent.vision.add_concept("pcls")
                agent.lt_mem.lexicon.add(sym, ("pcls", new_conc_ind))
            if sym in agent.lang.unresolved_neologisms:
                del agent.lang.unresolved_neologisms[sym]
        for sa_type_name in sa_labeling.values():
            sym = ("n", sa_type_name)
            if sym not in agent.lt_mem.lexicon:
                new_conc_ind = agent.vision.add_concept("pcls")
                agent.lt_mem.lexicon.add(sym, ("pcls", new_conc_ind))
            if sym in agent.lang.unresolved_neologisms:
                del agent.lang.unresolved_neologisms[sym]
        # Also process any hyper/hyponymy relations provided
        for part_subtype, (part_supertype, raw) in hyp_rels.items():
            subtype_conc = agent.lt_mem.lexicon.s2d[("n", part_subtype)][0][1]
            supertype_conc = agent.lt_mem.lexicon.s2d[("n", part_supertype)][0][1]
            agent.lt_mem.kb.add(
                (
                    [Literal(f"pcls_{subtype_conc}", wrap_args("X"))],
                    [Literal(f"pcls_{supertype_conc}", wrap_args("X"))]
                ),
                U_IN_PR, raw, "taxonomy"
            )
            conc_supertypes[subtype_conc] = supertype_conc

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
        v2d_it = tqdm(
            vision_2d_data.items(), desc="Extracting 2D features", leave=False
        )
        for part_inst, examples in v2d_it:
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
            # player types)
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
    ass_it = tqdm(
        assembly_sequence, desc="Inferring 3D contacts", leave=False
    )
    for assembly_step in ass_it:
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
            if obj in part_subtype_labeling or obj in vision_2d_data:
                # Atomic part if listed in part_subtype_labeling (languageful
                # player types) or vision_2d_data (fallback for languageless
                # player types); create a new node
                sa_graph = nx.Graph()
                sa_graph.add_node(
                    obj, node_type="atomic",
                    conc=conc_supertypes.get(part_conc_ind, part_conc_ind),
                        # Specify slot part types at supertype level (unless
                        # there's no applicable hierarchy; e.g. bolt)
                    part_paths={ part_inst: part_inst }
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
        for lit in cons
        if len(cons) > 0 and spk == "Teacher" and clause_info["mood"] == "." \
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
    connection_graph, node_unifications = tabulated_results[3:5]
    atomic_node_concs, _, hyp_rels = tabulated_results[5:]

    # As done in `.interact._plan_assembly` procedure, we could try to further
    # narrow down possible unification choices for the unbounded nodes
    final_sa = next(
        sa for sa, sa_graph in exec_state["connection_graphs"].items()
        if len(sa_graph) > 1
    )
    obj2sa_map = {
        ex_obj: final_sa for ex_obj in exec_state["connection_graphs"][final_sa]
    }
    possible_mappings = _match_existing_subassemblies(
        connection_graph, obj2sa_map, node_unifications,
        atomic_node_concs, hyp_rels, exec_state
    )
    for ism in possible_mappings[final_sa]:
        for n, ex_obj in ism.items():
            node_unifications[ex_obj].add(n)

    # Collect possible atomic part types for each existing object
    obj_possible_concs = defaultdict(set)
    for oi, nodes in node_unifications.items():
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

        # Need exact subtype recognition for getting probability scores,
        # select best ones according to specified supertypes
        scn_subtype = {
            oi: max([
                subtype_conc for subtype_conc, supertype_conc in hyp_rels.items()
                if supertype_conc == conc
            ], key=lambda c: agent.vision.scene[oi]["pred_cls"][c])
                if conc in hyp_rels.values() else conc
            for oi, conc in scn.items()
        }
        prob_score = sum(
            agent.vision.scene[oi]["pred_cls"][c] for oi, c in scn_subtype.items()
        )
        scenarios_scored.append((scn_subtype, prob_score.item()))

    # Select the best allocation of objects to nodes by score sum    
    final_recognitions = sorted(
        scenarios_scored, reverse=True, key=lambda x: x[1]
    )[0][0]

    pr_thres = 0.7                  # Threshold for adding exemplars
    exemplars = {}; pointers = defaultdict(set)
    for oi, best_conc in final_recognitions.items():
        # Skip if supertype
        if best_conc in hyp_rels.values(): continue

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

    # Running below only when we have exemplars to add
    if len(pointers) > 0:
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

def _add_scene_and_exemplar_2d(scene_id, scene_img, pointer, mask, f_vec, ex_mem):
    """
    Helper method factored out for adding a scene, object and/or concept exemplar
    pointer
    """
    exemplar_spec, (is_new_obj, exemplar_id) = pointer

    if scene_id is None:
        # New scene img, image should be registered in XB and assigned a new ID
        assert scene_img is not None
    else:
        # Scene is already registered in XB
        assert scene_img is None

    # Add concept exemplar to memory
    if is_new_obj:
        # Scene object is new, store object info
        exemplars = [{ "scene_id": scene_id, "mask": mask, "f_vec": f_vec }]
    else:
        # Already stored in XB, no need to store again
        exemplars = []
    pointers = { exemplar_spec: [(is_new_obj, exemplar_id)] }
    added_inds = ex_mem.add_exs_2d(
        scene_img=scene_img, exemplars=exemplars, pointers=pointers
    )

    return added_inds[0] if is_new_obj else None
