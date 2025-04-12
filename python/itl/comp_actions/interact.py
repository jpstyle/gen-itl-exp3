"""
Implements composite agent interactions, environmental or dialogue, that need
coordination of different component modules
"""
import os
import re
import random
import logging
from operator import mul
from functools import reduce
from itertools import combinations, product, groupby
from collections import defaultdict, deque

import copy
import yaml
import torch
import numpy as np
import networkx as nx
from numpy.linalg import inv
from clingo import Control, SymbolType, Number, Function

from .constants import CON_GRAPH, STORE_VP_INDS
from .utils import rle_decode
from ..vision.utils import (
    xyzw2wxyz, flip_position_y, flip_quaternion_y, rmat2quat, transformation_matrix,
    blur_and_grayscale, visual_prompt_by_mask
)
from ..lpmln import Literal
from ..lpmln.utils import wrap_args


logger = logging.getLogger(__name__)

# Recursive helper method for checking whether rule cons/ante uses a reserved (pred
# type *) predicate 
has_reserved_pred = \
    lambda cnjt: cnjt.name.startswith("sp_") \
        if isinstance(cnjt, Literal) else any(has_reserved_pred(nc) for nc in cnjt)

def attempt_Q(agent, utt_pointer):
    """
    Attempt to answer an unanswered question from user.
    
    If it turns out the question cannot be answered at all with the agent's
    current knowledge (e.g. question contains unresolved neologism), do nothing
    and wait for it to become answerable.

    If the agent can come up with an answer to the question, right or wrong,
    schedule to actually answer it by adding a new agenda item.
    """
    dialogue_record = agent.lang.dialogue.export_resolved_record()

    predicates_mentioned = set()
    clauses_covered = set()
    clauses_mentioned = {utt_pointer}
    while len(clauses_mentioned) > 0:
        # Keep fetching all relevant clauses (denoted by their event variables)
        # until no more to fetch, while gathering all predicates mentioned in
        # each clause
        ti, ci = clauses_mentioned.pop()
        clauses_covered.add((ti, ci))
        _, _, ante, cons = dialogue_record[ti][1][ci][0]
        ante = ante or ()
        cons = cons or ()

        predicates_mentioned |= {lit.name for lit in ante+cons}

        relevant_clauses = {
            (int(ev_ref[0][0]), int(ev_ref[0][1]))
            for lit in ante + cons
            for arg, _ in lit.args
            if len(ev_ref := re.findall(r"t(\d+)c(\d+)", arg)) == 1
        }
        clauses_mentioned |= {
            cl_ind for cl_ind in relevant_clauses
            if cl_ind not in clauses_covered
        }

    pred_concs = {
        (
            (pred_spl := pred.split("_"))[0],
            int(pred_spl[1]) if pred_spl[1].isdigit() else pred_spl[1]
        )
        for pred in predicates_mentioned
    }
    if any(
        agent.lt_mem.lexicon.d2s[conc][0] in agent.lang.unresolved_neologisms
        for conc in pred_concs
    ):
        # Question cannot be answered for some relevant clause(s) include
        # neologism unknown to agent
        return
    else:
        # Schedule to answer the question
        agent.planner.agenda.appendleft(("answer_Q", utt_pointer))
        return

def prepare_answer_Q(agent, utt_pointer):
    """
    Prepare an answer to a question that has been deemed answerable. Ideally,
    implementations should be able to handle a more diverse range of questions
    in a more general manner (somewhat like the predecessors of this codebase),
    but for this project we will handcraft how to handle relevant question types.
    """
    # The question is about to be answered
    agent.lang.dialogue.unanswered_Qs.remove(utt_pointer)
    agent_action = None     # Return value placeholder

    exec_state = agent.planner.execution_state      # Shortcut var

    ti, ci = utt_pointer
    dialogue_record = agent.lang.dialogue.export_resolved_record()
    gq, bvars, ante, cons = dialogue_record[ti][1][ci][0]

    intention_Qs = [
        entailment[1]["terms"][("T2", True)][0]
        for c_lit in cons
        if (entailment := Literal.entailing_mapping_btw(
            [Literal("sp_intend", wrap_args("T1", "_self", "T2"))], [c_lit]
        ))[0] == 1
    ]
    if len(intention_Qs) > 0:
        # Answering question of type "What were you trying to join (through
        # the previous action you have just executed)?"

        # Get the sentential complement clause, i.e., the intended event
        assert len(intention_Qs) == 1
        sc_ind = re.findall(r"t(\d+)c(\d+)", intention_Qs[0])[0]
        intended_ev = dialogue_record[int(sc_ind[0])][1][int(sc_ind[1])][0]
        _, _, intended_ante, intended_cons = intended_ev

        # For now, can only process simple (antecedent-free) intentions
        # where the action predicate is 'join'
        assert len(intended_ante) == 0 and len(intended_cons) > 0
        join_conc = agent.lt_mem.lexicon.s2d[("va", "join")][0][1]
        assert {lit.name for lit in intended_cons} == {f"arel_{join_conc}"}

        # Answer by elaborating original intention of joining which part
        # instances, along with their types as recognized
        intended_join, last_action_type = _last_intended_join(agent)

        # Getting appropriate name handles, concepts and NL symbols
        get_conc = lambda s: agent.lt_mem.lexicon.s2d[("va", s)][0][1]
        pick_up_actions = [get_conc("pick_up_left"), get_conc("pick_up_right")]
        side_strs = ["Left", "Right"]
        syms = []; dens = []; ent_paths = []; ref_phrases = []
        for side, obj in enumerate(intended_join):
            if obj in agent.vision.scene:
                det = "this"
                ent = agent.vision.scene[obj]["env_handle"]
                if last_action_type in pick_up_actions:
                    if side == pick_up_actions.index(last_action_type):
                        ent_path = f"/*/{ent}"
                    else:
                        ent_path = f"/Student Agent/{side_strs[side]} Hand/*/{ent}" \
                            if exec_state["manipulator_states"][side][0] is not None \
                            else f"/*/{ent}"
                else:
                    ent_path = f"/Student Agent/{side_strs[side]} Hand/*/{ent}"
            else:
                det = "a"
                ent_path = None
            den = exec_state["recognitions"][obj]
            sym = agent.lt_mem.lexicon.d2s[("pcls", den)][0][1]
            ref_phrase = f"{det} {sym}"

            syms.append(sym); dens.append(den)
            ent_paths.append(ent_path); ref_phrases.append(ref_phrase)

        # NL surface forms and corresponding logical forms
        surface_form_0 = f"I was trying to join {ref_phrases[0]} and {ref_phrases[1]}."
        surface_form_1 = f"# to-infinitive phrase ('to join {ref_phrases[0]} and {ref_phrases[1]}')"
        gq_0 = gq_1 = (); bvars_0 = bvars_1 = (); ante_0 = ante_1 = []
        cons_0 = [
            ("sp", "intend", [("e", 0), "x0", ("e", 1)]),
            ("sp", "pronoun1", ["x0"]),
        ]
        # Note: Tuple argument in form of ("e", integer) refers to another clause
        # in generation buffer specified by the integer offset; e.g., ("e", 1) would
        # refer to the next clause after this one. Notation's a bit janky, but this
        # will do for now.
        cons_1 = [
            ("va", "join", [("e", 0), "x0", "x1", "x2"]),
            ("n", syms[0], ["x1"]),
            ("n", syms[1], ["x2"])
        ]
        logical_form_0 = (gq_0, bvars_0, ante_0, cons_0)
        logical_form_1 = (gq_1, bvars_1, ante_1, cons_1)

        # Referents & predicates info
        referents_0 = { 
            "e": { "mood": ".", "tense": "past" },      # Past indicative
            "x0": { "entity": "_self", "rf_info": {} }
        }
        referents_1 = {
            "e": { "mood": "~" },       # Infinitive
            "x0": { "entity": None, "rf_info": {} }
        }
        for side, obj in enumerate(intended_join):
            if obj in agent.vision.scene:
                referents_1[f"x{side+1}"] = { "entity": obj, "rf_info": {} }
            else:
                referents_1[f"x{side+1}"] = { "entity": None, "rf_info": {} }
        predicates_0 = {
            "pc0": (("sp", "intend"), "sp_intend"),
            "pc1": (("sp", "pronoun1"), "sp_pronoun1")
        }
        predicates_1 = {
            "pc0": (("va", "join"), f"arel_{join_conc}"),
            "pc1": (("n", syms[0]), f"pcls_{dens[0]}"),
            "pc2": (("n", syms[1]), f"pcls_{dens[1]}")
        }

        # Point to each part involved in the originally intended join
        dem_refs_0 = {}
        for ref_phrase, ent_path in zip(ref_phrases, ent_paths):
            if ent_path is None: continue
            offset = surface_form_0.find(ref_phrase)
            dem_refs_0[(offset, offset+len(ref_phrase))] = (ent_path, False)
                # Note: False values in the tuples specify that the demonstrative
                # references are conveyed via string name handles instead of masks

        # Append to & flush generation buffer
        record_0 = (
            logical_form_0, surface_form_0, referents_0, predicates_0, dem_refs_0
        )
        record_1 = (
            logical_form_1, surface_form_1, referents_1, predicates_1, {}
        )
        agent.lang.dialogue.to_generate += [record_0, record_1]
        agent_action = agent.lang.generate()

    return agent_action

def _last_intended_join(agent):
    """
    Helper method factored out for fetching the latest join intended by
    agent, of which the latest agent actions in service.
    """
    exec_state = agent.planner.execution_state      # Shortcut var

    get_conc = lambda s: agent.lt_mem.lexicon.s2d[("va", s)][0][1]
    pick_up_actions = [get_conc("pick_up_left"), get_conc("pick_up_right")]
    assemble_actions = [
        get_conc("assemble_right_to_left"), get_conc("assemble_left_to_right")
    ]
    last_action_type, last_action_params = [
        (action_type, action_params)
        for action_type, action_params, actor in exec_state["action_history"]
        if actor == "Student"
    ][-1]
    if last_action_type in pick_up_actions:
        # The previous action undone was a picking up which resulted in
        # holding a pair of subassemblies that can never be joined. In this
        # case, the intended join is the next one in the (scrapped) planner
        # agenda.
        intended_join = next(
            (todo_args[1][1], todo_args[1][4])
            for todo_type, todo_args in exec_state["last_scrapped_plan"]
            if todo_type == "execute_command" and todo_args[0] in assemble_actions
        )
    else:
        # The previous action undone was an assembly between a (valid) pair
        # of subassemblies, yet at an incorrect pose. In this case, the
        # intended join is the undone assembly action.
        assert last_action_type in assemble_actions
        intended_join = (last_action_params[1], last_action_params[4])

    return intended_join, last_action_type

def attempt_command(agent, utt_pointer):
    """
    Attempt to execute an unexecuted command from user.
    
    If it turns out the command cannot be executed at all with the agent's
    current knowledge (e.g. has not heard of the target concept, does not know
    how to interact with environment to reach target state), report inability
    and dismiss the command; soon a demonstration or a definition will be provided
    from user.

    If the agent can come up with a plan to fulfill the command, right or wrong,
    schedule to actually plan & execute it by adding a new agenda item.
    """
    dialogue_record = agent.lang.dialogue.export_resolved_record()

    ti, ci = utt_pointer
    (_, _, _, cons), raw, _ = dialogue_record[ti][1][ci]

    command_executable = True
    addressed_before = agent.lang.dialogue.unexecuted_commands[utt_pointer]

    action_lit = [lit for lit in cons if lit.name.startswith("arel_")][0]
    action_type = int(action_lit.name.replace("arel_", ""))
    action_params = {}
    for arg in action_lit.args[2:]:
        arg_describing_lits = [lit for lit in cons if lit.args==[arg]]

        if agent.lang.dialogue.referents["dis"][arg[0]].get("is_pred"):
            # Referent denotes a predicate category
            action_params[arg[0]] = [lit.name for lit in arg_describing_lits]
            pred_concs = {
                ((pred_spl := pred.split("_"))[0], int(pred_spl[1]))
                for pred in action_params[arg[0]]
            }
            if any(
                agent.lt_mem.lexicon.d2s[conc][0] in agent.lang.unresolved_neologisms
                for conc in pred_concs
            ):
                # Neologism included in action parameter config, cannot execute
                # (neologism will be separately handled by `report_neologism` method)
                command_executable = False
                break
        else:
            # Referent denotes an environment entity
            action_params[arg[0]] = agent.lang.dialogue.referents["env"][-1][arg[0]]

    if command_executable:
        # Some reformatting of action parameters according to action type
        action_name = agent.lt_mem.lexicon.d2s[("arel", action_type)][0][1]
        if action_name == "build":
            build_target = list(action_params.values())[0][0].split("_")
            build_target = (build_target[0], int(build_target[1]))
            action_params = (build_target, False)       # `replan`: False

        # Schedule to generate plan & execute towards fulfilling the command
        agent.planner.agenda.append(("execute_command", (action_type, action_params)))
        del agent.lang.dialogue.unexecuted_commands[utt_pointer]
        return

    elif addressed_before:
        # Value true only if the command was attempted before but proven unable
        # to be executed, then verbally reported to user. Do not report again,
        # silently skipping.
        return

    else:
        # Command proven to be not executable; report inability, only if never
        # addressed before

        # Mark 'addressed'
        agent.lang.dialogue.unexecuted_commands[utt_pointer] = True

        ri_command = f"t{ti}c{ci}"                  # Denotes original request

        # NL surface form and corresponding logical form
        surface_form = f"I am unable to {raw[0].lower()}{raw[1:]}"
        gq = (); bvars = (); ante = []
        cons = [
            ("sp", "unable", [("e", 0), "x0", ri_command]),
            ("sp", "pronoun1", ["x0"])
        ]
        logical_form = (gq, bvars, ante, cons)

        # Referents & predicates info
        referents = {
            "e": { "mood": "." },       # Indicative
            "x0": { "entity": "_self", "rf_info": {} }
        }
        predicates = { "pc0": (("sp", "unable"), "sp_unable") }

        agent.lang.dialogue.to_generate.append(
            (logical_form, surface_form, referents, predicates, {})
        )
        agent_action = agent.lang.generate()

        return agent_action

def execute_command(agent, action_spec):
    """
    Execute a command (that was deemed executable before by `attempt_command`
    method) by appropriate planning, based on the designated action type and
    parameters provided as arguments.
    """
    action_type, action_params = action_spec

    if action_type is None:
        # Special token representing the policy 'wait until teacher reaction,
        # either by silent observation or interruption for corrective feedback'.
        # Block execution of remaining plan steps indefinitely until teacher's
        # reaction.
        agent.planner.agenda.appendleft(("execute_command", (None, None)))
        return [(None, None)]

    exec_state = agent.planner.execution_state      # Shortcut var

    # Currently considered commands: some long-term commands that requiring
    # long-horizon planning (e.g., 'build'), and some primitive actions that
    # are to be executed---that is, signaled to Unity environment---immediately

    # Record the spec of action being executed
    exec_state["action_history"].append(
        (action_type, action_params, "Student")
    )

    action_name = agent.lt_mem.lexicon.d2s[("arel", action_type)][0][1]
    match action_name:
        case "build":
            return _execute_build(agent, action_params)

        case "pick_up_left" | "pick_up_right":
            return _execute_pick_up(agent, action_name, action_params)

        case "drop_left" | "drop_right":
            return _execute_drop(agent, action_name)

        case "assemble_right_to_left" | "assemble_left_to_right":
            return _execute_assemble(agent, action_name, action_params)

        case "inspect_left" | "inspect_right":
            return _execute_inspect(agent, action_name, action_params)

def _execute_build(agent, action_params):
    """
    Planning for 'Build a X' type of commands, which would take a sequence of
    primitive actions to accomplish. Process the command by:
        1) Running appropriate vision module methods to obtain segmentation
            masks; in this study, we will assume learner have access to ground-
            truth segmentation masks of atomic parts on the tabletop (only),
            whereas relaxing this would require using some general-purpose
            segmentation model, meaning non-objects (i.e., invalid masks)
            may be introduced.
        2) Synthesizing ASP program for planning out of the visual prediction
            results, current knowledge base status and the provided goal action
            spec (We only consider executing 'build X' commands for now)
        3) Solving the compiled ASP program with clingo, adding the obtained
            sequence of atomic actions to agenda
    """
    build_target, replan = action_params
    exec_state = agent.planner.execution_state    # Shortcut var

    if not replan:
        # Initialize plan execution state tracking dict
        agent.planner.execution_state |= {
            "plan_goal": ("build", build_target),
            "connection_graphs": {},
            "recognitions": {},
            "nogood_objects": set(),        # Logs distractors
            "metrics": {
                "num_planning_attempts": 0,
                "num_collision_queries": 0
            }
        }

    # Incorporate any previous labeling feedback from user
    resolved_record = agent.lang.dialogue.export_resolved_record()
    color_concs = {
        agent.lt_mem.lexicon.s2d[("a", col)][0][1]
        for col in ["red", "green", "blue", "white", "gold"]
    }
    labeling_map = {}
    for spk, turn_clauses in resolved_record:
        if spk != "Teacher": continue
        for ((_, _, ante, cons), _, clause_info) in turn_clauses:
            if clause_info["mood"] != ".": continue
            if not (len(ante) == 0 and len(cons) > 0): continue
            for lit in cons:
                if not lit.name.startswith("pcls"): continue
                if lit.args[0][0] not in agent.vision.scene: continue
                if lit.naf: continue
                conc = int(lit.name.strip("pcls_"))
                if conc in color_concs: continue
                labeling_map[lit.args[0][0]] = conc

    # Update execution state to reflect recognitions hitherto committed
    # and those certified by user feedback
    exec_state["recognitions"] = {
        n: exec_state["recognitions"][n]
        for gr in exec_state["connection_graphs"].values()
        for n in gr
        if n in exec_state["recognitions"]
    } | {
        objs: labels
        for objs, labels in exec_state["recognitions"].items()
        if isinstance(objs, tuple)
    } | labeling_map

    # Note to self: One could imagine we could skip re-planning from scratch
    # after the user's labeling feedback if it did not involve an instance
    # that is already referenced in the remainder of the plan (i.e., when the
    # corrective feedback does not refute the premise based on agent's previous
    # object recognition output)... However, it turned out it caused too much
    # headache to juice such opportunities, and it's much easier to simply
    # re-plan after any form of knowledge update (after dropping all held
    # objects in hands).

    # Plan towards building valid target structure
    action_sequence, recognitions, plan_complete = _plan_assembly(
        agent, build_target
    )

    # Record how the agent decided to recognize each (non-)object
    agent.planner.execution_state["recognitions"] = recognitions

    # Enqueue appropriate agenda items and finish
    agent.planner.agenda = deque(
        ("execute_command", action_step) for action_step in action_sequence
    )       # Whatever steps remaining, replace
    if plan_complete:
        agent.planner.agenda.append(("posthoc_episode_analysis", ()))
        agent.planner.agenda.append(("utter_simple", ("Done.", { "mood": "." })))
    else:
        agent.planner.agenda.append(("report_planning_failure", ()))

def _plan_assembly(agent, build_target):
    """
    Helper method factored out for planning towards the designated target
    structure. May be used for replanning after user's belief/knowledge status
    changed, picking up from current assembly progress state recorded.
    """
    exec_state = agent.planner.execution_state    # Shortcut var

    # Tracking numbers of calls to ASP planner and those to collision checker
    total_planning_attempts = total_query_count = 0
    planning_forfeited = False                    # Start optimistic...

    # Fetching action concept indices
    get_conc = lambda s: agent.lt_mem.lexicon.s2d[("va", s)][0][1]
    pick_up_actions = [get_conc("pick_up_left"), get_conc("pick_up_right")]
    drop_actions = [get_conc("drop_left"), get_conc("drop_right")]
    assemble_actions = [
        get_conc("assemble_right_to_left"), get_conc("assemble_left_to_right")
    ]

    # First run ASP inference for selecting a viable target structure based
    # on agent's perception of scene objects and ongoing assembly progress
    # in environment so far
    best_model = _goal_selection(agent, build_target)
    if best_model is None:
        # If agent wasn't able to find a fitting target structure that comply
        # with the demonstrated join, it might mean two things, either currently
        # committed part recognition for the joined parts are incorrect, or
        # agent doesn't know a valid structure that can comply with the 
        # recognitions. In any case, forfeit planning for the current round.
        # Also, uncommit the recognitions involved in any of teacher's assemble~
        # actionssince agent wouldn't have made such joins without being interrupted
        # by teacher.
        planning_forfeited = True
        for action_type, action_params, actor in exec_state["action_history"]:
            if actor != "Teacher": continue
            action_name = agent.lt_mem.lexicon.d2s[("arel", action_type)][0][1]
            if action_name.startswith("assemble"):
                _, atomic_left, _, _, atomic_right, _, _ = action_params
                if atomic_left in exec_state["recognitions"]:
                    del exec_state["recognitions"][atomic_left]
                if atomic_right in exec_state["recognitions"]:
                    del exec_state["recognitions"][atomic_right]
        
    else:
        # Compile the optimal solution into appropriate data structures
        tabulated_results = _tabulate_goal_selection_result(best_model)
        assembly_hierarchy, obj2node_map = tabulated_results[:2]
        connect_edges, connection_graph = tabulated_results[2:4]
        node_unifications, atomic_node_concs = tabulated_results[4:6]
        part_recognitions, hyp_rels = tabulated_results[6:]

        # Build target may have one or more 'template' options that count as
        # valid structure for the concept, fetch the index of the selected
        # template in order to retrieve storage of any known scope entailments
        selected_tmpl = next(
            (atm.arguments[1].number, atm.arguments[2].number)
            for atm in best_model
            if atm.name=="node_sa_template" and \
                (top_node := atm.arguments[0]).type == SymbolType.Number and \
                top_node.number == 0
        )

        # At this point, check if the task is already finished (if so, presumably
        # by user through demonstration), in which case no more planning is
        # needed. Check this by seeing if there's only one existing subassembly,
        # and the number of its constituent atomics are equal to the size of
        # the connection graph.
        nonatomic_subassemblies = [
            sa_graph for sa_graph in exec_state["connection_graphs"].values()
            if len(sa_graph) > 1
        ]
        if len(nonatomic_subassemblies) == 1:
            sole_sa = nonatomic_subassemblies[0]
            if len(sole_sa) == len(connection_graph):
                # Can return with empty action sequence, same recognitions and
                # True `plan_complete` flag
                return [], exec_state["recognitions"], True

        # In principle, unbounded existing objects could be unified with any of the
        # nodes as long as they don't uniquely unify with some other object, but that
        # still would be too broad. We could eliminate many options by means of graph
        # matching algorithm.
        obj2sa_map = {
            ex_obj: sa
            for sa, sa_graph in exec_state["connection_graphs"].items()
            for ex_obj in sa_graph if len(sa_graph) > 1
        }
        possible_mappings = _match_existing_subassemblies(
            connection_graph, obj2sa_map, node_unifications,
            atomic_node_concs, hyp_rels, exec_state
        )
        covered_objs = {
            obj for group_mappings in possible_mappings.values()
            for ism in group_mappings
            for obj in ism.values()
        }
        unmapped_subassemblies = {
            sa for sa, sa_graph in exec_state["connection_graphs"].items()
            for ex_obj in sa_graph
            if len(sa_graph) > 1 and ex_obj not in covered_objs
        }
        if len(unmapped_subassemblies) > 0:
            # We know as invariant that existing subassemblies will always be
            # subgraphs of the full connection graph, so if any subassembly is
            # not matched, it should be due to inaccurate part type recognition
            # premise. Uncommit type recognitions for the latest join demo'ed
            # by teacher, so that planning is possible at least later. Forfeit
            # planning for the current round.
            for sa in unmapped_subassemblies:
                for obj in exec_state["connection_graphs"][sa]:
                    if obj in exec_state["recognitions"]:
                        del exec_state["recognitions"][obj]
            planning_forfeited = True

    if planning_forfeited:
        # First early exit opportunity when planning is forfeited for the
        # current round
        log_msg = "Forfeited planning after "
        log_msg += "0 (re)planning attempts "
        log_msg += "(0 calls total)"
        logger.info(log_msg)

        return [], exec_state["recognitions"], False

    # Turns out if a structure consists of too many atomic parts (say, more
    # than 8 or so), ASP planner performance is significantly affected. We
    # handle this by breaking each planning subproblem down to multiple
    # smaller ones... In principle, this may risk planning failure when
    # physical collision check is involved, depending on how the target
    # structure is partitioned. Note that agents that are aware of valid
    # substructures are relatively free from such concerns, as sizes of
    # those substructures tend to be more manageable in general.

    # Sample a 'compression sequence', i.e. a sequence of collection of nodes
    # that altogether make up a meaningful substructure of the target structure.
    # The sequential sampling is achieved by iteratively 'compressing' valid
    # set of nodes in the assembly hierarchy, where by valid we mean all
    # assembly preconditions are cleared. In the edge case where the assembly
    # hierarchy is flat and thus has max depth of just 2, the length of the
    # resulting compression sequence would be just 1, where the singleton
    # collection contains all atomic nodes.
    sa_nodes = {
        n: [chd for _, chd in assembly_hierarchy.out_edges(n)]
        for n in assembly_hierarchy
        if assembly_hierarchy.nodes[n]["node_type"] == "sa"
    }
    compression_steps = len(sa_nodes)
    compression_sequence = []
    for _ in range(compression_steps):
        compressible_nodes = [
            n for n, children in sa_nodes.items()
            if not any(chd in sa_nodes for chd in children)
        ]
        selected = random.sample(compressible_nodes, 1)[0]
        compression_sequence.append((selected, sa_nodes.pop(selected)))

    # Node unification result may not always be unique, and some objects
    # in existing subassemblies may be potentially unifiable to more
    # than one nodes in connection graph. Some unification decisions
    # may lead to unsolvable planning problems, whereas others allow
    # solutions. Iterate across all possible unification scenarios and
    # try divide-and-conquer solving of the planning problem based on
    # corresponding premises.

    # Sometimes, if there's too much uncertainty on types of parts, the
    # number of unification scenarios to consider can be simply too large.
    # It is just not worth it to examine every single case, most of which
    # may not even be significantly distinct. Since we are in an interactive
    # task setting, we will just give up and cry for help so that user can
    # step in to provide appropriate feedback.
    num_total_mappings = reduce(
        mul, [1] + [len(opts) for opts in possible_mappings.values()]
    )
    valid_scenarios = []
    if num_total_mappings > 1000:
        planning_forfeited = True       # Too many options to validate
    else:
        # Select valid scenarios by filtering out any in which some objects are
        # unified with the same node
        unification_scenarios = product(*[
            isms for _, isms in possible_mappings.items()
        ])
        valid_scenarios = []; obj2sa_map = {}
        for scenario in unification_scenarios:
            # Merge the per-group isomorphic mappings into a single mapping,
            # while fetching object-to-subassembly affiliations as well
            combined_mapping = {}
            for mapping in scenario:
                for n, ex_obj in mapping.items():
                    for sa, sa_graph in exec_state["connection_graphs"].items():
                        if ex_obj not in sa_graph: continue
                        obj2sa_map[ex_obj] = sa
                        combined_mapping[ex_obj] = n

            if len(set(combined_mapping.values())) != len(combined_mapping):
                # Some different objects mapped to the same node, invalid
                continue
            if any(
                atomic_node_concs[combined_mapping[objs[0]]] == -labels[0] and \
                    atomic_node_concs[combined_mapping[objs[1]]] == -labels[1]
                for objs, labels in exec_state["recognitions"].items()
                if isinstance(objs, tuple) and objs[0] in combined_mapping and \
                    objs[1] in combined_mapping
            ):
                # Mapping violates existing pairwise negative type constraint
                continue

            valid_scenarios.append(combined_mapping)

    if len(valid_scenarios) > 100 or num_total_mappings > 1000:
        # Give up if too many join trees to inspect. The latter check, which
        # is redundant with that above, is to catch such cases and not let them
        # flow into the else case below.
        planning_forfeited = True
    else:
        # Extracted mapping from string part names to agent's internal concept
        # denotations
        if agent.cfg.exp.player_type in ["bool", "demo"]:
            # While agent and user doesn't have shared vocab, agent has been 
            # injected with a codesheet of correspondence between part concept
            # and string identifier used in Unity
            part_names = agent.lt_mem.lexicon.codesheet
        else:
            assert agent.cfg.exp.player_type in ["label", "full"]
            part_names = {
                d[1]: s[0][1] for d, s in agent.lt_mem.lexicon.d2s.items()
                if d[0]=="pcls"
            }
        # Ground-truth oracle for inspecting what each contact point learned
        # by agent was supposed to stand for in the knowledge encoded in Unity
        # assets
        cp_names = {
            cp_conc: gt_handle
            for _, _, _, cps in agent.lt_mem.exemplars.object_3d.values()
            for cp_conc, (_, gt_handle) in cps.items()
        }

        valid_join_trees = []
        structures = agent.lt_mem.kb.assembly_structures
        _, scope_entailments, verified_joins = \
            structures[("pcls", selected_tmpl[0])][selected_tmpl[1]]
        violated_entailments_by_obj = set()
        for unification_choice in valid_scenarios:
            if total_planning_attempts >= 30:
                # Tolerate up to 30 total planning attempts; if couldn't
                # find a solution after that forfeit planning altogether
                planning_forfeited = True
                break

            # Try solving the planning problem with the unification premise
            # encoded in `unification_choice`
            unification_choice_inv = {
                n: ex_obj for ex_obj, n in unification_choice.items()
            }
            join_tree, planning_attempts, query_count, viol_ents = \
                _divide_and_conquer(
                    compression_sequence, connection_graph, exec_state["recognitions"],
                    atomic_node_concs, connect_edges, part_names, cp_names,
                    (exec_state["connection_graphs"], unification_choice),
                    scope_entailments, verified_joins,
                    agent.cfg.paths.assets_dir, agent.cfg.seed
                )
            total_planning_attempts += planning_attempts
            total_query_count += query_count
            violated_entailments_by_obj |= {
                (
                    frozenset([unification_choice_inv[n] for n in ante]),
                    unification_choice_inv[cons]
                )
                for ante, cons in viol_ents
                if all(n in unification_choice_inv for n in ante | {cons})
            }
            if join_tree is not None:
                valid_join_trees.append((join_tree, unification_choice))

        if len(valid_join_trees) == 0:
            # None of the scenarios resulted in valid join trees, which shouldn't
            # happen if agent has all of its part type recognition premise correct.
            # Find all relevant joins made by Teacher and uncommit their part type
            # recognitions. Forfeit planning for the current round.
            for ante, cons in violated_entailments_by_obj:
                involved_objs = ante | {cons}
                for action_type, action_params, actor in exec_state["action_history"]:
                    if actor != "Teacher": continue
                    action_name = agent.lt_mem.lexicon.d2s[("arel", action_type)][0][1]
                    if not action_name.startswith("assemble"): continue
                    _, atomic_left, _, _, atomic_right, _, _ = action_params

                    for atomic in [atomic_left, atomic_right]:
                        if atomic in involved_objs and \
                            atomic in exec_state["recognitions"]:
                            del exec_state["recognitions"][atomic]
            planning_forfeited = True

    if planning_forfeited:
        # Search space too wide, couldn't finish planning within reasonable
        # time frame.
        log_msg = "Forfeited planning after "
        action_sequence = []
        node2obj_map = {}
        plan_complete = False
    else:
        # Inspect the unification premises to see if node unification options
        # are unique or not
        collected_unification_choices = defaultdict(set)
        for _, unification_choice in valid_join_trees:
            for obj, n in unification_choice.items():
                collected_unification_choices[obj].add(n)

        # Obtain intersection of sets of joins in all the join trees, invariant
        # to naming of intermediate subassemblies. The intersection represents
        # collection of edges common to all of the sanitized join trees, thus
        # safe to make.
        def remaining_join_signatures(join_tree):
            join_tree = join_tree.copy()
            available_joins = [
                n for n in join_tree if len(nx.ancestors(join_tree, n))==2
            ]
            signatures = {}
            while len(available_joins) > 0:
                for join_res in available_joins:
                    n1, n2 = tuple(n for n, _ in join_tree.in_edges(join_res))
                    signatures[join_res] = frozenset([
                        (signatures.get(n1, n1), join_tree.edges[(n1, join_res)]["join_by"]),
                        (signatures.get(n2, n2), join_tree.edges[(n2, join_res)]["join_by"])
                    ])
                    join_tree.remove_edge(n1, join_res); join_tree.remove_edge(n2, join_res)
                available_joins = [
                    n for n in join_tree if len(nx.ancestors(join_tree, n))==2
                ]
            return set(signatures.values())
        common_joins = set.intersection(*[
            remaining_join_signatures(join_tree) for join_tree, _ in valid_join_trees
        ])
        # Build the final intersection tree, combining the commonly occurring joins
        # into a single tree/forest, naming intermediate products on the fly
        tree_intersection = nx.DiGraph(); i = 0
        sa_sgns = {}; sa_sgns_inv = {}
        while len(common_joins) > 0:
            unprocessed_joins = []
            for n1, n2 in common_joins:
                if isinstance(n1[0], str) and isinstance(n2[0], str):
                    # Can be processed now
                    product_name = f"i{i}"
                    join_signature = frozenset([
                        (sa_sgns.get(n1[0], n1[0]), n1[1]),
                        (sa_sgns.get(n2[0], n2[0]), n2[1])
                    ])
                    tree_intersection.add_edge(n1[0], product_name, join_by=n1[1])
                    tree_intersection.add_edge(n2[0], product_name, join_by=n2[1])
                    sa_sgns[product_name] = join_signature
                    sa_sgns_inv[join_signature] = product_name
                    i += 1
                else:
                    # To be processed at a later iteration
                    unprocessed_joins.append((n1, n2))

            common_joins = {
                frozenset([
                    (sa_sgns_inv.get(n1[0], n1[0]), n1[1]),
                    (sa_sgns_inv.get(n2[0], n2[0]), n2[1])
                ])
                for n1, n2 in unprocessed_joins
            }

        # Join tree is complete if all the nodes and edges are retained after
        # the series of intersection
        plan_complete = len(tree_intersection.nodes) == len(valid_join_trees[0][0].nodes) \
            and len(tree_intersection.edges) == len(valid_join_trees[0][0].edges)

        # Identifying 'user-certified' labelings, defined as pairs of "Is there
        # a X?" questions and immediately following user-provided labels. This
        # is to prevent repetitive asking of the exact same question, operating
        # under the assumption that the user-labeled objects provided as answer
        # to such questions are ground-truth part instances always included in
        # the desired goal structure and thus always should fill a node
        resolved_record = agent.lang.dialogue.export_resolved_record()
        certified_pool = defaultdict(set); q_asked = False
        for spk, turn_clauses in resolved_record:
            for ((_, _, ante, cons), _, clause_info) in turn_clauses:
                if spk == "Student" and clause_info["mood"] == "?" and len(cons) == 1:
                    q_asked = True
                if spk == "Teacher" and clause_info["mood"] == "." and len(cons) == 1 \
                    and q_asked and cons[0].name.startswith("pcls"):
                    q_conc = int(cons[0].name.split("_")[1])
                    if q_conc in hyp_rels:
                        q_conc_supertypes = nx.ancestors(hyp_rels, q_conc) | {q_conc}
                    else:
                        q_conc_supertypes = {q_conc}
                    pool_conc = q_conc_supertypes & set(atomic_node_concs.values())
                    assert len(pool_conc) == 1
                    pool_conc = list(pool_conc)[0]
                    certified_pool[pool_conc].add(cons[0].args[0][0])
                    q_asked = False

        safe_unification_choice = {
            n: list(objs)[0] for n, objs in collected_unification_choices.items()
            if len(objs) == 1
        }
        action_sequence, node2obj_map = _linearize_join_tree(
            copy.deepcopy(tree_intersection), exec_state, connection_graph,
            connect_edges, atomic_node_concs, safe_unification_choice,
            obj2node_map, certified_pool,
            (pick_up_actions, drop_actions, assemble_actions)
        )       # The valid `unification_choice` obtained above is passed as well

        plan_dscr = "Full" if plan_complete else "Partial"
        log_msg = f"{plan_dscr} plan found after "

    # 'Commit' to a set of part type recognitions, i.e., decisions as to
    # of which type to consider each object as an instance
    recognitions = {}
    for oi, conc in node2obj_map.values():
        if oi in node_unifications and oi not in safe_unification_choice:
            # Skip if already unified and unification is not safe
            continue
        if oi in agent.vision.scene:
            # For existing scene objects
            if oi in part_recognitions:
                # Specify subtypes as committed by _goal_selection if available
                recognitions[oi] = part_recognitions[oi]
            elif conc in hyp_rels:
                # Select the most likely one by prediction score
                recognitions[oi] = max([
                    subtype_conc for subtype_conc in nx.descendants(hyp_rels, conc)
                    if len(hyp_rels.out_edges(subtype_conc)) == 0       # Most specific ones only
                ], key=lambda c: agent.vision.scene[oi]["pred_cls"][c])
            else:
                # No relevant supertype/subtypes, just singleton in the taxonomy
                recognitions[oi] = conc
        else:
            # For non-objects, keep the broad supertype so they can be queried
            # for later
            recognitions[oi] = conc
    # Prioritize existing ones
    recognitions.update(exec_state["recognitions"])

    # Update and record metrics
    exec_state["metrics"]["num_planning_attempts"] += total_planning_attempts
    exec_state["metrics"]["num_collision_queries"] += total_query_count

    # If there's any violation of the pairwise negative labeling feedback in
    # the resulting recognition decisions, consider the plan as invalid; ditch
    # the plan and report 'planning forfeited'
    pair_neg_viols = [
        (obj1, conc1, obj2, conc2)
        for (obj1, conc1), (obj2, conc2) in combinations([(obj, conc) for obj, conc in recognitions.items() if isinstance(obj, str)], 2)
        if recognitions.get((obj1, obj2)) == (-conc1, -conc2) or recognitions.get((obj2, obj1)) == (-conc2, -conc1)
    ]
    if len(pair_neg_viols) > 0:
        log_msg = "Forfeited planning after "
        log_msg += f"{total_planning_attempts} (re)planning attempts "
        log_msg += f"({total_query_count} calls total)"
        logger.info(log_msg)
        return [], exec_state["recognitions"], False

    # Log metrics
    log_msg += f"{total_planning_attempts} (re)planning attempts "
    log_msg += f"({total_query_count} calls total)"
    logger.info(log_msg)

    return action_sequence, recognitions, plan_complete

def _goal_selection(agent, build_target):
    """
    Helper method factored out for selecting a viable target structure to be
    built out of currently available scene objects as perceived, all the while
    accounting for any progress so far accomplished in the 'real world'. In
    our scope this procedure will always succeed and return an optimal clingo
    model, which contains information about the target structure as well as
    any unification results for atomic objects used in existing subassemblies.
    """
    exec_state = agent.planner.execution_state      # Shortcut var

    # Convert current visual scene (from ensemble prediction) into ASP fact
    # literals (note difference vs. agent.lt_mem.kb.visual_evidence_from_scene
    # method, which is for probabilistic reasoning by LP^MLN)
    threshold = 0.15; dsc_bin = 30
    observations = set()
    likelihood_values = np.stack([
        obj["pred_cls"] for obj in agent.vision.scene.values()
    ])
    # Obtaining per-category minimums/maximums of probability scores for
    # normalization and discretization later
    min_vals = np.maximum(likelihood_values.min(axis=0), threshold)
    max_vals = likelihood_values.max(axis=0)
    val_ranges = max_vals - min_vals
    color_concs = {
        agent.lt_mem.lexicon.s2d[("a", col)][0][1]
        for col in ["red", "green", "blue", "white", "gold"]
    }
    for oi, obj in agent.vision.scene.items():
        if oi in exec_state["nogood_objects"]:
            # Object marked as not to be used in planning, skip
            continue
        elif oi in exec_state["recognitions"]:
            # Object recognition already committed with confidence, or labeling
            # feedback directly provided by user; max confidence; assign sufficiently
            # high score such that it will always raise final score if included
            ci = exec_state["recognitions"][oi]
            obs_lit = Literal("type_likely", wrap_args(oi, ci, dsc_bin))
            observations.add(obs_lit)
        else:
            # List all non-color predictions with values above threshold
            for ci, val in enumerate(obj["pred_cls"]):
                if val < threshold: continue
                if ci in color_concs: continue

                # Normalize & discretize within [0,dsc_bin]; the more bins we
                # use for discrete approximation, the more time it takes to
                # solve the program
                if val_ranges[ci] == 0:
                    dsc_val = 0         # Pessimistic
                else:
                    nrm_val = (val-min_vals[ci]) / val_ranges[ci]
                    dsc_val = int(nrm_val * dsc_bin)

                if dsc_val >= dsc_bin / 2:
                    # Setting another threshold by the discrete score
                    obs_lit = Literal("type_likely", wrap_args(oi, ci, dsc_val))
                    observations.add(obs_lit)

        # List color predictions, assuming mutual exclusivity: i.e. include
        # only the max-score prediction, only if the score is above threshold
        likely_colors = [
            (ci, val) for ci, val in enumerate(obj["pred_cls"])
            if ci in color_concs and val > threshold
        ]
        if len(likely_colors) > 0:
            max_col, max_val = max(likely_colors, key=lambda x: x[1])
            if max_val >= threshold:
                nrm_val = (max_val-min_vals[max_col]) / val_ranges[max_col]
                dsc_val = int(nrm_val * dsc_bin)
                obs_lit = Literal("color_likely", wrap_args(oi, max_col, dsc_val))
                observations.add(obs_lit)

        # for ci, val in enumerate(obj["pred_cls"]):
        #     if val < threshold: continue
        #     if ci not in color_concs: continue

        #     # Normalize & discretize within [0,dsc_bin]; the more bins we
        #     # use for discrete approximation, the more time it takes to
        #     # solve the program
        #     if val_ranges[ci] == 0:
        #         dsc_val = 0         # Pessimistic
        #     else:
        #         nrm_val = (val-min_vals[ci]) / val_ranges[ci]
        #         dsc_val = int(nrm_val * dsc_bin)

        #     if dsc_val >= dsc_bin / 2:
        #         # Setting another threshold by the discrete score
        #         obs_lit = Literal("color_likely", wrap_args(oi, ci, dsc_val))
        #         observations.add(obs_lit)

    # Compile assembly structure knowledge into ASP fact literals
    structures = agent.lt_mem.kb.assembly_structures
    assembly_pieces = set()
    for (_, sa_conc), templates in structures.items():
        # For each valid template option
        for ti, (template, _, _) in enumerate(templates):
            # Register this specific template option
            assembly_pieces.add(Literal(
                "template_option", wrap_args(sa_conc, ti)
            ))

            # Register part/subassembly requirement for each node
            for n, data in template.nodes(data=True):
                if data["node_type"] == "atomic":
                    conc_ind = data["conc"]
                    assembly_pieces.add(Literal(
                        "req_atomic", wrap_args(sa_conc, ti, n, conc_ind)
                    ))
                else:
                    assert data["node_type"] == "sa"
                    conc_ind, sa_ind = data["conc"]
                    assembly_pieces.add(Literal(
                        "req_template",
                        wrap_args(sa_conc, ti, n, conc_ind, sa_ind)
                    ))

            # Register assembly contact signatures, iterating over edges
            sgn_str = lambda sg: str(sg[0]) if len(sg) == 1 \
                else f"c({sg[0]},{sgn_str(sg[1:])})"        # Serialization
            for u, v, data in template.edges(data=True):
                u, v = min(u, v), max(u, v)     # Ensure sorted order
                sgn_u = sgn_str(data["contact"][u][0])
                sgn_v = sgn_str(data["contact"][v][0])
                    # `sgn` (signature) represents a sequence of subassembly
                    # templates with terminal atomic part suffix
                cp_u = "p_" + str(data["contact"][u][1])
                cp_v = "p_" + str(data["contact"][v][1])
                conn_sgn_lit = Literal(
                    "connection_signature",
                    wrap_args(sa_conc, ti, u, v, sgn_u, sgn_v, cp_u, cp_v)
                )
                assembly_pieces.add(conn_sgn_lit)

    # Retrieve and encode hyper/hyponymy relations
    for (ante, cons), _, _, knowledge_type in agent.lt_mem.kb.entries:
        if knowledge_type != "taxonomy": continue
        subtype_conc = int(ante[0].name.strip("pcls_"))
        supertype_conc = int(cons[0].name.strip("pcls_"))
        assembly_pieces.add(
            Literal("subtype_of", wrap_args(subtype_conc, supertype_conc))
        )

    # Accommodate any pairwise negative labeling feedback
    pairwise_neg_constraints = set()
    for objs, concs in exec_state["recognitions"].items():
        if not isinstance(objs, tuple): continue
        constraint_str = \
            f":- type_committed({objs[0]},{-concs[0]}), type_committed({objs[1]},{-concs[1]}).\n"
        pairwise_neg_constraints.add(constraint_str)

    # Add any constraints from 'property'-type entries in KB. Each constraint
    # will have an associated 'severity score', the maximum of which will be
    # offset from the final compatibility score.
    kb_constraints = set()
    have_conc = agent.lt_mem.lexicon.s2d[("vs", "have")][0]
    have_pred = f"{have_conc[0]}_{have_conc[1]}"
    for ei, ((ante, cons), _, _, knowledge_type) in enumerate(agent.lt_mem.kb.entries):
        # Only include property rules
        if knowledge_type != "property": continue
        # Skip if rule involves any unresolved neologisms
        involved_preds = set()
        for lit in ante+cons:
            if not lit.name.startswith("pcls"): continue
            conc_ind = int(lit.name.split("_")[1])
            if ("pcls", conc_ind) in agent.lt_mem.lexicon.d2s:
                sym = agent.lt_mem.lexicon.d2s[("pcls", conc_ind)][0]
                involved_preds.add(sym)
        if any(pred in agent.lang.unresolved_neologisms for pred in involved_preds):
            continue

        skolem_fn_terms = {
            a for lit in cons for a, _ in lit.args if isinstance(a, tuple)
        }

        if len(skolem_fn_terms):
            # Existential constraints signaled by skolem function terms

            # The scope concept is identified from the only antecedant literal
            assert len(ante) == 1
            scope_conc = int(ante[0].name.split("_")[1])

            # Build constraint rule string, using ASP's #count aggregate for checking
            # (non-)existence
            constraint_str = f"exists_violation({ei}) :- node_sa(N,{scope_conc}), "
            constraint_str += "#count { O : component_obj(O,N)"
            for lit in cons:
                if lit.name == have_pred: continue       # Already incorporated
                conc_ind = int(lit.name.split("_")[1])
                if conc_ind in color_concs:
                    constraint_str += f", color_likely(O,{conc_ind},_)"
                else:
                    constraint_str += f", type_committed(O,{conc_ind})"
            constraint_str += " } = 0.\n"

        else:
            # Universal constraints signaled by lack of skolem function terms
            pseudohypernym_color_conc = agent.lt_mem.lexicon.s2d[("n", "color")][0][1]
                # Needed for handling color equality constraints

            # The scope variable is identified as the only first arguments of
            # 'have' literals; scope concept naturally follows
            scope_var = {lit.args[0][0] for lit in ante if lit.name == have_pred}
            assert len(scope_var) == 1
            scope_var = list(scope_var)[0]
            scope_conc = next(
                int(lit.name.split("_")[1]) for lit in ante
                if len(lit.args) == 1 and lit.args[0][0] == scope_var
            )

            # Build constraint rule string, appropriately translating predicates
            constraint_body = []
            for lit in ante:
                if lit.name == have_pred:
                    arg0 = lit.args[0][0]; arg1 = lit.args[1][0]
                    trans_lits = f"component_obj({arg1},{arg0})"
                    if lit.naf:
                        trans_lits = f"not {trans_lits}"
                else:
                    assert len(lit.args) == 1
                    arg0 = lit.args[0][0]
                    conc_ind = int(lit.name.split("_")[1])
                    if arg0 == scope_var:
                        trans_lits = f"node_sa({arg0},{scope_conc})"
                    else:
                        if conc_ind in color_concs:
                            trans_lits = f"color_likely({arg0},{conc_ind},_)"
                        else:
                            trans_lits = f"type_committed({arg0},{conc_ind})"
                    if lit.naf:
                        trans_lits = f"not {trans_lits}"
                constraint_body.append(trans_lits)
            have_args = {
                lit.args[1][0] for lit in ante
                if lit.name == have_pred and not lit.naf
            }
            if len(have_args) > 0:
                # By adding this, the forall-constraint is disabled for cases
                # where all have-argument objects are already included in some
                # subassembly; i.e., applied only when some component object
                # is yet to be joined
                lit_set = "; ".join(f"fresh_obj({a})" for a in have_args)
                constraint_body.append(f"1{{ {lit_set} }}")
            for lit in cons:
                conc_type, conc_ind = lit.name.split("_")
                if conc_ind.isdigit(): conc_ind = int(conc_ind)
                if (conc_type, conc_ind) == ("pcls", pseudohypernym_color_conc):
                    # Abstract pseudohypernym "color", skip
                    continue

                if len(lit.args) == 1:
                    # More general, unary specifications
                    arg0 = lit.args[0][0]
                    if conc_ind in color_concs:
                        trans_lits = f"color_likely({arg0},{conc_ind},_)"
                    else:
                        trans_lits = f"type_committed({arg0},{conc_ind})"
                    if not lit.naf:
                        # Notice the sign; if positive, violated by not being (proven)
                        # positive
                        trans_lits = f"not {trans_lits}"
                else:
                    # Handling color-related, binary specifications
                    assert len(lit.args) == 2
                    arg0 = lit.args[0][0]; arg1 = lit.args[1][0]
                    if lit.name == have_pred:
                        (obj_var, _), (color_var, _) = lit.args
                        assert Literal(
                            f"pcls_{pseudohypernym_color_conc}", wrap_args(color_var)
                        ) in cons
                        trans_lits = f"color_likely({obj_var},{color_var},_)"
                    else:
                        assert lit.name == "sp_equal"
                        comp_op = "=" if lit.naf else "!="      # Notice flipped polarity
                        trans_lits = f"{arg0} {comp_op} {arg1}"
                constraint_body.append(trans_lits)

            constraint_body = ", ".join(constraint_body)
            constraint_str = f"forall_violation({ei}) :- {constraint_body}.\n"

        kb_constraints.add(constraint_str)

    # Additional constraints introduced by current assembly progress. Selected
    # goal structure must be compliant with existing subassemblies assembled;
    # namely, existing structures must be unifiable with fragments of final
    # goal structure
    ext_info = set()
    for sa_name, sa_graph in exec_state["connection_graphs"].items():
        # Dismiss singletons as they do not provide any additional constraints
        if len(sa_graph) == 1: continue

        # Adding each existing part along with committed recognition
        for ext_obj in sa_graph.nodes:
            if ext_obj in exec_state["recognitions"]:
                # Atomic part type info available, for the part instance
                # was recognized and picked up by agent
                ext_conc = exec_state["recognitions"][ext_obj]
                ext_obj_lit = Literal(
                    "ext_obj", wrap_args(ext_obj, ext_conc)
                )
            else:
                # Atomic part type info not available, most likely because
                # the type of the atomic part is not recognized by agent;
                # instead, the part instance is picked up during demo
                # fragment by user
                ext_obj_lit = Literal("ext_obj", wrap_args(ext_obj))
            ext_info.add(ext_obj_lit)
        # Adding each connection between existing parts
        for ext_obj1, ext_obj2, cps in sa_graph.edges(data="cps"):
            if cps is not None:
                # Contact point info available, list all info
                ext_conn_lit = Literal(
                    "ext_conn", wrap_args(
                        ext_obj1, ext_obj2,
                        f"p_{cps[ext_obj1]}", f"p_{cps[ext_obj2]}"
                    )
                )
            else:
                # Contact point info not available, just include objects
                ext_conn_lit = Literal(
                    "ext_conn", wrap_args(ext_obj1, ext_obj2)
                )
            ext_info.add(ext_conn_lit)

    # Encode assembly target info into ASP fact literal
    target_lit = Literal("build_target", wrap_args(build_target[1]))

    # Keep running 'goal selection' <-> 'join tree planning' cycle until
    # a valid join tree is obtained. In general, the cycle only needs to
    # run once only if we have full, accurate labeling info on existing
    # subassemblies; join tree planning failure only happens when we don't
    # have such info, which happens with language-less player types that
    # must rely on demonstrations as learning signals.

    # Load ASP encoding for selecting & committing to a goal structure
    lp_path = os.path.join(agent.cfg.paths.assets_dir, "planning_encoding")
    commit_ctl = Control(["--warn=none"])
    commit_ctl.configuration.solve.models = 1
    commit_ctl.configuration.solver.seed = agent.cfg.seed
    commit_ctl.load(os.path.join(lp_path, "goal_selection.lp"))

    # Add and ground all the fact literals obtained above
    all_lits = sorted(
        {target_lit} | observations | assembly_pieces | ext_info,
        key=lambda x: x.name
    )
    facts_prg = "".join(str(lit) + ".\n" for lit in all_lits)
    constraints_prg = "".join(kb_constraints) + "".join(pairwise_neg_constraints)
    commit_ctl.add("facts", [], facts_prg + constraints_prg)

    # Optimize with clingo; using the incremental multi-shot optimization
    # procedure supported from clingo 4, as built-in optimization seems to
    # not work for some reason...
    commit_ctl.ground([("base", []), ("facts", [])])
    best_model = None; best_score = -1000       # Low enough baseline
    while True:
        commit_ctl.ground([("check", [Number(best_score)])])
        commit_ctl.assign_external(
            Function("query", [Number(best_score)]), True
        )

        model = None
        with commit_ctl.solve(yield_=True, async_=True) as solve_gen:
            solve_gen.resume()
            model_ready = solve_gen.wait(5)   # Don't spend more than 5 secs
            if model_ready:
                m = solve_gen.model()
                if m is not None:
                    model = m.symbols(atoms=True)
            else:
                solve_gen.cancel()
        
        if model is None:
            # Cannot find a better solution
            break
        else:
            # Doable, raise the score threshold and try again
            commit_ctl.release_external(
                Function("query", [Number(best_score)])
            )
            commit_ctl.configuration.solver.seed = agent.cfg.seed
            best_model = model
            best_score = next(
                atm.arguments[0].number
                for atm in best_model if atm.name=="final_score"
            )

    return best_model

def _tabulate_goal_selection_result(model):
    """
    Helper method factored out for collecting decisions contained in the
    best goal selection result (inferred with ASP) into appropriate data
    structures.
    """
    # Utiltiy method for serializing clingo function arguments into flat
    # string representations
    serialize_node = lambda n: f"n_{n.number}" if n.type == SymbolType.Number \
        else f"{serialize_node(n.arguments[0])}_{n.arguments[1]}"

    assembly_hierarchy = nx.DiGraph()
    obj2node_map = {}
    connect_edges = {}; connection_graph = nx.Graph()
    node_unifications = defaultdict(set)
    part_recognitions = defaultdict(set); hyp_rels = nx.DiGraph()
    for atm in model:
        match atm.name:
            case "node_atomic" | "node_sa_template":
                # Each 'node' literal contains identifiers of its own and its direct
                # parent's, add an edge linking the two to the committed structure 
                # graph
                node_rn = serialize_node(atm.arguments[0])
                node_type = atm.name.split("_")[1]
                conc_ind = atm.arguments[1].number
                assembly_hierarchy.add_node(
                    node_rn, node_type=node_type, conc=conc_ind
                )

                if atm.arguments[0].type == SymbolType.Function:
                    # Add edge between the node and its direct parent
                    assert atm.arguments[0].name == "n"
                    assembly_hierarchy.add_edge(
                        serialize_node(atm.arguments[0].arguments[0]),
                        node_rn
                    )

            case "subtype_of":
                # Retrieving supertype/subtype relations
                subtype_conc = atm.arguments[0].number
                supertype_conc = atm.arguments[1].number
                hyp_rels.add_edge(supertype_conc, subtype_conc)

            case "fill_node":
                # Decision as to use which recognized object to fill which
                # atomic node in the assembly hierarchy graph. This mapping
                # is tentative, as assignments may be interchanged within
                # same types later for prioritizing joins of filled nodes
                # over unfilled nodes.
                obj_name = atm.arguments[0].name
                node_rn = serialize_node(atm.arguments[1])
                obj2node_map[obj_name] = node_rn

            case "type_committed":
                # Decision as to recognize each object as which (visually
                # licensed) part type; all entailed supertypes/subtypes
                # present, first collect them all and they will be filtered
                # later
                obj_name = atm.arguments[0].name
                part_conc = atm.arguments[1].number
                part_recognitions[obj_name].add(part_conc)

            case "to_connect":
                # Derived assembly connection to make between nodes
                node_u_rn = serialize_node(atm.arguments[0])
                node_v_rn = serialize_node(atm.arguments[1])
                cp_u = atm.arguments[2].name
                cp_v = atm.arguments[3].name
                connect_edges[(node_u_rn, node_v_rn)] = (cp_u, cp_v)
                connect_edges[(node_v_rn, node_u_rn)] = (cp_v, cp_u)
                connection_graph.add_edge(node_u_rn, node_v_rn)

            case "must_unify" | "may_unify":
                # Unification between objects already used as subassembly
                # component in environment vs. matching nodes in hierarchy
                ex_obj = atm.arguments[0].name
                node_rn = serialize_node(atm.arguments[1])
                node_unifications[ex_obj].add(node_rn)

    # Note: This assembly hierarchy structure object also serves to provide
    # explanation which part is required for building which subassembly
    # (that is direct parent in the hierarchy)

    # Part type requirements of each atomic node in the selected template
    atomic_node_concs = {
        n: data["conc"]
        for n, data in assembly_hierarchy.nodes(data=True)
        if data["node_type"] == "atomic"
    }

    # Leave only the most specific supertypes
    part_recognitions = {
        obj: list(all_concs)[0] if len(all_concs) == 1 else next(
            c for c in all_concs if hyp_rels.out_degree[c] == 0
        )
        for obj, all_concs in part_recognitions.items()
    }

    tabulated_results = (
        assembly_hierarchy, obj2node_map, connect_edges, connection_graph,
        node_unifications, atomic_node_concs, part_recognitions, hyp_rels
    )
    return tabulated_results

def _match_existing_subassemblies(
    connection_graph, obj2sa_map, unification_options,
    atomic_node_concs, hyp_rels, exec_state
):
    """
    Helper method factored out for establishing correspondence between
    atomic nodes in assembly hierarchy vs. atomic part objects in existing
    subassemblies, based on graph matching and currently committed part
    type recognitions.
    """
    recognitions = exec_state["recognitions"]       # Shortcut
    get_supertypes = lambda c: (nx.ancestors(hyp_rels, c) | {c}) \
        if c in hyp_rels else {c}
        # Helper for obtaining all supertypes of subtype c if any exists,
        # including self

    possible_mappings = defaultdict(list)
    uniquely_unified = {
        ex_obj: list(nodes)[0] for ex_obj, nodes in unification_options.items()
        if len(nodes) == 1
    }

    anchored_sas = {
        sa for ex_obj in uniquely_unified
        if len(exec_state["connection_graphs"][(sa := obj2sa_map[ex_obj])]) > 1
    }
    lifted_sas = {
        sa for sa in set(exec_state["connection_graphs"]) - anchored_sas
        if len(exec_state["connection_graphs"][sa]) > 1
    }
    conn_graph_match = copy.deepcopy(connection_graph)
    for n in conn_graph_match:
        if n in uniquely_unified.values():
            conn_graph_match.nodes[n]["unif"] = n
    # First match and process existing subassemblies which have uniquely
    # unifiable component objects
    for sa in anchored_sas:
        sa_graph_match = exec_state["connection_graphs"][sa].to_undirected()
        for ex_obj in sa_graph_match:
            if ex_obj in uniquely_unified:
                unified_node = uniquely_unified[ex_obj]
                sa_graph_match.nodes[ex_obj]["unif"] = unified_node
        matcher = nx.isomorphism.ISMAGS(
            conn_graph_match, sa_graph_match,
            node_match=nx.isomorphism.categorical_node_match("unif", "*")
        )
        for ism in matcher.find_isomorphisms(symmetry=False):
            # Discard the matching if any existing object with known type
            # doesn't typecheck
            if any(
                ex_obj in recognitions and \
                    atomic_node_concs[n] not in get_supertypes(recognitions[ex_obj])
                for n, ex_obj in ism.items()
            ): continue
            possible_mappings[sa].append(ism)
    # Punch out uniquely unified nodes
    conn_graph_match.remove_nodes_from(uniquely_unified.values())
    # Now match and process existing subassemblies that are not anchored
    # to any nodes in connection graph, with remaining `conn_graph_match`
    for sa in lifted_sas:
        sa_graph_match = exec_state["connection_graphs"][sa].to_undirected()
        matcher = nx.isomorphism.ISMAGS(conn_graph_match, sa_graph_match)
        for ism in matcher.find_isomorphisms(symmetry=False):
            # Typechecking again
            if any(
                ex_obj in recognitions and \
                    atomic_node_concs[n] not in get_supertypes(recognitions[ex_obj])
                for n, ex_obj in ism.items()
            ): continue
            possible_mappings[sa].append(ism)

    return possible_mappings

def _divide_and_conquer(
    compression_sequence, connection_graph, recognitions,
    atomic_node_concs, contacts, part_names, cp_names,
    current_progress, scope_entailments, verified_joins,
    assets_dir, seed
):
    """
    Helper method factored out for piecewise planning of a join sequence
    (which is actually a linearization of the underlying partial order join
    plan, see above). Divide the large problem into a sequence of smaller
    subproblems and conquer each subproblem. Start from scratch if a bad
    subproblem division led to deadend.
    """
    # Test if the planning problem is outright unsolvable with given premise
    # and the history of current episode; in particular, if the premise
    # would enforce a join that's proven invalid by teacher
    unification_choice_inv = {
        n: ex_obj for ex_obj, n in current_progress[1].items()
    }
    for u, v in connection_graph.edges:
        if u not in unification_choice_inv: continue
        if v not in unification_choice_inv: continue
        obj_u = unification_choice_inv[u]
        obj_v = unification_choice_inv[v]
        conc_u = atomic_node_concs[u]
        conc_v = atomic_node_concs[v]
        if recognitions.get((obj_u, obj_v)) == (-conc_u, -conc_v) or \
            recognitions.get((obj_v, obj_u)) == (-conc_v, -conc_u):
            # Pairwise negative feedback from teacher exists, such that
            # it shouldn't hold that obj_u is an instance of conc_u and
            # obj_v of conc_v simultaneously
            return None, 0, 0, set()

    # Inverse direction mapping of `atomic_node_concs`, from concept to
    # corresponding node group
    atomic_nodes_by_conc = {
        k: {n for n, _ in v} for k, v in groupby(
            sorted(atomic_node_concs.items(), key=lambda x: x[1]),
            key=lambda x: x[1]
        )
    }

    # Ground-truth oracle for querying whether two atomic parts (or their
    # subassemblies) can be disassembled from their final positions without
    # collision along six directions (x+/x-/y+/y-/z+/z-). Coarse abstraction
    # of low-level motion planner, which need to be integrated if such
    # oracle is not available in real use cases.
    knowledge_path = os.path.join(assets_dir, "domain_knowledge")
    with open(f"{knowledge_path}/collision_table.yaml") as yml_f:
        collision_table = yaml.safe_load(yml_f)

    # Method for obtaining a new instance of physical collision 'theory'
    # propagator
    def new_propagator():
        return _PhysicalAssemblyPropagator(
            (connection_graph, contacts), atomic_node_concs,
            (collision_table, part_names, cp_names),
            scope_entailments, verified_joins
        )
    # Instantiate a new propagator, which will establish a mapping between
    # connection graph nodes and collision table instances.
    collision_checker = new_propagator()

    # Compress the connection graph part by part, randomly selecting chunks
    # of parts or subassemblies to form smaller planning problems
    remaining_subproblems = deque(
        _planning_subproblems(
            compression_sequence, connection_graph, current_progress
        )
    )

    # At any point, test if initial configuration of a subproblem violates
    # any scope entailment from start, in which case the subproblem is
    # fundamentally unsolvable. Keep testing with scope entailments being
    # continuously updated.
    def subproblem_invalid(subproblem):
        chunks_assembled = subproblem[1]
        viol_ents = set()
        for ante, cons in scope_entailments:
            # Return true if ante is included in some chunk but cons is not
            # included in the same chunk
            viol_ents |= {
                (ante, cons) for atomics in chunks_assembled.values()
                if ante <= atomics and cons not in atomics
            }
        return viol_ents
    sp_checks = set.union(*[subproblem_invalid(sp) for sp in remaining_subproblems])
    if len(sp_checks) > 0:
        # Some subproblem proven to be unsolvable, return with violated
        # scope entailments
        return None, 0, 0, sp_checks

    # Tracking how many piecewise planning attempts and oracle calls were made
    planning_attempts = 1; query_count = 0

    # For remembering sequences of chunking that has proven to reach deadend
    # for this particular subproblem
    current_chunk_sequence = ()
    invalid_chunk_sequences = set()
    # Sequence of joins to be made
    join_sequence_subprob = []      # For each subproblem
    join_sequence = []              # Accumulated across subproblems
    # Tracking remaining nodes per atomic part type
    node_pool_checkpoint = copy.deepcopy(atomic_nodes_by_conc)
    node_pool = copy.deepcopy(node_pool_checkpoint)
    # Index of subproblem being solved
    step_ind = 0
    # Continue until the target structure is fully compressed and planned for    
    while len(remaining_subproblems) > 0:
        # Pop the next pending subproblem
        compression_graph, chunks_assembled = remaining_subproblems.popleft()
        # Inverse of chunks_assembled, mapping from atomic to affiliated chunk
        chunk_affiliations = {
            n: chunk for chunk, atomics in chunks_assembled.items()
            for n in atomics
        }

        if len(compression_graph) == 1:
            # Subproblem completely solved; collect the obtained solution
            # and prepare for the next subproblem (if any)
            if len(join_sequence_subprob) > 0:
                # Renaming the final product for the subproblem, if nonempty
                # join sequence is obtained for this subproblem
                join_sequence_subprob[-1] = \
                    join_sequence_subprob[-1][:-1] + (f"c{step_ind}",)
            collision_checker = new_propagator()
            join_sequence += join_sequence_subprob
            join_sequence_subprob = []
            current_chunk_sequence = ()
            invalid_chunk_sequences = set()
            step_ind += 1
            continue

        # Continue joining groups of nodes in compression graph that are valid
        # according to (any) scope entailments

        # Candidate chunks of nodes and involved atomics, with seeds initialized
        # as all edges in compression graph
        get_atomics = lambda n: chunks_assembled[n] \
            if n in chunks_assembled else {n}
        candidate_chunks = {
            frozenset([n1, n2]): get_atomics(n1) | get_atomics(n2)
            for n1, n2 in compression_graph.edges
        }
        # Complete any candidate chunk that violates any scope entailment
        # by continuously adding entailment consequent nodes
        while True:
            new_candidate_chunks = {}
            for chunk, atomics in candidate_chunks.items():
                for ante, cons in scope_entailments:
                    if ante <= atomics and cons not in atomics:
                        # Consequent node needs to be added to the current
                        # chunk being inspected
                        chunk = chunk | {chunk_affiliations.get(cons, cons)}
                        atomics = atomics | get_atomics(cons)
                else:
                    # Current chunk is good to go
                    new_candidate_chunks[chunk] = atomics

            if set(new_candidate_chunks) == set(candidate_chunks):
                # Fixpoint, can break
                break
            else:
                # Run another round of chunk inspection
                candidate_chunks = new_candidate_chunks

        # Filter out chunks that would lead to invalid chunk sequence
        candidate_chunks = [
            chk for chk in candidate_chunks
            if current_chunk_sequence + (frozenset(chk),) not in \
                invalid_chunk_sequences
        ]
        if len(candidate_chunks) > 0:
            # Sample a chunk among the valid candidates
            chunk_selected = random.sample(candidate_chunks, 1)[0]
            chunk_subgraph = compression_graph.subgraph(chunk_selected)
        else:
            # No valid chunks, deadend reached
            if len(current_chunk_sequence) == 0:
                # Current sequence length zero indicates the subproblem
                # is unsolvable, possibly relying on bad premises.
                return None, planning_attempts, query_count, scope_entailments
            else:
                # Register current chunk sequence as invalid so that it
                # won't be tried again. Update tracker states and restart.
                planning_attempts += 1
                invalid_chunk_sequences.add(current_chunk_sequence)
                remaining_subproblems = deque(
                    _planning_subproblems(
                        compression_sequence, connection_graph, current_progress
                    )[step_ind:]
                )
                sp_checks = set.union(*[subproblem_invalid(sp) for sp in remaining_subproblems])
                if len(sp_checks) > 0:
                    # Some subproblem proven to be unsolvable, return with violated
                    # scope entailments
                    return None, planning_attempts, query_count, sp_checks
                else:
                    # Start again
                    current_chunk_sequence = ()
                    join_sequence_subprob = []
                    node_pool = copy.deepcopy(node_pool_checkpoint)
                    continue

        # Remove instances in the selected chunk from node_pool
        for n in chunk_selected:
            if n not in atomic_node_concs: continue     # Non-atomic
            conc_n = atomic_node_concs[n]
            if n in node_pool[conc_n]: node_pool[conc_n].remove(n)

        # Load ASP encoding for planning sequence of atomic actions
        collision_checker = new_propagator()
        lp_path = os.path.join(assets_dir, "planning_encoding")
        plan_ctl = Control(["--warn=none"])
        plan_ctl.register_propagator(collision_checker)
        plan_ctl.configuration.solve.models = 0
        plan_ctl.configuration.solve.opt_mode = "opt"
        plan_ctl.configuration.solver.seed = seed
        plan_ctl.load(os.path.join(lp_path, "assembly_sequence.lp"))

        # Committed goal condition expressed as ASP fact literals
        all_lits = set()
        for u, v in chunk_subgraph.edges:
            for n in [u, v]:
                if n in chunks_assembled:
                    # Each atomic part ~ subassembly relation as initial fluent
                    for n_obj in chunks_assembled[n]:
                        all_lits.add(
                            Literal("init", wrap_args(("part_of", [n_obj, n])))
                        )
                else:
                    # Each (non-)object; annotate type and add initial fluent
                    # Re-use the object name handle for the singleton subassembly
                    all_lits.add(Literal("init", wrap_args(("part_of", [n, n]))))

            # Each valid (non-)object pair to connect
            u_objs = chunks_assembled[u] if u in chunks_assembled else {u}
            v_objs = chunks_assembled[v] if v in chunks_assembled else {v}
            for o_u, o_v in product(u_objs, v_objs):
                if (o_u, o_v) in connection_graph.edges:
                    all_lits.add(Literal("to_connect", wrap_args(o_u, o_v)))

        all_lits = sorted(all_lits, key=lambda x: x.name)
        facts_prg = "".join(str(lit) + ".\n" for lit in all_lits)
        plan_ctl.add("facts", [], facts_prg)
        plan_ctl.ground([("base", []), ("facts", [])])

        # Multi-shot solving: incrementing the number of action steps in the
        # plan, whilst optimizing for the number of consecutive join actions
        # made without dropping subassemblies
        step = 0; optimal_model = None
        while True:
            if step > len(chunk_selected):
                # If reached here, ASP problem has no valid solution, probably
                # due to faulty subgraph sampling; update the set of assembly
                # entailments as appropriate

                # Break from this while loop to restart planning (after due
                # updates of scope entailments)
                break

            if step > 0:
                plan_ctl.release_external(Function("query", [Number(step)]))

            plan_ctl.ground([
                ("step", [Number(step)]),
                ("check", [Number(step+1)])
            ])
            plan_ctl.assign_external(Function("query", [Number(step+1)]), True)

            with plan_ctl.solve(yield_=True) as solve_gen:
                models = [
                    (m.symbols(atoms=True), m.cost) for m in solve_gen
                ]
                if len(models) > 0:
                    optimal_model = models[-1]
                    break
                else:
                    step += 1
            
            plan_ctl.configuration.solver.seed = seed

        # Update list of known scope entailments from invalid joins witnessed
        # by the collision check propagator
        for ante_i, cons_i in collision_checker.scope_entailments:
            ante_n = frozenset(collision_checker.inst2node[i] for i in ante_i)
            cons_n = collision_checker.inst2node[cons_i]
            scope_entailments.add((ante_n, cons_n))
        # Similarly, update list of verified joins witnessed
        for cached_s1, cached_s2 in collision_checker.verified_joins:
            nodes_s1 = frozenset(collision_checker.inst2node[i] for i in cached_s1)
            nodes_s2 = frozenset(collision_checker.inst2node[i] for i in cached_s2)
            subsumed = set()
            for cached_s1, cached_s2 in verified_joins:
                if nodes_s1 >= cached_s1 and nodes_s2 >= cached_s2 or \
                    nodes_s1 >= cached_s2 and nodes_s2 >= cached_s1:
                    # Can safely remove this already cached join as this
                    # will be entailed
                    subsumed.add(frozenset((cached_s1, cached_s2)))
                if nodes_s1 <= cached_s1 and nodes_s2 <= cached_s2 or \
                    nodes_s1 <= cached_s2 and nodes_s2 <= cached_s1:
                    # Already entailed, no need to add
                    break
            else:
                verified_joins -= subsumed
                verified_joins.add(frozenset((nodes_s1, nodes_s2)))

        # Tracking sequence of selected chunks, so as to remember invalid
        # sequences that led to deadend states and prevent following the
        # exact same sequence again
        current_chunk_sequence = \
            current_chunk_sequence + (frozenset(chunk_selected),)

        query_count += collision_checker.query_count
        if optimal_model is None:
            # Planning failure, start again from scratch. Update tracker states
            # and restart.
            planning_attempts += 1
            invalid_chunk_sequences.add(current_chunk_sequence)
            remaining_subproblems = deque(
                _planning_subproblems(
                    compression_sequence, connection_graph, current_progress
                )[step_ind:]
            )
            sp_checks = set.union(*[subproblem_invalid(sp) for sp in remaining_subproblems])
            if len(sp_checks) > 0:
                # Some subproblem proven to be unsolvable, return with violated
                # scope entailments
                return None, planning_attempts, query_count, sp_checks
            else:
                # Start again
                current_chunk_sequence = ()
                join_sequence_subprob = []
                node_pool = copy.deepcopy(node_pool_checkpoint)
                continue

        # Project model into join action sequence
        projection = sorted([
            atm for atm in optimal_model[0]
            if atm.name=="occ" and atm.arguments[0].name=="join"
        ], key=lambda atm: atm.arguments[1].number)
        projection = [
            atm.arguments[0].arguments for atm in projection
        ]

        # Extract action parameters for each join action
        resultant_ind_offset = len(join_sequence_subprob)
        arg_val = lambda a: a.name if a.type == SymbolType.Function \
            else f"i{step_ind}_{a.number+resultant_ind_offset}"
        for i, (s1, n1, s2, n2) in enumerate(projection):
            s1, n1 = arg_val(s1), arg_val(n1)
            s2, n2 = arg_val(s2), arg_val(n2)
            # Naming resultant subassembly
            resultant_name = f"i{step_ind}_{i+resultant_ind_offset}"
            # Append joining step info
            join_sequence_subprob.append(
                (s1, n1, s2, n2, resultant_name)
            )

        # Update compression graph accordingly
        resultant_name = f"i{step_ind}_{len(join_sequence_subprob)-1}"
        _compress_chunk(compression_graph, chunk_selected, resultant_name)

        # Remember which components constitute the new chunk
        chunks_assembled[resultant_name] = {
            n for s in chunk_selected if s in chunks_assembled
            for n in chunks_assembled[s]
        } | {
            n for n in chunk_selected if n not in chunks_assembled
        }
        # Remove old chunks that are now not a node in compression graph
        chunks_assembled = {
            chunk: atomics for chunk, atomics in chunks_assembled.items()
            if chunk in compression_graph
        }

        # Save the node pool status
        node_pool_checkpoint = node_pool

        # Push back the reduced subproblem for continuation
        remaining_subproblems.appendleft(
            (compression_graph, chunks_assembled)
        )

    # What we have figured out, stored in `join_sequence`, is actually one
    # of the possible linearizations of the valid underlying partial-order
    # joining plan. Recover the partial-order plan (represented as tree),
    # then we will later find the linearization that will postpone
    # involvements of non-objects as much as possible.
    assert join_sequence is not None
    join_tree = nx.DiGraph()
    for s1, n1, s2, n2, res in join_sequence:
        join_tree.add_edge(s1, res, join_by=n1)
        join_tree.add_edge(s2, res, join_by=n2)

    return join_tree, planning_attempts, query_count, scope_entailments

def _compress_chunk(compression_graph, chunk, name):
    """
    Helper method for updating compression graph by replacing the collection
    of nodes with a new node representing the subassembly newly assembled
    from the specified collection
    """
    compression_graph.add_node(name)
    edges_to_add = []
    for u, v in list(compression_graph.edges):
        if u in chunk:
            if u in compression_graph: compression_graph.remove_node(u)
            if v not in chunk: edges_to_add.append((name, v))
        if v in chunk:
            if v in compression_graph: compression_graph.remove_node(v)
            if u not in chunk: edges_to_add.append((u, name))
    compression_graph.add_edges_from(edges_to_add)

def _planning_subproblems(
    compression_sequence, connection_graph, current_progress
):
    """
    Helper method for obtaining a set of planning subproblems from the
    provided compression sequence and connection graph, while accounting
    for current assembly progress
    """
    compression_sequence = copy.deepcopy(compression_sequence)
    connection_status, node_unifications = current_progress
    chunks_status = {
        sa: {node_unifications[ex_obj] for ex_obj in sa_graph}
        for sa, sa_graph in connection_status.items()
        if len(sa_graph) > 1            # Dismiss singletons
    }                       # Node grouping state tracker across sequence

    subproblems = []        # Return value

    # Method for flattening nonatomic node to set of contained atomic nodes
    flatten_node = lambda n, cmp_seq: {n} if n not in cmp_seq \
        else set.union(*[flatten_node(m, cmp_seq) for m in cmp_seq[n]])

    # Obtain graph representation of the assembly subprogram for the
    # current progress, by replaying the compression sequence up to
    # the previous step, then taking subgraph for the current step
    compression_graph = connection_graph.copy()
    intermediate_chunks = {
        sa_node: f"c{i}"
        for i, (sa_node, _) in enumerate(compression_sequence)
    }
    for step_ind in range(len(compression_sequence)):
        # First compress by intermediate chunks
        compressed_nodes = compression_sequence[step_ind][1]
        compressed_nodes = {
            intermediate_chunks.get(n, n) for n in compressed_nodes
        }
        subproblem_graph = compression_graph.subgraph(compressed_nodes).copy()
        _compress_chunk(compression_graph, compressed_nodes, f"c{step_ind}")
            # Compress for the remaining steps

        # Chunks assembled so far involved in the subproblem, with their
        # component atomics
        chunks_assembled = {
            intermediate_chunks[sa_node]: set.union(*[
                flatten_node(n, dict(compression_sequence))
                for n in component_nodes
            ])
            for sa_node, component_nodes in compression_sequence[:step_ind]
            if intermediate_chunks[sa_node] in compressed_nodes
        }

        # Account for any progress made in environment so far
        node_components = chunks_assembled | {
            n: {n} for n in subproblem_graph if n not in chunks_assembled
        }
        for sa, sa_atomics in chunks_status.items():
            # Finding the set of nodes in compression graph covered by
            # this existing subassembly
            subsumed_nodes = [
                n for n, atomics in node_components.items()
                if atomics <= sa_atomics
            ]
            if len(subsumed_nodes) == 0: continue

            # Further compress set of nodes covered by this existing subassembly
            _compress_chunk(subproblem_graph, subsumed_nodes, sa)
            chunks_assembled[sa] = sa_atomics
            for n in subsumed_nodes:
                if n in chunks_assembled and n != sa:
                    # If n == sa, allow 'overwriting'
                    chunks_assembled.pop(n)

        if len(subproblem_graph) > 1:
            # Meaningful joins are yet to be made out of this subproblem,
            # resulting in the final product with name as listed in
            # `intermediate_chunks`. Update `chunks_status`.
            sa_node = compression_sequence[step_ind][0]
            chunk_atomics = set()
            for n in subproblem_graph:
                if n in chunks_status:
                    chunk_atomics |= chunks_status.pop(n)
                else:
                    chunk_atomics.add(n)
            chunks_status[intermediate_chunks[sa_node]] = chunk_atomics

        # Now we can safely replace `compression_sequence[step_ind][1]` with
        # the set of all atomics covered by this subproblem
        compression_sequence[step_ind] = (
            compression_sequence[step_ind][0],
            list(set.union(*[
                chunks_assembled[n] if n in chunks_assembled else {n}
                for n in subproblem_graph
            ]))
        )

        # Append to the list of subproblems
        subproblems.append((subproblem_graph, chunks_assembled))

    return subproblems

class _PhysicalAssemblyPropagator:
    def __init__(
        self, connections, atomic_node_concs, cheat_sheet,
        scope_entailments, verified_joins
    ):
        connection_graph, contacts = connections
        collision_table, part_names, cp_names = cheat_sheet

        self.connection_graph = connection_graph

        # Inverses of mappings stored in the ground-truth oracle data
        part_group_inv = {
            v: k for k, vs in collision_table["part_groups"].items()
            for v in vs
        }
        cp_indexing_inv = {
            v: k for k, v in collision_table["contact_points"].items()
        }
        template_parts_inv = {
            tuple(signature): inst
            for inst, signature in collision_table["part_instances"].items()
        }

        # (Injective) Mapping of individual nodes to corresponding atomic part
        # instances in the collision table
        self.node2inst = {}; self.inst2node = {}
        while len(self.node2inst) != len(atomic_node_concs):
            for n in atomic_node_concs:
                # If already mapped
                if n in self.node2inst: continue

                # See if uniquely determined by part concept alone
                conc_n = atomic_node_concs[n]
                conc_n = part_group_inv[part_names[conc_n]]
                conc_sgn = (conc_n,) + (None,)*3
                if conc_sgn in template_parts_inv:
                    # Match found
                    inst = template_parts_inv[conc_sgn]
                    self.node2inst[n] = inst
                    self.inst2node[inst] = n
                    continue

                # Matching with full signature if reached here
                for (u, v), (cp_u, cp_v) in contacts.items():
                    if n not in (u, v): continue

                    cp_u = int(re.findall(r"p_(.*)$", cp_u)[0])
                    cp_v = int(re.findall(r"p_(.*)$", cp_v)[0])
                    cp_u = cp_indexing_inv[cp_names[cp_u]]
                    cp_v = cp_indexing_inv[cp_names[cp_v]]

                    other = v if n == u else u
                    if other not in self.node2inst: continue
                    inst_other = self.node2inst[other]

                    cp_n = cp_u if n == u else cp_v
                    cp_other = cp_v if n == u else cp_u

                    full_sgn = (conc_n, inst_other, cp_n, cp_other)
                    if full_sgn in template_parts_inv:
                        # Match found
                        inst = template_parts_inv[full_sgn]
                        self.node2inst[n] = inst
                        self.inst2node[inst] = n
                        break

        # Collision matrix translated to accommodate the context
        self.collision_table = {
            tuple(int(inst) for inst in pair.split(",")): set(colls)
            for pair, colls in collision_table["pairwise_collisions"].items()
        }
        self.collision_table = self.collision_table | {
            (i2, i1): {-coll_dir for coll_dir in colls}
            for (i1, i2), colls in self.collision_table.items()
        }           # Also list reverse directions in lower triangle

        # For tracking per-thread status
        self.assembly_status = None

        # For caching inclusion-minimal 'scope entailments', such that if
        # the 'antecedent' part is included in one side of the join, the
        # 'consequent' part must also be included in that side so as to
        # avoid planning failure by lack of collision-free paths
        self.scope_entailments = {
            (frozenset([self.node2inst[n] for n in ante]), self.node2inst[cons])
            for ante, cons in scope_entailments
        }
        # Similarly, caching inclusion-maximal joins that are verified to
        # have collision-free assembly paths
        self.verified_joins = {
            frozenset((
                frozenset([self.node2inst[n] for n in nodes1]),
                frozenset([self.node2inst[n] for n in nodes2])
            ))
            for nodes1, nodes2 in verified_joins
        }

        # Count of how many times the oracle was queried; in case we do not
        # have any access to oracle and need to resort to some path planning
        # algorithm, this number could bear more practical significance!
        self.query_count = 0

    def init(self, init):
        # Storage of atoms of interest by argument & time step handle
        self.join_actions_a2l = {}
        self.join_actions_l2a = defaultdict(set)
        self.part_of_fluents_a2l = {}
        self.part_of_fluents_l2a = defaultdict(set)

        arg_val = lambda a: a.name if a.type == SymbolType.Function \
            else a.number

        for atm in init.symbolic_atoms:
            # Watch occ(join(...),t) atoms and holds(part_of(...),t) atoms
            # and establish correspondence between solver literals and
            # action/fluent arguments
            pred = atm.symbol.name
            arg1 = atm.symbol.arguments[0]

            occ_join = pred == "occ" and arg1.name == "join"
            holds_part_of = pred == "holds" and arg1.name == "part_of" \
                and atm.symbol.positive     # Dismiss -holds() atoms

            if occ_join:
                lit = init.solver_literal(atm.literal)
                init.add_watch(lit)
                t = atm.symbol.arguments[1].number
                s1 = arg_val(arg1.arguments[0])
                n1 = arg_val(arg1.arguments[1])
                s2 = arg_val(arg1.arguments[2])
                n2 = arg_val(arg1.arguments[3])

                self.join_actions_a2l[(s1, n1, s2, n2, t)] = lit
                self.join_actions_l2a[lit].add((s1, n1, s2, n2, t))

            if holds_part_of:
                lit = init.solver_literal(atm.literal)
                init.add_watch(lit)
                t = atm.symbol.arguments[1].number
                n = arg_val(arg1.arguments[0])
                s = arg_val(arg1.arguments[1])

                self.part_of_fluents_a2l[(n, s, t)] = lit
                self.part_of_fluents_l2a[lit].add((n, s, t))

        if self.assembly_status is None:
            # Initialize per-thread status tracker
            self.assembly_status = [
                {
                    "joins": {},
                    "parts": defaultdict(lambda: defaultdict(set)),
                    "parts_inv": defaultdict(dict)
                }
                for _ in range(init.number_of_threads)
            ]

    def propagate(self, control, changes):
        status = self.assembly_status[control.thread_id]

        updated = set()     # Tracks which join actions need checking
        for lit in changes:
            # Update ongoing assembly status for this thread accordingly
            if lit in self.join_actions_l2a:
                # Join of s1 & s2 happens at time step t
                for s1, n1, s2, n2, t in self.join_actions_l2a[lit]:
                    status["joins"][t] = (s1, n1, s2, n2)
                    updated.add((s1, t))
                    updated.add((s2, t))

            if lit in self.part_of_fluents_l2a:
                # Node n is part of subassembly s at time step t
                for n, s, t in self.part_of_fluents_l2a[lit]:
                    if n in status["parts_inv"][t]:
                        # An atomic (non-)object cannot belong to more than one
                        # subassembly at each time step! Not explicitly prohibited
                        # by some program rule (only entailed) but let's add the
                        # set of fluents as a nogood in advance.
                        s_dup = status["parts_inv"][t][n]
                        dup_lit = self.part_of_fluents_a2l[(n, s_dup, t)]
                        may_continue = control.add_nogood([lit, dup_lit]) \
                            and control.propagate()
                        if not may_continue: return

                    status["parts"][t][s].add(n)
                    status["parts_inv"][t][n] = s
                    updated.add((s, t))

        # After the round of updates, test if each join action would entail
        # a collision-free path
        for t, (s1, n1, s2, n2) in status["joins"].items():
            if not ((s1, t) in updated or (s2, t) in updated):
                # Nothing about this join action has been updated, can skip
                continue

            nodes_s1 = frozenset(status["parts"][t][s1])
            nodes_s2 = frozenset(status["parts"][t][s2])
            insts_s1 = frozenset(self.node2inst[n] for n in nodes_s1)
            insts_s2 = frozenset(self.node2inst[n] for n in nodes_s2)

            # Checking if join can be verified to be valid without querying
            # the oracle. Note that if either of nodes_s1 or nodes_s2 is empty,
            # the vacuous join will always be treated as 'not impossible';
            # i.e., not ruled out due to physical collision.
            verified_by_cache = False
            for cached_s1, cached_s2 in self.verified_joins:
                if insts_s1 <= cached_s1 and insts_s2 <= cached_s2 or \
                    insts_s1 <= cached_s2 and insts_s2 <= cached_s1:
                    verified_by_cache = True
                    break

            if verified_by_cache:
                join_possible = True
            else:
                entailment_violating = any(
                    (insts_s1 >= ante and insts_s2 >= {cons}) or \
                        (insts_s2 >= ante and insts_s1 >= {cons})
                    for ante, cons in self.scope_entailments
                )
                if entailment_violating:
                    # Some subset pair recognized and cached as invalid
                    join_possible = False
                else:
                    # Test against table needed; bipartite atomic-atomic
                    # collision check, then union across the pairwise checks
                    # to obtain collision test result
                    collision_directions = set.union(*[
                        self.collision_table.get((i_s1, i_s2), set())
                        for i_s1, i_s2 in product(insts_s1, insts_s2)
                    ])
                    self.query_count += 1       # Update count
                    # Collision-free join is possible when the resultant
                    # union does not have six members, standing for each
                    # direction of assembly, hence a feasible join path
                    join_possible = len(collision_directions) < 6

            if join_possible:
                # Store the join as verified, if there isn't any stricter
                # verified join already stored
                if not verified_by_cache:
                    subsumed = set()        # Delete any weaker (subsumed) joins
                    for cached_s1, cached_s2 in self.verified_joins:
                        if insts_s1 >= cached_s1 and insts_s2 >= cached_s2 or \
                            insts_s1 >= cached_s2 and insts_s2 >= cached_s1:
                            subsumed.add(frozenset((cached_s1, cached_s2)))
                    self.verified_joins -= subsumed
                    self.verified_joins.add(frozenset((insts_s1, insts_s2)))
            else:
                # Join of this subassembly pair proved to be unreachable
                # while including current object members; add nogood
                # as appropriate and return
                if not entailment_violating:
                    minimal_pair = self.analyze_collision((nodes_s1, nodes_s2))
                    if minimal_pair is not None:
                        # If one side of the minimal pair is an atomic singleton,
                        # we can treat the pair as a scope entailment where the
                        # singleton side is its consequent and the other side is
                        # its antecedent.
                        insts1, insts2 = minimal_pair
                        if len(insts1) == 1:
                            self.scope_entailments.add((insts2, list(insts1)[0]))
                        if len(insts2) == 1:
                            self.scope_entailments.add((insts1, list(insts2)[0]))

                join_lit = self.join_actions_a2l[(s1, n1, s2, n2, t)]
                part_of_lits = [
                    self.part_of_fluents_a2l[(n, s1, t)] for n in nodes_s1
                ] + [
                    self.part_of_fluents_a2l[(n, s2, t)] for n in nodes_s2
                ]
                nogood = [join_lit] + part_of_lits
                
                may_continue = control.add_nogood(nogood) \
                    and control.propagate()
                if not may_continue:
                    return

    def undo(self, thread_id, assignment, changes):
        status = self.assembly_status[thread_id]

        for lit in changes:
            # Update ongoing assembly status for this thread accordingly
            if lit in self.join_actions_l2a:
                # Join of s1 & s2 actually doesn't happen at time step t
                for s1, n1, s2, n2, t in self.join_actions_l2a[lit]:
                    if t in status["joins"]: 
                        assert status["joins"][t] == (s1, n1, s2, n2)
                        del status["joins"][t]

            if lit in self.part_of_fluents_l2a:
                # Node n is actually not a part of s at time step t
                for n, s, t in self.part_of_fluents_l2a[lit]:
                    if n in status["parts"][t][s]:
                        status["parts"][t][s].remove(n)
                        del status["parts_inv"][t][n]

    def analyze_collision(self, nodes_pair):
        """
        Helper method factored out for analyzing an object set pair that
        caused collision, finding an inclusion-minimal object set pair
        """
        nodes1, nodes2 = nodes_pair

        # Iterate over the lattice of pair of possible subset sizes of
        # nodes1, nodes2, sorted so that 'smaller' size pairs are always
        # processed earlier than 'larger' ones
        pair_sizes = sorted(
            (min(size1, size2) + 1, size1 + 1, size2 + 1)
            for size1, size2 in product(range(len(nodes1)), range(len(nodes2)))
        )
        for _, size1, size2 in pair_sizes:
            if size1 == size2 == 1:
                # Trivially valid joins, given legitimate target structure
                continue

            # Obtain every possible subsets of nodes1, nodes2 of resp. sizes
            # and iterate across possible pairs
            subsets1 = combinations(nodes1, size1)
            subsets2 = combinations(nodes2, size2)
            for nodes_s1, nodes_s2 in product(subsets1, subsets2):
                nodes_s1 = frozenset(nodes_s1)
                nodes_s2 = frozenset(nodes_s2)
                insts_s1 = frozenset(self.node2inst[n] for n in nodes_s1)
                insts_s2 = frozenset(self.node2inst[n] for n in nodes_s2)

                ss_subgraph = self.connection_graph.subgraph(
                    frozenset.union(nodes_s1, nodes_s2)
                )
                # Process the pair only if the subgraph is connected
                if not nx.is_connected(ss_subgraph):
                    continue

                # Test this pair against table
                collision_directions = set.union(*[
                    self.collision_table.get((i_s1, i_s2), set())
                    for i_s1, i_s2 in product(insts_s1, insts_s2)
                ])
                join_possible = len(collision_directions) < 6

                if not join_possible:
                    # Potential minimal source of collision found; return
                    return frozenset([insts_s1, insts_s2])

        # No connected source found
        return None

def _linearize_join_tree(
    join_tree, exec_state, connection_graph, contacts, atomic_node_concs,
    node_unifications, obj2node_map, certified_pool, action_inds
):
    """
    Helper method factored out for finding the best linearization of the
    given join tree, in the sense that any joining steps involving nonobjs
    are postponed as much as possible. Flesh out the linearization into
    a sequence of atomic actions (i.e., pick-up, assemble, drop) and return.
    Also return mappings from nodes to scene objects.
    """
    pick_up_actions, drop_actions, assemble_actions = action_inds

    # Return values
    action_sequence = []
    node2obj_map = {}

    if len(join_tree) == 0:
        # If empty join tree was provided, it means all necessary joins
        # are already accomplished and there's no more action to take.
        # Drop whatever is held in the hands and call it a day.
        for side in [0, 1]:
            if exec_state["manipulator_states"][side][0] is not None:
                action_sequence.append((drop_actions[side], ()))

        return action_sequence, {}

    # Collect parts by type to allow flexible switching of object assignments
    part_type_pool = defaultdict(set, certified_pool)
    for obj, n in obj2node_map.items():
        part_type_pool[atomic_node_concs[n]].add(obj)

    # Stipulate any unified node-to-object mappings
    for sa, sa_graph in exec_state["connection_graphs"].items():
        if len(sa_graph) == 1: continue     # Dismiss singletons
        for obj in sa_graph.nodes:
            # Atomic parts already used in some subassembly shouldn't
            # be considered as candidate again
            for cnd_objs in part_type_pool.values():
                if obj in cnd_objs: cnd_objs.remove(obj)
            # Account for any applicable pairwise negative label info
            for objs, labels in exec_state["recognitions"].items():
                if not isinstance(objs, tuple): continue
                if obj not in objs: continue
                obj1, obj2 = objs
                label1, label2 = -labels[0], -labels[1]
                if obj == obj1 and obj2 in part_type_pool.get(label2, {}):
                    part_type_pool[label2].remove(obj2)
                if obj == obj2 and obj1 in part_type_pool.get(label1, {}):
                    part_type_pool[label1].remove(obj1)

            # Collect specified obj-to-node unifications
            if obj not in node_unifications: continue
            unified_node = node_unifications[obj]
            conc_n = atomic_node_concs[unified_node]
            node2obj_map[unified_node] = (obj, conc_n)

    # Offset to integer index for resultant subassemblies, obtained from
    # list of any existing subassemblies; ensure the newly assigned index
    # doesn't overlap with any of the previously assigned ones
    sa_ind_offset = max([
        int(re.findall(r"s(\d+)", sa)[0])
        for sa, sa_graph in exec_state["connection_graphs"].items()
        if len(sa_graph) > 1
    ] + [-1]) + 1           # [-1] to ensure max value is always obtained

    # Implement the select linearization as an actual action sequence, with
    # all the necessary information fleshed out (to be passed to the Unity
    # environment as parameters)
    hands = [exec_state["manipulator_states"][side][0] for side in [0, 1]]

    # If either hand is non-empty, see if any join can be made immediately
    # without dropping them
    next_join = None
    if any(held is not None for held in hands):
        available_joins = [
            n for n in join_tree if len(nx.ancestors(join_tree, n))==2
        ]
        available_joins = [
            (join_res,) + tuple(n for n, _ in join_tree.in_edges(join_res))
            for join_res in available_joins
        ]
        available_without_dropping = [
            (join_res, n1, n2)
            for join_res, n1, n2 in available_joins
            if set(hands) - {None} <= {n1, n2}
        ]
        if len(available_without_dropping) > 0:
            # Narrow down the options for the first join to the currently
            # possible joins
            next_join = available_without_dropping[0][0]
        else:
            # No such joins possible, drop currently handheld objects
            # and just randomly pick a join
            for side in [0, 1]:
                if hands[side] is not None:
                    hands[side] = None
                    action_sequence.append((drop_actions[side], ()))

    sa_ind = 0; nonobj_ind = 0
    sa_name_mapping = {}
    while len(list(nx.isolates(join_tree))) != len(join_tree):
        # List of all next-to-terminal nodes, which should have only two
        # ancestors, representing currently available joins
        available_joins = [
            n for n in join_tree if len(nx.ancestors(join_tree, n))==2
        ]
        available_joins_without_nonobjs = [
            n for n in available_joins
            if all(
                u not in atomic_node_concs or 
                    len(part_type_pool[atomic_node_concs[u]]) > 0
                for u, _ in join_tree.in_edges(n)
            )
        ]

        # `next_join` is not None only if the first join can be made
        # with object(s) currently handheld; otherwise, select one,
        # prioritizing those not involving any non-objs
        if next_join is None:
            # Select one available join, and continue building up as much as
            # possible
            if len(available_joins_without_nonobjs) > 0:
                next_join = available_joins_without_nonobjs[0]
            else:
                next_join = available_joins[0]

        (n1, a1), (n2, a2) = tuple(
            (u, join_tree.edges[(u, v)]["join_by"])
            for u, v in join_tree.in_edges(next_join)
        )
        res = next_join

        # Convert atomic nodes in hierarchy into randomly sampled objects
        # with matching part type, on first encounter
        for n in [a1, a2]:
            if n in atomic_node_concs and n not in node2obj_map:
                conc_n = atomic_node_concs[n]
                if len(part_type_pool[conc_n]) > 0:
                    obj = part_type_pool[conc_n].pop()
                    # Account for any applicable pairwise negative label info
                    for objs, labels in exec_state["recognitions"].items():
                        if not isinstance(objs, tuple): continue
                        if obj not in objs: continue
                        obj1, obj2 = objs
                        label1, label2 = -labels[0], -labels[1]
                        if obj == obj1 and obj2 in part_type_pool.get(label2, {}):
                            part_type_pool[label2].remove(obj2)
                        if obj == obj2 and obj1 in part_type_pool.get(label1, {}):
                            part_type_pool[label1].remove(obj1)
                else:
                    obj = f"n{nonobj_ind}"
                    nonobj_ind += 1
                node2obj_map[n] = (obj, conc_n)

        # Fetch matching (non-)object and unify with subassembly names if
        # necessary
        o1 = node2obj_map[a1][0]; o2 = node2obj_map[a2][0]
        s1 = o1 if n1 == a1 else n1
        s2 = o2 if n2 == a2 else n2

        if None not in hands:
            # Both hands are full, nothing to do here
            pass
        elif s1 in hands or s2 in hands:
            # Immediately using the subassembly built in the previous join
            empty_side = hands.index(None)
            occupied_side = list({0, 1} - {empty_side})[0]

            # Pick up the other involved subassembly with the empty hand
            pick_up_target = s2 if hands[occupied_side] == s1 else s1
            action_sequence.append((
                pick_up_actions[empty_side], (pick_up_target,)
            ))
        else:
            # Boths hand are (now) empty; pick up s1 and s2 on the left and
            # the right side each
            action_sequence.append((pick_up_actions[0], (s1,)))
            action_sequence.append((pick_up_actions[1], (s2,)))

        # Determine joining direction by comparing the degrees of the part
        # nodes involved: smaller to larger. If the degrees are equal, break
        # tie by comparing average neighbor degree.
        deg_handle = lambda n: (
            connection_graph.degree[n],
            nx.average_neighbor_degree(connection_graph)[n]
        )
        if deg_handle(a1) >= deg_handle(a2):
            # a1 is more 'central', a2 is more 'peripheral'; join a2 to a1
            if hands == [s1, s2]:
                direction = 0
            elif hands == [s2, s1]:
                direction = 1
            elif s1 in hands:
                direction = 1 if hands.index(None) == 0 else 0
            elif s2 in hands:
                direction = 0 if hands.index(None) == 0 else 1
            else:
                direction = 0
        else:
            # a2 is more 'central', a1 is more 'peripheral'; join a1 to a2
            if hands == [s1, s2]:
                direction = 1
            elif hands == [s2, s1]:
                direction = 0
            elif s1 in hands:
                direction = 0 if hands.index(None) == 0 else 1
            elif s2 in hands:
                direction = 1 if hands.index(None) == 0 else 0
            else:
                direction = 1

        # Join s1 and s2 at appropriate contact point
        sa_name = f"s{sa_ind+sa_ind_offset}"
        sa_name_mapping[res] = sa_name
        cp_a1, cp_a2 = contacts[(a1, a2)]
        cp_a1 = int(re.findall(r"p_(.*)$", cp_a1)[0])
        cp_a2 = int(re.findall(r"p_(.*)$", cp_a2)[0])
        if hands == [s1, s2]:
            # Already holding s1 and s2 on left and right respectively
            join_params = (s1, o1, cp_a1, s2, o2, cp_a2, sa_name)
        elif hands == [s2, s1]:
            # Already holding s2 and s1 on left and right respectively
            join_params = (s2, o2, cp_a2, s1, o1, cp_a1, sa_name)
        elif s1 in hands:
            if hands.index(None) == 0:
                # s1 held in right hand
                join_params = (s2, o2, cp_a2, s1, o1, cp_a1, sa_name)
            else:
                # s1 held in left hand
                join_params = (s1, o1, cp_a1, s2, o2, cp_a2, sa_name)
        elif s2 in hands:
            if hands.index(None) == 0:
                # s2 held in right hand
                join_params = (s1, o1, cp_a1, s2, o2, cp_a2, sa_name)
            else:
                # s2 held in left hand
                join_params = (s2, o2, cp_a2, s1, o1, cp_a1, sa_name)
        else:
            # Use join parameter order as-is
            join_params = (s1, o1, cp_a1, s2, o2, cp_a2, sa_name)
        action_sequence.append((assemble_actions[direction], join_params))

        # Update hands status
        hands = [None, None]
        hands[direction] = res

        sa_ind += 1         # Increment subassembly index

        # Current join accomplished, remove from tree
        join_tree.remove_node(n1); join_tree.remove_node(n2)
        if len(list(join_tree.out_edges(res))) == 0:
            # Drop and end current loop if finished for the current tree
            empty_side = hands.index(None)
            occupied_side = list({0, 1} - {empty_side})[0]
            action_sequence.append((drop_actions[occupied_side], ()))
            hands = [None, None]
            next_join = None
            continue

        # Can continue to climb up the tree if either:
        #   1) All remaining joins involve non-objs
        #   2) The parent join can be accomplished immediately, without
        #       involving non-objs
        next_join_candidate = list(join_tree.out_edges(res))[0][1]
        other = list({
            u for u, _ in join_tree.in_edges(next_join_candidate)
        } - {res})[0]
        can_continue = len(join_tree.in_edges(other)) == 0 and (
            len(available_joins_without_nonobjs) == 0 or (
                other not in atomic_node_concs or
                len(part_type_pool[atomic_node_concs[other]]) > 0
            )
        )
        if can_continue:
            # Fetch next join info
            next_join = next_join_candidate
        else:
            # Cannot continue without involving a non-obj, drop and move
            # onto a different starting join
            empty_side = hands.index(None)
            occupied_side = list({0, 1} - {empty_side})[0]
            action_sequence.append((drop_actions[occupied_side], ()))
            hands = [None, None]
            next_join = None
            continue

    # Rename subassembly names in action parameters according to the
    # collected mapping
    action_sequence = [
        (
            action_type,
            tuple(sa_name_mapping.get(prm, prm) for prm in action_params)
        )
        for action_type, action_params in action_sequence
    ]

    return action_sequence, node2obj_map

def _execute_pick_up(agent, action_name, action_params):
    """
    Pick up a designated object. Object may not have been present (as far as
    agent's visual perception is concerned) at the time of planning, in which
    case agent must inquire the user first if there exists an instance of the
    target concept in the scene. If yes, user provides a statement pointing
    to an instance, so the object can be referenced in this method.
    """
    target = action_params[0]
    target_info = agent.vision.scene.get(target, {})

    exec_state = agent.planner.execution_state      # Shortcut var

    if "env_handle" in target_info:
        # Existing atomic object, provide the environment side name
        env_handle = target_info["env_handle"]

        # Handle differently depending on whether next action is 3D structure
        # inspection
        next_action = agent.planner.agenda[0][1][0]
        next_action = agent.lt_mem.lexicon.d2s[("arel", next_action)][0][1]
        if "inspect" in next_action:
            # Picking up for 3D structure inspection, not making any type
            # assumption; do not introduce random pose perturbation, providing
            # the skeleton key "GT" as param
            estim_type = "GT"
        else:
            # Whether the agent has estimated the type of the target object
            # is important
            estim_type = exec_state["recognitions"][target]
            if agent.cfg.exp.player_type in ["bool", "demo"]:
                estim_type = agent.lt_mem.lexicon.codesheet[estim_type]
            else:
                assert agent.cfg.exp.player_type in ["label", "full"]
                estim_type = agent.lt_mem.lexicon.d2s[("pcls", estim_type)][0][1]
        agent_action = [
            (action_name, {
                "parameters": (f"str|{env_handle}", f"str|{estim_type}"),
                "pointing": {}
            }),
            ("generate", (f"# Action: {action_name}({target})", {}))
        ]
        manip_ind = 0 if action_name.endswith("left") else 1
        exec_state["manipulator_states"][manip_ind] = \
            (action_params[0], None, None)
        if "connection_graphs" in exec_state:
            singleton_gr = nx.DiGraph()
            singleton_gr.add_node(target)
            exec_state["connection_graphs"][target] = singleton_gr

    # This condition should be tested after the "env_handle" key test above
    # since singletons may also be recorded in "connection_graphs" field
    elif target in exec_state["connection_graphs"]:
        # Subassembly with name determined by agent, can provide the string
        # name handle as parameter to Unity environment
        agent_action = [
            (action_name, {
                "parameters": (f"str|{target}", "str|SA"), "pointing": {}
            }),
            ("generate", (f"# Action: {action_name}({target})", {}))
        ]
        manip_ind = 0 if action_name.endswith("left") else 1
        exec_state["manipulator_states"][manip_ind] = \
            (action_params[0], None, None)

    else:
        # Non-object, different strategies adopted for coping with the
        # grounding failure

        # Target concept, as recognized (estimation might be incorrect)
        target_conc = exec_state["recognitions"][target]

        if agent.cfg.exp.player_type in ["bool", "demo"]:
            # Language-less agents do not share vocabulary referring to parts
            # and thus need to report inability to proceed. Agent reports
            # it was not able to find a part needed for finishing the task,
            # but due to the lack of shared vocab, the following demo frag
            # from user is not guaranteed to address this particular grounding
            # failure.

            # NL surface form and corresponding logical form
            surface_form = "I cannot find a part I need on the table."
            gq = ("forall",); bvars = ("x0",)
            ante = [("n", "_needed_part", ["x0"])]
                # Not a proper vocabulary... Does not matter for now, though
            cons = []
            logical_form = (gq, bvars, ante, cons)

            # Referents & predicates info
            referents = {
                "e": { "mood": "." },       # Indicative
                "x0": { "entity": None, "rf_info": {} }
            }
            predicates = { "pa0": (("n", "_needed_part"), f"pcls_{target_conc}") }

            # Enter pause mode, waiting for teacher's partial demo up to
            # next valid join
            agent.execution_paused = True

            # Scrap currently queued plan, replan after teacher's demo
            goal_action, goal_target = exec_state["plan_goal"]
            goal_action = agent.lt_mem.lexicon.s2d[("va", goal_action)][0][1]
            exec_state["last_scrapped_plan"] = agent.planner.agenda
            agent.planner.agenda = deque([
                ("execute_command", (goal_action, (goal_target, True)))
            ])

        else:
            # Language-conversant agents can inquire the agent whether there
            # exists an instance of a part they need
            assert agent.cfg.exp.player_type in ["label", "full"]

            # NL symbol for the needed part
            target_sym = agent.lt_mem.lexicon.d2s[("pcls", target_conc)][0][1]

            # NL surface form and corresponding logical form
            surface_form = f"Is there a {target_sym}?"
            gq = ("Q",); bvars = ("x0",); ante = []
            cons = [("n", target_sym, ["x0"])]
            logical_form = (gq, bvars, ante, cons)

            # Referents & predicates info
            referents = {
                "e": { "mood": "?" },       # Interrogative
                "x0": { "entity": None, "rf_info": {} }
            }
            predicates = { "pc0": (("n", target_sym), f"pcls_{target_conc}") }

            # Scrap currently queued plan, replan after getting teacher's response
            goal_action, goal_target = exec_state["plan_goal"]
            goal_action = agent.lt_mem.lexicon.s2d[("va", goal_action)][0][1]
            exec_state["last_scrapped_plan"] = agent.planner.agenda
            agent.planner.agenda = deque([
                ("execute_command", (goal_action, (goal_target, True)))
            ])

        # Append to & flush generation buffer
        agent.lang.dialogue.to_generate.append(
            (logical_form, surface_form, referents, predicates, {})
        )
        agent_action = agent.lang.generate()

    # Wait before executing the rest of the plan until teacher reacts
    # (either with silent observation or interruption)
    agent.planner.agenda.appendleft(("execute_command", (None, None)))

    return agent_action

def _execute_drop(agent, action_name):
    """
    Drop whatever is held in the manipulator on the designated side. Not a lot
    of complications, no action parameters to consider; just signal the Unity
    environment.
    """
    agent_action = [(action_name, { "parameters": (), "pointing": {} })]

    # Wait before executing the rest of the plan until teacher reacts
    # (either with silent observation or interruption)
    agent.planner.agenda.appendleft(("execute_command", (None, None)))

    return agent_action

def _execute_assemble(agent, action_name, action_params):
    """
    Assemble two subassemblies held in each manipulator at the designated objects
    and contact points. Agent must provide desired transformation of the manipulator
    to move, such that the movement will achieve such join as action parameters to
    Unity---as opposed to the user who knows the ground-truth name handles of objects
    and contact points in the environment. Perform pose estimation of relevant
    objects from agent's visual input and compute pose difference. Necessary
    information must have been provided from Unity environment as 'effects' of
    previous actions.
    """
    obj_left, atomic_left, cp_left = action_params[:3]
    obj_right, atomic_right, cp_right = action_params[3:6]
    product_name = action_params[6]

    exec_state = agent.planner.execution_state      # Shortcut var

    mnp_state_left = exec_state["manipulator_states"][0]
    mnp_state_right = exec_state["manipulator_states"][1]
    held_left, pose_mnp_left, poses_atomics_left = mnp_state_left
    held_right, pose_mnp_right, poses_atomics_right = mnp_state_right
    graph_left = exec_state["connection_graphs"][held_left]
    graph_right = exec_state["connection_graphs"][held_right]

    # Sanity checks
    assert obj_left == held_left and obj_right == held_right
    assert atomic_left in graph_left and atomic_right in graph_right

    tmat_atomic_left = transformation_matrix(*poses_atomics_left[atomic_left])
    tmat_atomic_right = transformation_matrix(*poses_atomics_right[atomic_right])

    atomic_conc_left = exec_state["recognitions"][atomic_left]
    atomic_conc_right = exec_state["recognitions"][atomic_right]
    cp_list_left = agent.lt_mem.exemplars.object_3d[atomic_conc_left][3]
    cp_list_right = agent.lt_mem.exemplars.object_3d[atomic_conc_right][3]

    # Relation among manipulator, object & contact point transformations:
    # [Tr. of contact point in global coordinate]
    # = [Tr. of part instance in global coordinate] *
    #   [Tr. of contact point in part local coordinate]
    # = [Tr. of manipulator in global coordinate] *
    #   [Tr. of part instance in subassembly local coordinate] *
    #   [Tr. of contact point in part local coordinate]
    #
    # Based on these relations, we first obtain [Tr. of contact point in global
    # coordinate] and [Tr. of part instance in subassembly local coordinate],
    # then [Desired Tr. of manipulator in global coordinate] by equating
    # transforms of the two contact points of interest in global coordinate.
    tmat_cp_local_left = cp_list_left[cp_left][0][cp_right]
    tmat_cp_local_right = cp_list_right[cp_right][0][cp_left]
    tmat_cp_local_left = transformation_matrix(*tmat_cp_local_left)
    tmat_cp_local_right = transformation_matrix(*tmat_cp_local_right)
    if action_name.endswith("left"):
        # Target is on left, move right manipulator
        tmat_tgt_part = tmat_atomic_left
        tmat_src_part = tmat_atomic_right
        tmat_tgt_cp_local = tmat_cp_local_left
        tmat_src_cp_local = tmat_cp_local_right
        tmat_mnp_before = transformation_matrix(*pose_mnp_right)
    else:
        # Target is on right, move left manipulator
        assert action_name.endswith("right")
        tmat_tgt_part = tmat_atomic_right
        tmat_src_part = tmat_atomic_left
        tmat_tgt_cp_local = tmat_cp_local_right
        tmat_src_cp_local = tmat_cp_local_left
        tmat_mnp_before = transformation_matrix(*pose_mnp_left)
    tmat_tgt_cp_global = tmat_tgt_part @ tmat_tgt_cp_local
    tmat_part_offset = inv(tmat_mnp_before) @ tmat_src_part
    tmat_mnp_desired = tmat_tgt_cp_global @ \
        inv(tmat_part_offset @ tmat_src_cp_local)

    # Update assembly progress state by joining the connection graphs, annotating
    # the new edge with relative transformation between the two involved parts.
    # Edge direction: target to source.
    tmat_part_moved = tmat_mnp_desired @ tmat_part_offset
    tmat_rel = inv(tmat_tgt_part) @ tmat_part_moved
    graph_joined = nx.union(graph_left, graph_right)
    cps = { atomic_left: cp_left, atomic_right: cp_right }
    if action_name.endswith("left"):
        graph_joined.add_edge(atomic_left, atomic_right, rel_tr=tmat_rel, cps=cps)
    else:
        graph_joined.add_edge(atomic_right, atomic_left, rel_tr=tmat_rel, cps=cps)
    exec_state["connection_graphs"][product_name] = graph_joined
    # Remove used objects from connection graph listing
    del exec_state["connection_graphs"][obj_left]
    del exec_state["connection_graphs"][obj_right]
    # Update objects held in manipulators
    src_side = 1 if action_name.endswith("left") else 0
    tgt_side = 0 if action_name.endswith("left") else 1
    exec_state["manipulator_states"][src_side] = (None,) * 3
    exec_state["manipulator_states"][tgt_side] = \
        (product_name, None, None)

    # # Dev code for visually inspecting the result of the joining action;
    # # uncomment, copy, paste and run in debugger for sanity check
    # import open3d as o3d
    # parts_sorted = list(nx.topological_sort(graph_joined))
    # point_clouds = []
    # for part in parts_sorted:
    #     part_conc = exec_state["recognitions"][part]
    #     pcl = agent.lt_mem.exemplars.object_3d[part_conc][0]
    #     pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcl))
    #     pcl.paint_uniform_color(np.random.rand(3))

    #     connection_path = next(nx.shortest_simple_paths(
    #         graph_joined, parts_sorted[0], part
    #     ))
    #     tmat_abs = np.eye(4)
    #     while len(connection_path) > 1:
    #         # Need to aggregate relative transformations
    #         u = connection_path.pop(0)
    #         v = connection_path[0]
    #         tmat_rel = graph_joined.edges[(u, v)]["rel_tr"]
    #         tmat_abs = tmat_rel @ tmat_abs

    #     pcl.transform(tmat_abs)
    #     point_clouds.append(pcl)
    # o3d.visualization.draw_geometries(point_clouds)

    # Deconstruct the desired transformation matrix into quaternion and position,
    # serialized into action parameter string
    rot_mnp_desired = flip_quaternion_y(rmat2quat(tmat_mnp_desired[:3,:3]))
    pos_mnp_desired = flip_position_y(tmat_mnp_desired[:3,3].tolist())
    rot_serialized = "/".join(f"{a:.4f}" for a in rot_mnp_desired)
    pos_serialized = "/".join(f"{a:.4f}" for a in pos_mnp_desired)
    unity_action_params = (obj_left, atomic_left, obj_right, atomic_right)
    agent_action = [
        (action_name, {
            "parameters": (
                f"str|{product_name}",      # Also provide product name
                f"floats|{rot_serialized}", f"floats|{pos_serialized}"
            ),
            "pointing": {}
        }),
        ("generate", (
            f"# Action: {action_name}({','.join(unity_action_params)},{product_name})",
            {})
        )
    ]

    # Wait before executing the rest of the plan until teacher reacts (either with
    # silent observation or interruption)
    agent.planner.agenda.appendleft(("execute_command", (None, None)))

    return agent_action

def _execute_inspect(agent, action_name, action_params):
    """
    Inspect the target object, currently held in manipulator, at the designated
    viewpoint for 3D structure extraction. Not a lot of complications, just
    relay the target object's env_handle and the viewing angle index.
    """
    target = action_params[0]
    target = agent.vision.scene[target]["env_handle"]
    view_ind = action_params[1]

    agent_action = [
        (action_name, {
            "parameters": (f"str|{target}", f"int|{view_ind}"),
            "pointing": {}
        })
    ]

    # Wait before executing the rest of the plan until teacher reacts
    # (either with silent observation or interruption)
    agent.planner.agenda.appendleft(("execute_command", (None, None)))

    return agent_action

def report_planning_failure(agent):
    """
    Agent wasn't able to plan till the end product because some of the part
    instances (which user had used in partial demo) had unknown types and
    thus couldn't be deterministically unified with a assembly template node.
    Prepare a request to user for next valid partial demonstration and append
    to utterance generation queue.
    """
    exec_state = agent.planner.execution_state      # Shortcut var

    # Fetch current build target
    action_name, action_target = exec_state["plan_goal"]
    assert action_name == "build"       # Only task of interest in our scope
    action_conc = agent.lt_mem.lexicon.s2d[("va", action_name)][0][1]
    target_name = agent.lt_mem.lexicon.d2s[action_target][0][1]

    # NL surface forms and corresponding logical forms
    surface_form_0 = f"I couldn't plan further to build a {target_name}."
    surface_form_1 = f"# to-infinitive phrase ('to build a {target_name}')"
    gq_0 = gq_1 = (); bvars_0 = bvars_1 = (); ante_0 = ante_1 = []
    cons_0 = [
        ("sp", "unable", [("e", 0), "x0", ("e", 1)]),
        ("sp", "pronoun1", ["x0"]),
    ]
    # Note: Tuple argument in form of ("e", integer) refers to another clause
    # in generation buffer specified by the integer offset; e.g., ("e", 1) would
    # refer to the next clause after this one. Notation's a bit janky, but this
    # will do for now.
    cons_1 = [
        ("va", "build", [("e", 0), "x0", "x1"]),
        ("n", target_name, ["x1"])
    ]
    logical_form_0 = (gq_0, bvars_0, ante_0, cons_0)
    logical_form_1 = (gq_1, bvars_1, ante_1, cons_1)

    # Referents & predicates info
    referents_0 = { 
        "e": { "mood": "." },        # Indicative
        "x0": { "entity": "_self", "rf_info": {} }
    }
    referents_1 = {
        "e": { "mood": "~" },       # Infinitive
        "x0": { "entity": None, "rf_info": {} },
        "x1": { "entity": None, "rf_info": { "is_pred": True } }
    }
    predicates_0 = {
        "pc0": (("sp", "unable"), "sp_unable"),
        "pc1": (("sp", "pronoun1"), "sp_pronoun1")
    }
    predicates_1 = {
        "pc0": (("va", "build"), f"arel_{action_conc}"),
        "pc1": (("n", target_name), f"{action_target[0]}_{action_target[1]}")
    }

    # Append to & flush generation buffer
    record_0 = (logical_form_0, surface_form_0, referents_0, predicates_0, {})
    record_1 = (logical_form_1, surface_form_1, referents_1, predicates_1, {})
    agent.lang.dialogue.to_generate += [record_0, record_1]
    
    # Enter pause mode
    agent.execution_paused = True

    # Scrap currently queued plan, replan after teacher's demo
    goal_action, goal_target = exec_state["plan_goal"]
    goal_action = agent.lt_mem.lexicon.s2d[("va", goal_action)][0][1]
    exec_state["last_scrapped_plan"] = agent.planner.agenda
    agent.planner.agenda = deque([
        ("execute_command", (goal_action, (goal_target, True)))
    ])

def handle_action_effect(agent, effect, actor):
    """
    Update agent's internal representation of relevant environment states, in
    the face of obtaining feedback after performing a primitive environmental
    action. In our scope, we are concerned with effects of two types of actions:
    'pick_up', 'drop' and 'assemble' actions.
    """
    effect_lit = [lit for lit in effect if lit.name.startswith("arel")][0]
    action_type = int(effect_lit.name.split("_")[-1])
    action_name = agent.lt_mem.lexicon.d2s[("arel", action_type)][0][1]

    # Shortcut vars
    referents = agent.lang.dialogue.referents
    exec_state = agent.planner.execution_state      # Shortcut var

    if len(exec_state) == 0:
        # Probably called after a full demonstration is given, no task to be
        # executed for this episode
        return

    # Mapping from Unity-side name to scene object index
    env_handle_inv = {
        obj["env_handle"]: oi for oi, obj in agent.vision.scene.items()
    }

    # Utility method for parsing serialized float lists passed as action effect
    parse_floats = lambda ai: tuple(
        float(v)
        for v in referents["dis"][effect_lit.args[ai][0]]["name"].split("/")
    )

    if action_name.startswith("pick_up"):
        # Pick-up action effects: Unity gameObject name of the pick-up target,
        # pose of moved manipulator, poses of all atomic parts included in the
        # pick-up target subassembly after picking up
        manip_ind = 0 if action_name.endswith("left") else 1
        manip_state = exec_state["manipulator_states"][manip_ind]

        # Manipulator pose after picking up
        manip_pose = (
            flip_quaternion_y(xyzw2wxyz(parse_floats(3))),
            flip_position_y(parse_floats(4))
        )

        # Poses of individual parts in the object picked up
        part_poses = {}
        num_parts = int(referents["dis"][effect_lit.args[5][0]]["name"])
        for i in range(num_parts):
            part_uid = referents["dis"][effect_lit.args[6+3*i][0]]["name"]
            part_poses[env_handle_inv[part_uid]] = (
                flip_quaternion_y(xyzw2wxyz(parse_floats(6+3*i+1))),
                flip_position_y(parse_floats(6+3*i+2))
            )

        if actor == "Teacher":
            # Agent-initiated pick-up action has its original intention
            # (naturally) available to the actor, which has been already
            # reflected in manip_state[0]. However, if this pick-up action
            # was not initiated by agent but by user, need to incorporate
            # the intention behind the action; i.e., the pick-up target
            target = referents["dis"][effect_lit.args[2][0]]["name"]
            if target not in exec_state["connection_graphs"]:
                # Must be an atomic part, find corresponding scene object
                # with matching env_handle
                target = env_handle_inv[target]
                singleton_gr = nx.DiGraph()
                singleton_gr.add_node(target)
                exec_state["connection_graphs"][target] = singleton_gr
            # Update manipulator state
            exec_state["manipulator_states"][manip_ind] = \
                (target, manip_pose, part_poses)
            # Update action history
            exec_state["action_history"].append((action_type, (target,), actor))
        else:
            # Target info already reflected in manipulator state, just
            # update poses
            assert actor == "Student"
            exec_state["manipulator_states"][manip_ind] = \
                manip_state[:1] + (manip_pose, part_poses)

    if action_name.startswith("drop"):
        # Drop action effects: not much, just clear the corresponding manipulator
        manip_ind = 0 if action_name.endswith("left") else 1
        exec_state["manipulator_states"][manip_ind] = (None, None, None)

        if actor == "Teacher":
            if agent.cfg.exp.player_type in ["bool", "demo"]:
                # First drop after teacher's interruption (by "Stop.") signals
                # agent's previous action is undone. In case of drop actions,
                # need to explicitly check if it is an undoing action of the
                # previous pick-up, which is achieved by consulting the one-off
                # flag `agent.interrupted`.
                if agent.interrupted:
                    # Obtain original join intention
                    (atomic_left, atomic_right), _ = _last_intended_join(agent)
                    rcgn_left = exec_state["recognitions"][atomic_left]
                    rcgn_right = exec_state["recognitions"][atomic_right]
                    # The premise that `atomic_left` and `atomic_right` are each an
                    # instance of `rcgn_left` and `rcgn_right` resp. at the same
                    # time is incorrect. Record this negative labeling info so
                    # that the information can be properly taken into account
                    # from now on.
                    exec_state["recognitions"][((atomic_left, atomic_right))] = (
                        -rcgn_left, -rcgn_right
                    )
                        # CNF-like interpretation; i.e., either `atomic_left` is
                        # not an instance of `rcgn_left` or `atomic_right` one of
                        # `rcgn_right`. In other words, it should never hold that
                        # `atomic_left` is a `rcgn_left` and `atomic_right` is a
                        # `rcgn_right` at the same time.
                    # Uncommit relevant recognitions
                    del exec_state["recognitions"][atomic_left]
                    del exec_state["recognitions"][atomic_right]
                    agent.interrupted = False        # Disable flag

            # Update action history
            exec_state["action_history"].append((action_type, (), actor))

    if action_name.startswith("assemble"):
        src_side = 1 if action_name.endswith("left") else 0
        tgt_side = 0 if action_name.endswith("left") else 1

        if actor == "Student":
            # Student-initiated Assemble action effects received from Unity: pose of
            # target-side manipulator, closest contact point pairs after the joining
            # movement*, poses of all atomic parts included in the assembled product
            # (*: not used by learner agent)

            # Target-site manipulator pose after joining
            manip_pose = (
                flip_quaternion_y(xyzw2wxyz(parse_floats(2))),
                flip_position_y(parse_floats(3))
            )
            # Adjusting index offset...
            num_cp_pairs = int(referents["dis"][effect_lit.args[4][0]]["name"])
            ind_offset = 5 + 4 * num_cp_pairs

            # Poses of individual parts in the object picked up
            part_poses = {}
            num_parts = int(referents["dis"][effect_lit.args[ind_offset][0]]["name"])
            for i in range(num_parts):
                start_ind = ind_offset + 1 + 3 * i
                part_uid = referents["dis"][effect_lit.args[start_ind][0]]["name"]
                part_poses[env_handle_inv[part_uid]] = (
                    flip_quaternion_y(xyzw2wxyz(parse_floats(start_ind+1))),
                    flip_position_y(parse_floats(start_ind+2))
                )

            product_name = exec_state["manipulator_states"][tgt_side][0]
            exec_state["manipulator_states"][tgt_side] = \
                (product_name, manip_pose, part_poses)
        else:
            # Teacher-initiated Assemble action effects received from Unity: name of
            # the resultant product, poses of source-side manipulator before and after
            # movement*, pose of target-side manipulator, poses of all atomic parts
            # included in the assembled product (*: not used by learner agent)
            assert actor == "Teacher"

            # Name of the intended resultant product; could keep track of the
            # names for agent and user individually, but let's just share the
            # same namespace
            product_name = referents["dis"][effect_lit.args[2][0]]["name"]

            # Involved contact points (including atomic part info), left and right
            cp_left = referents["dis"][effect_lit.args[3][0]]["name"].split("/")
            cp_right = referents["dis"][effect_lit.args[4][0]]["name"].split("/")
            atomic_left = env_handle_inv[cp_left[0]]
            atomic_right = env_handle_inv[cp_right[0]]

            # Target-site manipulator pose after joining
            manip_pose = (
                flip_quaternion_y(xyzw2wxyz(parse_floats(9))),
                flip_position_y(parse_floats(10))
            )

            # Poses of individual parts in the object picked up
            part_poses = {}
            num_parts = int(referents["dis"][effect_lit.args[11][0]]["name"])
            for i in range(num_parts):
                start_ind = 12 + 3 * i
                part_uid = referents["dis"][effect_lit.args[start_ind][0]]["name"]
                part_poses[env_handle_inv[part_uid]] = (
                    flip_quaternion_y(xyzw2wxyz(parse_floats(start_ind+1))),
                    flip_position_y(parse_floats(start_ind+2))
                )

            # Update status of current collection of subassemblies
            joined_left = exec_state["manipulator_states"][0][0]
            joined_right = exec_state["manipulator_states"][1][0]
            graph_left = exec_state["connection_graphs"][joined_left]
            graph_right = exec_state["connection_graphs"][joined_right]
            product_gr = nx.union(graph_left, graph_right)
            if tgt_side == 0:
                join_edge = (atomic_left, atomic_right)     # R to L
            else:
                join_edge = (atomic_right, atomic_left)     # L to R
            product_gr.add_edge(join_edge[0], join_edge[1])

            del exec_state["connection_graphs"][joined_left]
            del exec_state["connection_graphs"][joined_right]
            exec_state["connection_graphs"][product_name] = product_gr

            # Update manipulator states
            exec_state["manipulator_states"][src_side] = (None,) * 3
            exec_state["manipulator_states"][tgt_side] = \
                (product_name, manip_pose, part_poses)
            # Update action history
            assemble_params = (
                joined_left, atomic_left, None,     # Exact cp type unknown
                joined_right, atomic_right, None,   # Ditto
                product_name
            )
            exec_state["action_history"].append(
                (action_type, assemble_params, actor)
            )

    if action_name.startswith("disassemble"):
        # Disassemble action always come from teacher, when agent's previous
        # assemble action was based on incorrect atomic part type premise
        # and thus needs to be undone
        assert actor == "Teacher"
            # Note that there is no need to check `agent.interrupted` because
            # any disassmble action is an undoing action of the previous
            # assemble action. Still, don't forget to disable the flag later.

        # Disassemble action effects: poses of both manipulators, poses of all
        # atomic parts included in each disassembly resultant objects
        manip_ind = 0 if action_name.endswith("left") else 1

        # (Former) names of joined subassemblies, obtained from action history,
        # used as names of the disassembly results again
        prev_left = exec_state["action_history"][-1][1][0]
        prev_right = exec_state["action_history"][-1][1][3]

        # (Former) name of originally held subassembly before the disassembly
        disassembled = exec_state["manipulator_states"][manip_ind][0]
        # Connection graph of the disassembled subassembly; remove from
        # exec_state
        sa_graph = exec_state["connection_graphs"].pop(disassembled)

        # Manipulator poses, left and right
        manip_pose_left = (
            flip_quaternion_y(xyzw2wxyz(parse_floats(2))),
            flip_position_y(parse_floats(3))
        )
        manip_pose_right = (
            flip_quaternion_y(xyzw2wxyz(parse_floats(4))),
            flip_position_y(parse_floats(5))
        )

        # Poses of individual parts in the objects on left and right after
        # the separation
        ind_offset = 6
        num_parts_left = int(
            referents["dis"][effect_lit.args[ind_offset][0]]["name"]
        )
        part_poses_left = {}
        for i in range(num_parts_left):
            start_ind = ind_offset + 1 + 3 * i
            part_uid = referents["dis"][effect_lit.args[start_ind][0]]["name"]
            part_poses_left[env_handle_inv[part_uid]] = (
                flip_quaternion_y(xyzw2wxyz(parse_floats(start_ind+1))),
                flip_position_y(parse_floats(start_ind+2))
            )
        ind_offset += 1 + 3 * num_parts_left
        num_parts_right = int(
            referents["dis"][effect_lit.args[ind_offset][0]]["name"]
        )
        part_poses_right = {}
        for i in range(num_parts_right):
            start_ind = ind_offset + 1 + 3 * i
            part_uid = referents["dis"][effect_lit.args[start_ind][0]]["name"]
            part_poses_right[env_handle_inv[part_uid]] = (
                flip_quaternion_y(xyzw2wxyz(parse_floats(start_ind+1))),
                flip_position_y(parse_floats(start_ind+2))
            )

        exec_state["manipulator_states"] = [
            (prev_left, manip_pose_left, part_poses_left),
            (prev_right, manip_pose_right, part_poses_right),
        ]

        # Restore connection graphs after disassembly, obtained as subgraphs
        graph_left = sa_graph.subgraph(part_poses_left).copy()
        graph_right = sa_graph.subgraph(part_poses_right).copy()
        exec_state["connection_graphs"][prev_left] = graph_left
        exec_state["connection_graphs"][prev_right] = graph_right

        # Update action history
        disassemble_params = (
            set(part_poses_left), set(part_poses_right), prev_left, prev_right
        )
        exec_state["action_history"].append(
            (action_type, disassemble_params, actor)
        )

        if agent.cfg.exp.player_type in ["bool", "demo"]:
            (atomic_left, atomic_right), _ = _last_intended_join(agent)
            rcgn_left = exec_state["recognitions"][atomic_left]
            rcgn_right = exec_state["recognitions"][atomic_right]
            # The premise that `atomic_left` and `atomic_right` are each an instance
            # of `rcgn_left` and `rcgn_right` resp. at the same time is incorrect.
            # Record this negative labeling info so that the information can be
            # properly taken into account from now on.
            exec_state["recognitions"][((atomic_left, atomic_right))] = (
                -rcgn_left, -rcgn_right
            )
                # CNF-like interpretation; i.e., either `atomic_left` is not an
                # instance of `rcgn_left` or `atomic_right` one of `rcgn_right`.
                # In other words, it should never hold that `atomic_left` is a
                # `rcgn_left` and `atomic_right` is a `rcgn_right` at the same time.
            # Uncommit relevant recognitions
            del exec_state["recognitions"][atomic_left]
            del exec_state["recognitions"][atomic_right]
            agent.interrupted = False        # Disable flag

    if action_name.startswith("inspect"):
        # Inspect action effects: pose of the inspection target object in the
        # camera coordinate, keep collecting the pose along with the latest
        # visual input (raw scene image and mask) until the end of the inspection
        # action series (signaled by view angle index of 24). Afterwards, extract
        # the 3D structure from the aggregated data.

        # Making temporary data storage field if not present
        if "3d_inspection_data" not in exec_state:
            exec_state["3d_inspection_data"] = { "img": {}, "msk": {}, "pose": {} }

        # Aggregate data
        view_ind = exec_state["action_history"][-1][1][1]
        scene_img = agent.vision.latest_inputs[-1]
        part_pose = (
            flip_quaternion_y(xyzw2wxyz(parse_floats(2))),
            flip_position_y(parse_floats(3))
        )
        raw_mask = np.array(rle_decode(
            [int(v) for v in referents["dis"][effect_lit.args[4][0]]["name"].split("/")]
        ))
        raw_mask = raw_mask.reshape(scene_img.height, scene_img.width).astype(bool)
        if view_ind < 24:
            exec_state["3d_inspection_data"]["img"][view_ind] = scene_img
            exec_state["3d_inspection_data"]["pose"][view_ind] = part_pose
        if view_ind > 0:
            # Object mask passed as inspection action effect falls behind
            # by one step
            exec_state["3d_inspection_data"]["msk"][view_ind-1] = raw_mask

        if view_ind == 24:
            # All necessary data collected, extract the 3D structure
            reconstruction = agent.vision.reconstruct_3d_structure(
                exec_state["3d_inspection_data"]["img"],
                exec_state["3d_inspection_data"]["msk"],
                exec_state["3d_inspection_data"]["pose"],
                CON_GRAPH, STORE_VP_INDS
            )
            point_cloud, views, descriptors = reconstruction

            # Fetch inspection target concept
            conc = exec_state["action_history"][-1][1][-1]
            conc_str = f"{conc[0]}_{conc[1]}"

            # Simplifying assumption: all subtypes of a part supertype share
            # the same coordinate space (origin and axes) and contact point
            # poses defined within. Without the assumption, we'd have to
            # request join demonstrations for every single pair of (part
            # subtype, contact point).
            supertype_conc = [
                entry[0][1][0].name
                for ei in sorted(agent.lt_mem.kb.entries_by_pred[conc_str])
                if (entry := agent.lt_mem.kb.entries[ei])[3] == "taxonomy" and \
                    entry[0][0][0].name == conc_str
            ][-1]
                # Premise: under current design, the last relevant taxonomy
                # holds the supertype to follow for copying over the contacts
            all_subtype_concs = {
                entry[0][0][0].name
                for ei in agent.lt_mem.kb.entries_by_pred[supertype_conc]
                if (entry := agent.lt_mem.kb.entries[ei])[3] == "taxonomy" and \
                    entry[0][1][0].name == supertype_conc
            } - {conc_str}
            contact_points = next(
                agent.lt_mem.exemplars.object_3d[st_conc][3]
                for st_conc_str in all_subtype_concs
                if (st_conc := int(st_conc_str.split("_")[1])) in \
                    agent.lt_mem.exemplars.object_3d
            )

            # Store the extracted structure and the imported contact points in XB
            agent.lt_mem.exemplars.add_exs_3d(
                conc[1], np.asarray(point_cloud.points), views, descriptors,
                contact_points
            )

            # Also store some 2D exemplars from select viewpoints (as done
            # in learn.analyze_demonstration method)
            vis_model = agent.vision.model; vis_model.eval()
            updated_concs = set()
            with torch.no_grad():
                for view_ind in STORE_VP_INDS[:4]:
                    image = exec_state["3d_inspection_data"]["img"][view_ind]
                    mask = exec_state["3d_inspection_data"]["msk"][view_ind]
                    vis_prompt = visual_prompt_by_mask(
                        image, blur_and_grayscale(image), [mask]
                    )
                    vp_processed = vis_model.dino_processor(images=vis_prompt, return_tensors="pt")
                    vp_pixel_values = vp_processed.pixel_values.to(vis_model.dino.device)
                    vp_dino_out = vis_model.dino(pixel_values=vp_pixel_values, return_dict=True)
                    f_vec = vp_dino_out.pooler_output.cpu().numpy()[0]
                    
                    _, concs = agent.lt_mem.exemplars.add_exs_2d(
                        scene_img=image,
                        exemplars=[{ "scene_id": None, "mask": mask, "f_vec": f_vec }],
                        pointers={ ("pcls", conc[1], "pos"): {(True, 0)} }
                    )
                    updated_concs |= concs
            agent.lt_mem.exemplars.update_bin_clfs_2d(updated_concs)

            # Remove the temporary data storage field
            del exec_state["3d_inspection_data"]
