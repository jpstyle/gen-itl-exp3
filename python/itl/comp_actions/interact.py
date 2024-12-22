"""
Implements composite agent interactions, environmental or dialogue, that need
coordination of different component modules
"""
import os
import re
import logging
from itertools import combinations, product
from collections import defaultdict, deque

import yaml
import numpy as np
import networkx as nx
from numpy.linalg import inv
from clingo import Control, SymbolType, Number, Function

from ..vision.utils import (
    xyzw2wxyz, flip_position_y, flip_quaternion_y, rmat2quat, transformation_matrix
)
from ..lpmln import Literal
from ..lpmln.utils import wrap_args


logger = logging.getLogger(__name__)

# As it is simply impossible to enumerate 'every possible continuous value',
# we consider a discretized likelihood threshold as counterfactual alternative.
# The value represents an alternative visual observation that results in evidence
# that is 'reasonably' stronger that the corresponding literal actually being true.
# (cf. ..symbolic_reasoning.attribute)
HIGH = 0.85

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

    ti, ci = utt_pointer
    (_, question), _ = dialogue_record[ti][1][ci]

    if question is None:
        # Question cannot be answered for some reason
        return
    else:
        # Schedule to answer the question
        agent.planner.agenda.appendleft("answer_Q", utt_pointer)
        return

def prepare_answer_Q(agent, utt_pointer):
    """
    Prepare an answer to a question that has been deemed answerable, by first
    computing raw ingredients from which answer candidates can be composed,
    picking out an answer among the candidates, then translating the answer
    into natural language form to be uttered
    """
    # The question is about to be answered
    agent.lang.dialogue.unanswered_Qs.remove(utt_pointer)

    ti, ci = utt_pointer
    dialogue_record = agent.lang.dialogue.export_resolved_record()

    if agent.lang.dialogue.clause_info[f"t{ti}c{ci}"]["domain_describing"]:
        _answer_domain_Q(agent, utt_pointer, dialogue_record)
    else:
        _answer_nondomain_Q(agent, utt_pointer, dialogue_record)

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
    parse, raw, _ = dialogue_record[ti][1][ci]
    cons = parse[3]

    command_executable = True

    action_lit = [lit for lit in cons if lit.name.startswith("arel_")][0]
    action_type = int(action_lit.name.replace("arel_", ""))
    action_params = {}
    for arg in action_lit.args[2:]:
        arg_describing_lits = [lit for lit in cons if lit.args==[arg]]

        if agent.lang.dialogue.referents["dis"][arg[0]].get("is_pred"):
            # Referent denotes a predicate category
            action_params[arg[0]] = [lit.name for lit in arg_describing_lits]
            if any(pred == "sp_neologism" for pred in action_params[arg[0]]):
                # Neologism included in action parameter config, cannot execute
                # (neologism will be separately handled by `report_neologism` method)
                command_executable = False
                break
        else:
            # Referent denotes an environment entity
            action_params[arg[0]] = agent.lang.dialogue.referents["env"][-1][arg[0]]

    if command_executable:
        # Schedule to generate plan & execute towards fulfilling the command
        agent.planner.agenda.appendleft(("execute_command", (action_type, action_params)))
        agent.lang.dialogue.unexecuted_commands.remove(utt_pointer)
        return

    else:
        # Command cannot be executed for some reason; report inability and dismiss
        # the command
        ri_command = f"t{ti}c{ci}"                      # Denotes original request

        # NL surface form and corresponding logical form
        surface_form = f"I am unable to {raw[0].lower()}{raw[1:]}"
        gq = None; bvars = set(); ante = []
        cons = [
            ("sp", "unable", ["x0", ri_command]),
            ("sp", "pronoun1", ["x0"])
        ]
        logical_form = (gq, bvars, ante, cons)

        mood = "."      # Indicative

        # Referents & predicates info
        referents = { "x0": { "entity": "_self", "rf_info": {} } }
        predicates = { "pc0": (("sp", "unable"), "sp_unable") }

        agent.lang.dialogue.unexecuted_commands.remove(utt_pointer)
        agent.lang.dialogue.to_generate.append(
            (logical_form, surface_form, mood, referents, predicates, {})
        )

        return

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

    # Check if (the remainder) of the plan being executed need to be examined,
    # to see whether if it is still valid after agent's knowledg update, if any.
    replanning_needed = False
    xb_updated, kb_updated = agent.planner.execution_state.get(
        "knowledge_updated", (False, False)
    )

    if kb_updated:
        # Always plan again upon KB update
        replanning_needed |= True

    if xb_updated and not replanning_needed:
        # One scenario in which we can skip replanning from scratch is when
        # the teacher's labeling feedback, which resulted in XB update, did
        # not involve labeling of an instance that is already referenced in
        # the plan (i.e., when the corrective feedback doesn't refute the
        # premise based on agent's previous object recognition output). Note
        # that this only happens in case of grounding failure where agent
        # believes some part instance needed does not exist.
        resolved_record = agent.lang.dialogue.export_resolved_record()
        labeling_feedback = [
            lit
            for spk, turn_clauses in resolved_record
            for ((_, _, _, cons), _, mood) in turn_clauses
            for lit in cons
            if spk == "Teacher" and mood == "." and lit.name.startswith("pcls")
        ]
        labeling_feedback = {
            lit.args[0][0]: int(lit.name.strip("pcls_"))
            for lit in labeling_feedback
        }
        plan_remainder_args = set(sum([
            action_params or ()
            for agenda_type, (_, action_params) in agent.planner.agenda
            if agenda_type == "execute_command"
        ], ()))

        if len(set(labeling_feedback) & plan_remainder_args) > 0:
            # Need to plan again with updated XB
            replanning_needed |= True
        else:
            # First append the popped action spec back to the left of the agenda
            agent.planner.agenda.appendleft(("execute_command", action_spec))

            # Fetch last (postponed) action spec for re-execution
            action_type, action_params = exec_state["action_history"][-1]

            # Unify non-object handle with a labeled object by selecting one
            # with matching type and adding the non-object handle to vis.scene
            # along with env_handle field, so that _execute_pick_up can be
            # properly executed
            nonobj_handle = action_params[0]
            unifying_type = exec_state["recognitions"][nonobj_handle]
            matched_obj = [
                obj for obj, conc in labeling_feedback.items()
                if conc == unifying_type
            ][0]
            agent.vision.scene[nonobj_handle] = {
                "unified_obj": matched_obj,
                "env_handle": agent.vision.scene[matched_obj]["env_handle"]
            }

    exec_state["knowledge_updated"] = (False, False)    # Clear flags

    if replanning_needed:
        # Plan again for the remainder of the assembly based on current execution
        # state, belief and knowledge, then return without return value

        # Update execution state to reflect recognitions hitherto committed
        # and those certified by user feedback, by packaging them as value
        # of "recognitions" field
        exec_state.update({
            "recognitions": labeling_feedback | {
                n: exec_state["recognitions"][n]
                for gr in exec_state["connection_graphs"].values()
                for n in gr
                if "pred_cls" in agent.vision.scene[n]      # Ignore non-objects
            }
        })

        action_sequence, recognitions = _plan_assembly(agent, exec_state["plan_goal"][1])

        # Record how the agent decided to recognize each object
        agent.planner.execution_state["recognitions"] = recognitions

        # Enqueue appropriate agenda items and finish
        agent.planner.agenda = deque(
            ("execute_command", action_step) for action_step in action_sequence
        )       # Whatever steps remaining, replace
        agent.planner.agenda.append(("utter_simple", ("Done.", ".")))

    else:
        # Currently considered commands: some long-term commands that requiring
        # long-horizon planning (e.g., 'build'), and some primitive actions that
        # are to be executed---that is, signaled to Unity environment---immediately

        # Record the spec of action being executed
        if "action_history" in exec_state:
           exec_state["action_history"].append((action_type, action_params))

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
    # Target structure
    build_target = list(action_params.values())[0][0].split("_")
    build_target = (build_target[0], int(build_target[1]))

    # Initialize plan execution state tracking dict
    agent.planner.execution_state = {
        "plan_goal": ("build", build_target),
        "manipulator_states": [(None, None, None), (None, None, None)],
            # Resp. left & right held object, manipulator pose, component masks
        "part_identifiers": {},
            # Stores Unity-side identifiers mapped to individual atomic parts
        "connection_graphs": {},
        "recognitions": {},
        "knowledge_updated": (False, False),
        "action_history": []
    }

    # Plan towards building valid target structure
    action_sequence, recognitions = _plan_assembly(agent, build_target)

    # Record how the agent decided to recognize each object
    agent.planner.execution_state["recognitions"] = recognitions

    # Enqueue appropriate agenda items and finish
    agent.planner.agenda = deque(
        ("execute_command", action_step) for action_step in action_sequence
    )
    agent.planner.agenda.append(("utter_simple", ("Done.", ".")))

def _plan_assembly(agent, build_target):
    """
    Helper method factored out for planning towards the designated target
    structure. May be used for replanning after user's belief/knowledge status
    changed, picking up from current assembly progress state recorded.
    """
    exec_state = agent.planner.execution_state      # Shortcut var

    # Convert current visual scene (from ensemble prediction) into ASP fact
    # literals (note difference vs. agent.lt_mem.kb.visual_evidence_from_scene
    # method, which is for probabilistic reasoning by LP^MLN)
    threshold = 0.25; dsc_bin = 20
    observations = set()
    likelihood_values = np.stack([
        obj["pred_cls"] for obj in agent.vision.scene.values()
        if "pred_cls" in obj
    ])
    min_val = max(likelihood_values.min(), threshold)
    max_val = likelihood_values.max()
    val_range = max_val - min_val

    for oi, obj in agent.vision.scene.items():
        if "pred_cls" not in obj:
            # An artifact of a non-object added due to grounding failure, no
            # need to (redundantly) evidence likelihood literals
            continue

        if oi in exec_state["recognitions"]:
            # Object already committed with confidence, or labeling feedback
            # directly provided by user; max confidence
            ci = exec_state["recognitions"][oi]
            obs_lit = Literal("is_likely", wrap_args(oi, ci, dsc_bin + 1))
            observations.add(obs_lit)
        else:
            # List all predictions with values above threshold
            for ci, val in enumerate(obj["pred_cls"]):
                if val < threshold: continue

                # Normalize & discretize within [0,dsc_bin]; the more bins we
                # use for discrete approximation, the more time it takes to
                # solve the program
                nrm_val = (val-min_val) / val_range
                dsc_val = int(nrm_val * dsc_bin)

                obs_lit = Literal(
                    "is_likely", wrap_args(oi, ci, dsc_val)
                )
                observations.add(obs_lit)

    # Compile assembly structure knowledge into ASP fact literals
    structures = agent.lt_mem.kb.assembly_structures
    assembly_pieces = set()
    for (_, sa_conc), templates in structures.items():
        # For each valid template option
        template_desc = "sole" if len(templates)==1 else "possible"
        for ti, template in enumerate(templates):
            # Register this specific template option
            template_lit = Literal(
                f"sa_{template_desc}_template", wrap_args(sa_conc, ti)
            )
            assembly_pieces.add(template_lit)

            # Register part/subassembly options for each node
            for n, data in template.nodes(data=True):
                match data["node_type"]:
                    case "atomic_part":
                        options = data["parts"]
                        for opt_conc in options:
                            part_lit = Literal("atomic", wrap_args(opt_conc))
                            assembly_pieces.add(part_lit)
                    case "subassembly":
                        options = data["subassemblies"]
                        for opt_conc in options:
                            sa_lit = Literal("subassembly", wrap_args(opt_conc))
                            assembly_pieces.add(sa_lit)
                choice_desc = "sole" if len(options)==1 else "possible"
                for opt_conc in options:
                    opt_lit = Literal(
                        f"node_{choice_desc}_choice",
                        wrap_args(sa_conc, ti, n, opt_conc)
                    )
                    assembly_pieces.add(opt_lit)

            # Register assembly contact signatures, iterating over edges
            sgn_str = lambda sg: str(sg[0]) if len(sg) == 1 \
                else f"c({sg[0]},{sgn_str(sg[1:])})"        # Serialization
            for u, v, data in template.edges(data=True):
                u, v = min(u, v), max(u, v)     # Ensure sorted order
                cp_options_u = [
                    (sgn_str(signature), f"p_{cp[0]}_{cp[1]}")
                    for signature, cp in data["contact"][u].items()
                        # `signature` represents a sequence of subassembly
                        # templates with terminal atomic part suffix
                ]
                cp_options_v = [
                    (sgn_str(signature), f"p_{cp[0]}_{cp[1]}")
                    for signature, cp in data["contact"][v].items()
                ]

                for cp_opt_u, cp_opt_v in product(cp_options_u, cp_options_v):
                    sgn_u, cp_u = cp_opt_u; sgn_v, cp_v = cp_opt_v
                    conn_sgn_lit = Literal(
                        "connection_signature",
                        wrap_args(sa_conc, ti, u, v, sgn_u, sgn_v, cp_u, cp_v)
                    )
                    assembly_pieces.add(conn_sgn_lit)

    # Additional constraints introduced by current assembly progress. Selected
    # goal structure must be compliant with existing subassemblies assembled;
    # namely, existing structures must be unifiable with fragments of final
    # goal structure
    ext_constraints = set()
    for sa_name, sa_graph in exec_state["connection_graphs"].items():
        # Adding each existing part along with committed recognition
        for ext_node in sa_graph.nodes:
            if ext_node in exec_state["recognitions"]:
                # Fetch the part concept directly from scene
                ext_conc = exec_state["recognitions"][ext_node]
            else:
                # Redirect non-object names
                redirected_obj = agent.vision.scene[ext_node]["unified_obj"]
                ext_conc = exec_state["recognitions"][redirected_obj]
            ext_node_lit = Literal(
                "ext_node",
                wrap_args(f"{sa_name}_{ext_node}", ext_conc)
            )
            ext_constraints.add(ext_node_lit)
        # Adding each connection between existing parts
        for ext_node1, ext_node2 in sa_graph.edges:
            ext_edge_lit = Literal(
                "ext_edge",
                wrap_args(f"{sa_name}_{ext_node1}", f"{sa_name}_{ext_node2}")
            )
            ext_constraints.add(ext_edge_lit)

    # Encode assembly target info into ASP fact literal
    target_lit = Literal("build_target", wrap_args(build_target[1]))

    # Load ASP encoding for selecting & committing to a goal structure
    lp_path = os.path.join(agent.cfg.paths.assets_dir, "planning_encoding")
    commit_ctl = Control(["--warn=none"])
    commit_ctl.configuration.solve.models = 0
    commit_ctl.configuration.solve.opt_mode = "opt"
    commit_ctl.load(os.path.join(lp_path, "goal_selection.lp"))

    # Add and ground all the fact literals obtained above
    all_lits = sorted(
        observations | assembly_pieces | ext_constraints | {target_lit},
        key=lambda x: x.name
    )
    facts_prg = "".join(str(lit) + ".\n" for lit in all_lits)
    commit_ctl.add("facts", [], facts_prg)

    # Optimize with clingo
    commit_ctl.ground([("base", []), ("facts", [])])
    with commit_ctl.solve(yield_=True) as solve_gen:
        models = [m.symbols(atoms=True) for m in solve_gen]
        if len(models) > 0:
            optimal_model = models[-1]

    # Compile the optimal solution into appropriate data structures
    assembly_hierarchy = nx.DiGraph()
    part_candidates = defaultdict(set)
    connect_edges = {}
    rename_node = lambda n: str(n.number) if n.type == SymbolType.Number \
        else f"{rename_node(n.arguments[0])}_{n.arguments[1]}"
    for atm in optimal_model:
        match atm.name:
            case "atomic_node" | "sa_node":
                # Each 'node' literal contains identifiers of its own and its direct
                # parent's, add an edge linking the two to the committed structure 
                # graph
                node_rn = rename_node(atm.arguments[0])
                node_type = atm.name.split("_")[0]
                assembly_hierarchy.add_node(node_rn, node_type=node_type)

                if atm.arguments[0].type == SymbolType.Function:
                    # Add edge between the node and its direct parent
                    assert atm.arguments[0].name == "n"
                    assembly_hierarchy.add_edge(
                        rename_node(atm.arguments[0].arguments[0]),
                        node_rn
                    )

            case "commit_choice":
                # Decision as to fill which node with an instance of which atomic part
                # type
                node_rn = rename_node(atm.arguments[0])
                choice_conc = choice_conc=atm.arguments[1].number
                assembly_hierarchy.add_node(node_rn, choice_conc=choice_conc)
            
            case "use_as":
                # Decision as to use which recognized object as instance of which
                # atomic part type
                obj_name = atm.arguments[0].name
                part_conc = atm.arguments[1].number
                part_candidates[part_conc].add(obj_name)

            case "to_connect":
                if len(atm.arguments) != 4:
                    # Skip projected literals
                    continue

                # Derived assembly connection to make between nodes
                node_u_rn = rename_node(atm.arguments[0])
                node_v_rn = rename_node(atm.arguments[1])
                cp_u = atm.arguments[2].name
                cp_v = atm.arguments[3].name
                connect_edges[(node_u_rn, node_v_rn)] = (cp_u, cp_v)
                connect_edges[(node_v_rn, node_u_rn)] = (cp_v, cp_u)

    # Top node
    assembly_hierarchy.add_node("0", node_type="sa")
    # Note: This assembly hierarchy structure object also serves to provide
    # explanation which part is required for building which subassembly
    # (that is direct parent in the hierarchy)

    # Depth-first traversal of the committed assembly hierarchy for randomly
    # allocating objects from candidate pools
    def recursive_allocate(u):
        # Atomic part of subassembly concept for the node
        conc = assembly_hierarchy.nodes[u]["choice_conc"]

        match assembly_hierarchy.nodes[u]["node_type"]:
            case "atomic":
                # Sample from the pool of candidate objects for the part type
                # and assign to the node
                if len(part_candidates[conc]) > 0:
                    # Candidate object available, pop from pool
                    sampled_obj = part_candidates[conc].pop()
                    assembly_hierarchy.nodes[u]["obj_used"] = sampled_obj
                else:
                    # Ran out of candidate objects, mark as unavailable
                    assembly_hierarchy.nodes[u]["obj_used"] = None
            case "sa":
                # Recursive iteration over children nodes
                for _, v in assembly_hierarchy.out_edges(u):
                    recursive_allocate(v)

    recursive_allocate("0")        # Start from the top node ("0")

    # Assign unique indentifiers for 'non-objects', which represent failure to
    # perceive objects that ought to exist in order to fulfill the committed
    # goal structure. They may be present on the tabletop yet not spotted by
    # the agent's vision module, or they may not exist after all.
    nonobj_ids = {
        n: f"n{i}"
        for i, (n, obj) in enumerate(assembly_hierarchy.nodes(data="obj_used"))
        if assembly_hierarchy.nodes[n]["node_type"] == "atomic" and obj is None
    }
    nonobj_ids_inv = { v: k for k, v in nonobj_ids.items() }

    # Turns out if a structure consists of too many atomic parts (say, more
    # than 6 or so), ASP planner performance is significantly affected. We
    # handle this by breaking the planning problem down to multiple smaller
    # ones... In principle, this may risk planning failure when physical
    # collision check is involved, depending on how the target structure is
    # partitioned. Note that agents that are aware of semantically valid
    # substructures are entirely free from such concerns.
    connection_graph = nx.Graph()
    for u, v in connect_edges:
        obj_u = assembly_hierarchy.nodes[u]["obj_used"] or nonobj_ids[u]
        obj_v = assembly_hierarchy.nodes[v]["obj_used"] or nonobj_ids[v]
        connection_graph.add_edge(obj_u, obj_v)

    # Ground-truth oracle for querying whether two atomic parts (or their
    # subassemblies) can be disassembled from their final positions without
    # collision along six directions (x+/x-/y+/y-/z+/z-). Coarse abstraction
    # of low-level motion planner, which need to be integrated if such
    # oracle is not available in real use cases.
    knowledge_path = os.path.join(agent.cfg.paths.assets_dir, "domain_knowledge")
    with open(f"{knowledge_path}/collision_table.yaml") as yml_f:
        collision_table = yaml.safe_load(yml_f)
    # Extracted mapping from string part names to agent's internal concept
    # denotations
    part_names = {
        d[1]: s[0][1] for d, s in agent.lt_mem.lexicon.d2s.items()
        if d[0]=="pcls"
    }
    # Ground-truth oracle for inspecting what each contact point learned
    # by agent was supposed to stand for in the knowledge encoded in Unity
    # assets
    cp_oracle = {
        (cp_conc, cp_i): gt_handle
        for _, _, _, cps in agent.lt_mem.exemplars.object_3d.values()
        for cp_conc, cp_insts in cps.items()
        for cp_i, (_, gt_handle) in enumerate(cp_insts)
    }

    # Compress the connection graph part by part, randomly selecting chunks
    # of parts or subassemblies to form smaller planning problems
    compression_graph = connection_graph.copy()
    min_chunk_size = 5; chunks_assembled = {}
    invalid_chunk_sequences = []; current_chunk_sequence = []
    sa_ind = 0
    join_sequence = []
    # Record every 'assembly scope implication' such that if a set of certain
    # atomic objects are to be included in a chunk, some other object must
    # be included in the set as well, so as to avoid deadends due to violations
    # of 'precedence constraints'.
    scope_implications = { n: set() for n in connection_graph }

    valid_joins_cached = set(); invalid_joins_cached = set()
    replan_attempts = 0; query_count = 0
    # Continue until the target structure is fully compressed and planned for    
    while len(compression_graph) > 1:
        # Subsample designated number of nodes by BFS from each leaf node.
        # Select one with the least non-objects, break tie randomly.
        chunk_candidates = []
        for n, dg in compression_graph.degree:
            if dg > 1: continue
            bfs_gen = nx.bfs_predecessors(compression_graph, n)

            chunk = {n}
            while len(chunk) < min_chunk_size:
                try:
                    next_node = next(bfs_gen)[0]
                    chunk.add(next_node)
                    for ante, cons in scope_implications.get(next_node, set()):
                        objs_flat = {
                            n for n in chunk if n not in chunks_assembled
                        }
                        objs_hidden = {
                            o for n in chunk if n in chunks_assembled
                            for o in chunks_assembled[n]
                        }
                        chunk_unrolled = objs_flat | objs_hidden
                        if ante <= chunk_unrolled:
                            for s, objs in chunks_assembled.items():
                                if cons in objs:
                                    chunk.add(s)
                                    break
                            else:
                                chunk.add(cons)
                except StopIteration:
                    break

            nonobj_count = len([u for u in chunk if u in nonobj_ids_inv])
            chunk_candidates.append((chunk, nonobj_count))

        chunk_candidates = sorted(chunk_candidates, key=lambda x: x[1])
        for chunk, _ in chunk_candidates:
            # Select chunk with smallest counts of non-objects possible,
            # while ensuring that the selection doesn't result in an
            # invalid chunk sequence (remembered so far)
            potential_sequence = current_chunk_sequence + [chunk]
            if potential_sequence not in invalid_chunk_sequences:
                chunk_selected = chunk
                break
        else:
            # All possible options would lead to invalid sequence. Remember
            # the current sequence as invalid and start again.
            compression_graph = connection_graph.copy()
            chunks_assembled = {}
            invalid_chunk_sequences.append(current_chunk_sequence)
            current_chunk_sequence = []
            sa_ind = 0
            join_sequence = []
            replan_attempts += 1
            query_count += collision_checker.query_count
            valid_joins_cached |= collision_checker.valid_joins
            invalid_joins_cached |= collision_checker.invalid_joins
            continue

        chunk_subgraph = compression_graph.subgraph(chunk_selected)

        # Load ASP encoding for planning sequence of atomic actions
        collision_checker = _PhysicalAssemblyPropagator(
            (connection_graph, connect_edges),
            (dict(assembly_hierarchy.nodes(data=True)), nonobj_ids),
            (collision_table, part_names, cp_oracle),
            (valid_joins_cached, invalid_joins_cached)
        )
        plan_ctl = Control(["--warn=none"])
        plan_ctl.register_propagator(collision_checker)
        plan_ctl.configuration.solve.models = 0
        plan_ctl.configuration.solve.opt_mode = "opt"
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
            if step > len(chunk):
                # If reached here, ASP problem has no valid solution, probably
                # due to faulty subgraph sampling; update the set of assembly
                # implications as appropriate

                # Break from this while loop to restart planning (after due
                # updates of scope implications)
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

        # Update list of known scope implications from invalid joins witnessed
        for join_pair in collision_checker.invalid_joins:
            objs1, objs2 = join_pair

            # Process each direction only if the 'consequent' side
            # of the implication has one object in the set
            if len(objs1) == 1:
                for o in objs2:
                    scope_implications[o].add((objs2, list(objs1)[0]))
            if len(objs2) == 1:
                for o in objs1:
                    scope_implications[o].add((objs1, list(objs2)[0]))

        # Start again from scratch upon planning failure
        if optimal_model is None:
            compression_graph = connection_graph.copy()
            chunks_assembled = {}
            invalid_chunk_sequences.append(current_chunk_sequence)
            current_chunk_sequence = []
            sa_ind = 0
            join_sequence = []
            replan_attempts += 1
            query_count += collision_checker.query_count
            valid_joins_cached |= collision_checker.valid_joins
            invalid_joins_cached |= collision_checker.invalid_joins
            continue

        # Project model into join action sequence and extract action
        # parameters for each join action
        projection = sorted([
            atm for atm in optimal_model[0]
            if atm.name=="occ" and atm.arguments[0].name=="join"
        ], key=lambda atm: atm.arguments[1].number)
        projection = [
            atm.arguments[0].arguments for atm in projection
        ]
        arg_val = lambda a: a.name if a.type == SymbolType.Function \
            else f"i{a.number}_{sa_ind}"
        join_sequence += [
            (
                arg_val(s1), arg_val(s2),   # Involved subassemblies
                arg_val(o1), arg_val(o2),   # Involved objects
                f"s{sa_ind}" if i==len(projection)-1 \
                    else f"i{i}_{sa_ind}"   # Naming resultant subassembly
            )
            for i, (s1, s2, o1, o2) in enumerate(projection)
        ]

        # Update connection graph by replacing the collection of nodes with
        # a new node representing the subassembly newly assembled from the
        # selected chunk
        compression_graph.add_node(f"s{sa_ind}")
        for u, v in list(compression_graph.edges):
            if u in chunk:
                if u in compression_graph: compression_graph.remove_node(u)
                if v not in chunk: compression_graph.add_edge(f"s{sa_ind}", v)
            if v in chunk:
                if v in compression_graph: compression_graph.remove_node(v)
                if u not in chunk: compression_graph.add_edge(u, f"s{sa_ind}")

        # Tracking sequence of selected chunks, so as to remember invalid
        # sequences that led to deadend states and prevent following the
        # exact same sequence again
        current_chunk_sequence.append(chunk)
        # Remember which components constitute the new subassembly, while
        # removing old subassemblies in chunk that have been merged into
        # new subassembly
        chunks_assembled[f"s{sa_ind}"] = {
            o for s in chunk if s in chunks_assembled
            for o in chunks_assembled[s]
        } | {
            o for o in chunk if o not in chunks_assembled
        }
        for s in chunk:
            if s in chunks_assembled:
                del chunks_assembled[s]

        sa_ind += 1

    replan_attempts += 1
    query_count += collision_checker.query_count
    log_msg = f"Working plan found after {replan_attempts} (re)planning attempts "
    log_msg += f"({query_count} calls total)"
    logger.info(log_msg)

    # Instantiate the high-level plan into an actual action sequence, with
    # all the necessary information fleshed out (to be passed to the Unity
    # environment as parameters)
    get_conc = lambda s: agent.lt_mem.lexicon.s2d[("va", s)][0][1]
    pick_up_action = [get_conc("pick_up_left"), get_conc("pick_up_right")]
    drop_action = [get_conc("drop_left"), get_conc("drop_right")]
    assemble_action = [
        get_conc("assemble_right_to_left"), get_conc("assemble_left_to_right")
    ]

    action_sequence = []
    hands = [None, None]
    obj2node_map = { v: k for k, v in collision_checker.node2obj_map.items() }

    for s1, s2, o1, o2, res in join_sequence:
        if s1 in hands or s2 in hands:
            # Immediately using the subassembly built in the previous join
            empty_side = hands.index(None)
            occupied_side = list({0, 1} - {empty_side})[0]

            # Pick up the other involved subassembly with the empty hand
            pick_up_target = s2 if hands[occupied_side] == s1 else s1
            action_sequence.append((
                pick_up_action[empty_side], (pick_up_target,)
            ))
        else:
            if any(held is not None for held in hands):
                # One hand occupied, but held object needs to be dropped as
                # it is not used in this step
                empty_side = hands.index(None)
                occupied_side = list({0, 1} - {empty_side})[0]

                action_sequence.append((drop_action[occupied_side], None))

            # Boths hand are (now) empty; pick up s1 and s2 on the left and
            # the right side each
            action_sequence.append((pick_up_action[0], (s1,)))
            action_sequence.append((pick_up_action[1], (s2,)))

        # Determine joining direction by comparing the degrees of the involved
        # part nodes: smaller to larger. If the degrees are equal, break tie
        # by comparing average neighbor degree.
        deg_handle = lambda n: (
            connection_graph.degree[n],
            nx.average_neighbor_degree(connection_graph)[n]
        )
        if deg_handle(o1) >= deg_handle(o2):
            # o1 is more 'central', o2 is more 'peripheral'; join o2 to o1
            if s1 in hands:
                direction = 1 if hands.index(None) == 0 else 0
            elif s2 in hands:
                direction = 0 if hands.index(None) == 0 else 1
            else:
                direction = 0
        else:
            # o2 is more 'central', o1 is more 'peripheral'; join o1 to o2
            if s1 in hands:
                direction = 0 if hands.index(None) == 0 else 1
            elif s2 in hands:
                direction = 1 if hands.index(None) == 0 else 0
            else:
                direction = 1

        # Join s1 and s2 at appropriate contact point
        node_o1 = obj2node_map[o1]; node_o2 = obj2node_map[o2]
        cp_o1, cp_o2 = connect_edges[(node_o1, node_o2)]
        cp_o1 = re.findall(r"p_(.*)_(.*)$", cp_o1)
        cp_o2 = re.findall(r"p_(.*)_(.*)$", cp_o2)
        cp_o1 = (int(cp_o1[0][0]), int(cp_o1[0][1]))
        cp_o2 = (int(cp_o2[0][0]), int(cp_o2[0][1]))
        if s1 in hands:
            if hands.index(None) == 0:
                # s1 held in right hand
                join_params = (s2, s1, o2, o1, cp_o2, cp_o1, res)
            else:
                # s1 held in left hand
                join_params = (s1, s2, o1, o2, cp_o1, cp_o2, res)
        elif s2 in hands:
            if hands.index(None) == 0:
                # s2 held in right hand
                join_params = (s1, s2, o1, o2, cp_o1, cp_o2, res)
            else:
                # s2 held in left hand
                join_params = (s2, s1, o2, o1, cp_o2, cp_o1, res)
        else:
            # Use join parameter order as-is
            join_params = (s1, s2, o1, o2, cp_o1, cp_o2, res)
        action_sequence.append((assemble_action[direction], join_params))

        # Update hands status
        hands = [None, None]
        hands[direction] = res

    # Drop the finished product on the table
    empty_side = hands.index(None)
    occupied_side = list({0, 1} - {empty_side})[0]
    action_sequence.append((drop_action[occupied_side], None))

    # Agent's 'decisions' as to how each object is classified as instances
    # of (known) visual concepts
    recognitions = {
        oi: assembly_hierarchy.nodes[obj2node_map[oi]]["choice_conc"]
        for oi in connection_graph
    }

    return action_sequence, recognitions

class _PhysicalAssemblyPropagator:
    def __init__(self, connections, objs_data, cheat_sheet, test_result_cached):
        connection_graph, contacts = connections
        nodes, nonobj_ids = objs_data
        collision_table, part_names, cp_names = cheat_sheet
        valid_joins, invalid_joins = test_result_cached

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

        # (Injective) Mapping of individual objects to corresponding atomic
        # part instances in the collision table
        self.node2obj_map = {
            n: nodes[n]["obj_used"] or nonobj_ids[n]
            for u, v in contacts
            for n in [u, v]
        }
        obj2inst_map = {}; inst2obj_map = {}
        while len(obj2inst_map) != len(self.node2obj_map):
            for n, obj in self.node2obj_map.items():
                # If already mapped
                if obj in obj2inst_map: continue

                # See if uniquely determined by part concept alone
                conc_n = nodes[n]["choice_conc"]
                conc_n = part_group_inv[part_names[conc_n]]
                conc_sgn = (conc_n,) + (None,)*3
                if conc_sgn in template_parts_inv:
                    # Match found
                    inst = template_parts_inv[conc_sgn]
                    obj2inst_map[obj] = inst
                    inst2obj_map[inst] = obj
                    continue

                # Matching with full signature if reached here
                for (u, v), (cp_u, cp_v) in contacts.items():
                    if n not in (u, v): continue

                    cp_u = re.findall(r"p_(.*)_(.*)$", cp_u)
                    cp_v = re.findall(r"p_(.*)_(.*)$", cp_v)
                    cp_u = (int(cp_u[0][0]), int(cp_u[0][1]))
                    cp_v = (int(cp_v[0][0]), int(cp_v[0][1]))
                    cp_u = cp_indexing_inv[cp_names[cp_u]]
                    cp_v = cp_indexing_inv[cp_names[cp_v]]

                    other = v if n == u else u
                    if self.node2obj_map[other] not in obj2inst_map: continue
                    inst_other = obj2inst_map[self.node2obj_map[other]]

                    cp_n = cp_u if n == u else cp_v
                    cp_other = cp_v if n == u else cp_u

                    full_sgn = (conc_n, inst_other, cp_n, cp_other)
                    if full_sgn in template_parts_inv:
                        # Match found
                        inst = template_parts_inv[full_sgn]
                        obj2inst_map[self.node2obj_map[n]] = inst
                        inst2obj_map[inst] = self.node2obj_map[n]
                        break

        # Collision matrix translated to accommodate the context
        self.collision_table = {
            tuple(int(inst) for inst in pair.split(",")): set(colls)
            for pair, colls in collision_table["pairwise_collisions"].items()
        }
        self.collision_table = {
            (inst2obj_map[o1], inst2obj_map[o2]) : colls
            for (o1, o2), colls in self.collision_table.items()
            if o1 in inst2obj_map and o2 in inst2obj_map
        }
        self.collision_table = self.collision_table | {
            (i2, i1): {-coll_dir for coll_dir in colls}
            for (i1, i2), colls in self.collision_table.items()
        }           # Also list reverse directions in lower triangle

        # For tracking per-thread status
        self.assembly_status = None

        # For caching inclusion-minimal invalid object set pairs that
        # can/cannot be joined
        self.valid_joins = valid_joins
        self.invalid_joins = invalid_joins

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
                and atm.symbol.positive     # Dismiss -holds() atoms!

            if occ_join:
                lit = init.solver_literal(atm.literal)
                init.add_watch(lit)
                t = atm.symbol.arguments[1].number
                s1 = arg_val(arg1.arguments[0])
                s2 = arg_val(arg1.arguments[1])
                o1 = arg_val(arg1.arguments[2])
                o2 = arg_val(arg1.arguments[3])

                self.join_actions_a2l[(s1, s2, o1, o2, t)] = lit
                self.join_actions_l2a[lit].add((s1, s2, o1, o2, t))

            if holds_part_of:
                lit = init.solver_literal(atm.literal)
                init.add_watch(lit)
                t = atm.symbol.arguments[1].number
                o = arg_val(arg1.arguments[0])
                s = arg_val(arg1.arguments[1])

                self.part_of_fluents_a2l[(o, s, t)] = lit
                self.part_of_fluents_l2a[lit].add((o, s, t))

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
                for s1, s2, o1, o2, t in self.join_actions_l2a[lit]:
                    status["joins"][t] = (s1, s2, o1, o2)
                    updated.add((s1, t))
                    updated.add((s2, t))

            if lit in self.part_of_fluents_l2a:
                # Atomic (non-)object o is part of subassembly s at time step t
                for o, s, t in self.part_of_fluents_l2a[lit]:
                    if o in status["parts_inv"][t]:
                        # An atomic (non-)object cannot belong to more than one
                        # subassembly at each time step! Not explicitly prohibited
                        # by some program rule (only entailed) but let's add the
                        # set of fluents as a nogood in advance.
                        s_dup = status["parts_inv"][t][o]
                        dup_lit = self.part_of_fluents_a2l[(o, s_dup, t)]
                        may_continue = control.add_nogood([lit, dup_lit]) \
                            and control.propagate()
                        if not may_continue: return

                    status["parts"][t][s].add(o)
                    status["parts_inv"][t][o] = s
                    updated.add((s, t))

        # After the round of updates, test if each join action would entail
        # a collision-free path
        for t, (s1, s2, o1, o2) in status["joins"].items():
            if not ((s1, t) in updated or (s2, t) in updated):
                # Nothing about this join action has been updated, can skip
                continue

            objs_s1 = frozenset(status["parts"][t][s1])
            objs_s2 = frozenset(status["parts"][t][s2])

            # Checking if join can be verified to be valid without querying
            # the oracle. Note that if either of objs_s1 or objs_s2 is empty,
            # the vacuous join will always be treated as 'not impossible';
            # i.e., not ruled out due to physical collision.
            for cached_s1, cached_s2 in self.valid_joins:
                if objs_s1 <= cached_s1 and objs_s2 <= cached_s2:
                    verified_by_cache = True
                    break
                if objs_s1 <= cached_s2 and objs_s2 <= cached_s1:
                    verified_by_cache = True
                    break
            else:
                verified_by_cache = False

            if verified_by_cache:
                join_possible = True
            else:
                subset_cached_as_invalid = any(
                    (objs_s1 >= s1 and objs_s2 >= s2) or \
                        (objs_s1 >= s2 and objs_s2 >= s1)
                    for s1, s2 in self.invalid_joins
                )
                if subset_cached_as_invalid:
                    # Some subset pair recognized and cached as invalid
                    join_possible = False
                else:
                    # Test against table needed; bipartite atomic-atomic
                    # collision check, then union across the pairwise checks
                    # to obtain collision test result
                    collision_directions = set.union(*[
                        self.collision_table.get((o_s1, o_s2), set())
                        for o_s1, o_s2 in product(objs_s1, objs_s2)
                    ])
                    self.query_count += 1       # Update count
                    # Collision-free join is possible when the resultant
                    # union does not have six members, standing for each
                    # direction of assembly, hence a feasible join path
                    join_possible = len(collision_directions) < 6

            if join_possible:
                # Cache valid join
                self.valid_joins.add(frozenset([objs_s1, objs_s2]))
            else:
                # Join of this subassembly pair proved to be unreachable
                # while including current object members; add nogood
                # as appropriate and return
                if not subset_cached_as_invalid:
                    minimal_pair = self.analyze_collision((objs_s1, objs_s2))
                    if minimal_pair is not None:
                        self.invalid_joins.add(minimal_pair)

                join_lit = self.join_actions_a2l[(s1, s2, o1, o2, t)]
                part_of_lits = [
                    self.part_of_fluents_a2l[(o, s1, t)] for o in objs_s1
                ] + [
                    self.part_of_fluents_a2l[(o, s2, t)] for o in objs_s2
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
                for s1, s2, o1, o2, t in self.join_actions_l2a[lit]:
                    if t in status["joins"]: 
                        assert status["joins"][t] == (s1, s2, o1, o2)
                        del status["joins"][t]

            if lit in self.part_of_fluents_l2a:
                # Atomic (non-)object o is actually not a part of s at
                # time step t
                for o, s, t in self.part_of_fluents_l2a[lit]:
                    if o in status["parts"][t][s]:
                        status["parts"][t][s].remove(o)
                        del status["parts_inv"][t][o]

    def analyze_collision(self, objs_pair):
        """
        Helper method factored out for analyzing an object set pair that
        caused collision, finding an inclusion-minimal object set pair
        """
        objs1, objs2 = objs_pair

        # Iterate over the lattice of pair of possible subset sizes of
        # objs1, objs2, sorted so that 'smaller' size pairs are always
        # processed earlier than 'larger' ones
        pair_sizes = sorted(
            (min(size1, size2) + 1, size1 + 1, size2 + 1)
            for size1, size2 in product(range(len(objs1)), range(len(objs2)))
        )
        for _, size1, size2 in pair_sizes:
            if size1 == size2 == 1:
                # Trivially valid joins, given legitimate target structure
                continue

            # Obtain every possible subsets of objs1, objs2 of resp. sizes
            # and iterate across possible pairs
            subsets1 = combinations(objs1, size1)
            subsets2 = combinations(objs2, size2)
            for ss1, ss2 in product(subsets1, subsets2):
                ss1 = frozenset(ss1); ss2 = frozenset(ss2)

                ss_subgraph = self.connection_graph.subgraph(
                    frozenset.union(ss1, ss2)
                )
                # Process the pair only if the subgraph is connected
                if not nx.is_connected(ss_subgraph):
                    continue

                # Test this pair against table
                collision_directions = set.union(*[
                    self.collision_table.get((o_s1, o_s2), set())
                    for o_s1, o_s2 in product(ss1, ss2)
                ])
                join_possible = len(collision_directions) < 6

                if not join_possible:
                    # Potential minimal source of collision found; return
                    return frozenset([ss1, ss2])

        # No connected source found
        return None

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

    if target in exec_state["connection_graphs"]:
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

    elif "env_handle" in target_info:
        # Existing atomic object, provide the environment side name
        env_handle = target_info["env_handle"]
        estim_type = exec_state["recognitions"][target]
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
        singleton_gr = nx.DiGraph()
        singleton_gr.add_node(target)
        exec_state["connection_graphs"][target] = singleton_gr

    else:
        # Non-object, different strategies adopted for coping with the
        # grounding failure. Language-less agents do not share vocabulary
        # referring to parts and thus need to report inability to proceed.
        # Language-conversant agents can first inquire the agent whether
        # there exists an instance of a part they need.

        # Target concept, as recognized (estimation might be incorrect)
        target_conc = exec_state["recognitions"][target]
        target_sym = agent.lt_mem.lexicon.d2s[("pcls", target_conc)][0][1]

        # NL surface form and corresponding logical form
        surface_form = f"Is there a {target_sym}?"
        gq = None; bvars = {"x0"}; ante = []
        cons = [("n", target_sym, ["x0"])]
        logical_form = (gq, bvars, ante, cons)

        mood = "?"      # Interrogative

        # Referents & predicates info
        referents = {"x0": { "entity": None, "rf_info": {} } }
        predicates = { "pc0": (("sp", "unable"), f"pcls_{target_conc}") }

        # Append to & flush generation buffer
        agent.lang.dialogue.to_generate.append(
            (logical_form, surface_form, mood, referents, predicates, {})
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
    obj_left, obj_right = action_params[:2]
    part_left, part_right = action_params[2:4]
    cp_left, cp_right = action_params[4:6]
    product_name = action_params[6]

    exec_state = agent.planner.execution_state      # Shortcut var

    mnp_state_left = exec_state["manipulator_states"][0]
    mnp_state_right = exec_state["manipulator_states"][1]
    held_left, pose_mnp_left, poses_part_left = mnp_state_left
    held_right, pose_mnp_right, poses_part_right = mnp_state_right
    graph_left = exec_state["connection_graphs"][held_left]
    graph_right = exec_state["connection_graphs"][held_right]

    # Sanity checks
    assert obj_left == held_left and obj_right == held_right
    assert part_left in graph_left and part_right in graph_right

    tmat_part_left = transformation_matrix(*poses_part_left[part_left])
    tmat_part_right = transformation_matrix(*poses_part_right[part_right])

    part_conc_left = exec_state["recognitions"][part_left]
    part_conc_right = exec_state["recognitions"][part_right]
    structure_3d_left = agent.lt_mem.exemplars.object_3d[part_conc_left]
    structure_3d_right = agent.lt_mem.exemplars.object_3d[part_conc_right]

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
    tmat_cp_local_left = structure_3d_left[3][cp_left[0]][cp_left[1]][0]
    tmat_cp_local_right = structure_3d_right[3][cp_right[0]][cp_right[1]][0]
    tmat_cp_local_left = transformation_matrix(*tmat_cp_local_left)
    tmat_cp_local_right = transformation_matrix(*tmat_cp_local_right)
    if action_name.endswith("left"):
        # Target is on left, move right manipulator
        tmat_tgt_part = tmat_part_left
        tmat_src_part = tmat_part_right
        tmat_tgt_cp_local = tmat_cp_local_left
        tmat_src_cp_local = tmat_cp_local_right
        tmat_mnp_before = transformation_matrix(*pose_mnp_right)
    else:
        # Target is on right, move left manipulator
        assert action_name.endswith("right")
        tmat_tgt_part = tmat_part_right
        tmat_src_part = tmat_part_left
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
    graph_joined = nx.compose(graph_left, graph_right)
    if action_name.endswith("left"):
        graph_joined.add_edge(part_left, part_right, rel_tr=tmat_rel)
    else:
        graph_joined.add_edge(part_right, part_left, rel_tr=tmat_rel)
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
    agent_action = [
        (action_name, {
            "parameters": (
                f"str|{product_name}",      # Also provide product name
                f"floats|{rot_serialized}", f"floats|{pos_serialized}"
            ),
            "pointing": {}
        }),
        ("generate", (
            f"# Action: {action_name}({','.join(action_params[:4])},{product_name})",
            {})
        )
    ]

    # Wait before executing the rest of the plan until teacher reacts (either with
    # silent observation or interruption)
    agent.planner.agenda.appendleft(("execute_command", (None, None)))

    return agent_action

def handle_action_effect(agent, effect):
    """
    Update agent's internal representation of relevant environment states, in
    the face of obtaining feedback after performing a primitive environmental
    action. In our scope, we are concerned with effects of two types of actions:
    'pick_up' and 'assemble' actions.
    """
    effect_lit = [lit for lit in effect if lit.name.startswith("arel")][0]
    action_name = int(effect_lit.name.split("_")[-1])
    action_name = agent.lt_mem.lexicon.d2s[("arel", action_name)][0][1]

    # Shortcut vars
    referents = agent.lang.dialogue.referents
    exec_state = agent.planner.execution_state      # Shortcut var
    part_identifiers = exec_state["part_identifiers"]

    # Utility method for parsing serialized float lists passed as action effect
    parse_floats = lambda ai: tuple(
        float(v)
        for v in referents["dis"][effect_lit.args[ai][0]]["name"].split("/")
    )

    if action_name.startswith("pick_up"):
        # Pick-up action effects: Unity gameObject name of the pick-up target*,
        # pose of moved manipulator, poses of all atomic parts included in the
        # pick-up target subassembly after picking up (*: not used by learner
        # agent)
        manip_ind = 0 if action_name.endswith("left") else 1
        manip_state = exec_state["manipulator_states"][manip_ind]
        num_parts = int(referents["dis"][effect_lit.args[5][0]]["name"])

        # Manipulator pose after picking up
        manip_pose = (
            flip_quaternion_y(xyzw2wxyz(parse_floats(3))),
            flip_position_y(parse_floats(4))
        )

        # Track string identifiers referring to each part instance,
        # which is revealed every time a singleton wrapper object
        # is picked up in Unity environment (identifiable by number
        # of parts whose poses are reported: 1 for singletons)
        if num_parts == 1:
            part_uid = referents["dis"][effect_lit.args[6][0]]["name"]
            part_identifiers[part_uid] = manip_state[0]

        # Poses of individual parts in the object picked up
        part_poses = {}
        for i in range(num_parts):
            part_uid = referents["dis"][effect_lit.args[6+3*i][0]]["name"]
            part_poses[part_identifiers[part_uid]] = (
                flip_quaternion_y(xyzw2wxyz(parse_floats(6+3*i+1))),
                flip_position_y(parse_floats(6+3*i+2))
            )

        exec_state["manipulator_states"][manip_ind] = \
            (manip_state[0],) + (manip_pose, part_poses)

    if action_name.startswith("assemble"):
        # Assemble action effects: closest contact point pairs after the joining
        # movement*, pose of target-side manipulator, masks of all atomic parts
        # included in the assembled product (*: not used by learner agent)
        manip_ind = 0 if action_name.endswith("left") else 1
        manip_state = exec_state["manipulator_states"][manip_ind]
        num_cp_pairs = int(referents["dis"][effect_lit.args[4][0]]["name"])
        ind_offset = 5 + 4 * num_cp_pairs
        num_parts = int(referents["dis"][effect_lit.args[ind_offset][0]]["name"])

        # Target-site manipulator pose after joining
        manip_pose = (
            flip_quaternion_y(xyzw2wxyz(parse_floats(2))),
            flip_position_y(parse_floats(3))
        )

        # Poses of individual parts in the object picked up
        part_poses = {}; 
        for i in range(num_parts):
            arg_start_ind = ind_offset + 1 + 3 * i
            part_uid = referents["dis"][effect_lit.args[arg_start_ind][0]]["name"]
            part_poses[part_identifiers[part_uid]] = (
                flip_quaternion_y(xyzw2wxyz(parse_floats(arg_start_ind+1))),
                flip_position_y(parse_floats(arg_start_ind+2))
            )

        exec_state["manipulator_states"][manip_ind] = \
            (manip_state[0],) + (manip_pose, part_poses)
