"""
Implements dialogue-related composite actions
"""
import os
import re
import random
from copy import deepcopy
from functools import reduce
from itertools import permutations, combinations, product
from collections import defaultdict

import yaml
import torch
import numpy as np
import networkx as nx
from clingo import Control, SymbolType, Number, Function

from ..vision.utils import (
    xyzw2wxyz, flip_position_y, flip_quaternion_y,
    blur_and_grayscale, visual_prompt_by_mask, pose_estimation_with_mask,
    transformation_matrix
)
from ..lpmln import Literal
from ..lpmln.utils import flatten_cons_ante, wrap_args
from ..memory.kb import KnowledgeBase
from ..symbolic_reasoning.query import query
from ..symbolic_reasoning.utils import rgr_extract_likelihood, rgr_replace_likelihood


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
    translated = agent.symbolic.translate_dialogue_content(agent.lang.dialogue)

    ti, ci = utt_pointer
    (_, question), _ = translated[ti][1][ci]

    if question is None:
        # Question cannot be answered for some reason
        return
    else:
        # Schedule to answer the question
        agent.planner.agenda.insert(0, ("answer_Q", utt_pointer))
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
    translated = agent.symbolic.translate_dialogue_content(agent.lang.dialogue)

    if agent.lang.dialogue.clause_info[f"t{ti}c{ci}"]["domain_describing"]:
        _answer_domain_Q(agent, utt_pointer, translated)
    else:
        _answer_nondomain_Q(agent, utt_pointer, translated)

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
    translated = agent.symbolic.translate_dialogue_content(agent.lang.dialogue)

    ti, ci = utt_pointer
    parse, raw, _ = translated[ti][1][ci]
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
        agent.planner.agenda.insert(0, ("execute_command", (action_type, action_params)))
        agent.lang.dialogue.unexecuted_commands.remove(utt_pointer)
        return

    else:
        # Command cannot be executed for some reason; report inability and dismiss
        # the command
        ri_command = f"t{ti}c{ci}"                      # Denotes original request

        # New dialogue turn & clause index for the answer to be provided
        ti_new = len(agent.lang.dialogue.record)
        ci_new = len(agent.lang.dialogue.to_generate)
        ri_inability = f"t{ti_new}c{ci_new}"            # Denotes the inability state event
        ri_self = f"t{ti_new}c{ci_new}x0"               # Denotes self (i.e., agent)
        tok_ind = (f"t{ti_new}", f"c{ci_new}", "pc0")   # Denotes 'unable' predicate

        # Update cognitive state w.r.t. value assignment and word sense
        agent.symbolic.value_assignment[ri_self] = "_self"
        agent.symbolic.word_senses[tok_ind] = (("sp", "unable"), "sp_unable")

        # Corresponding logical form
        gq = None; bvars = None; ante = []
        cons = [
            ("sp", "unable", [ri_self, ri_command]),
            ("sp", "pronoun1", [ri_self])
        ]

        agent.lang.dialogue.unexecuted_commands.remove(utt_pointer)
        agent.lang.dialogue.to_generate.append(
            (
                (gq, bvars, ante, cons),
                f"I am unable to {raw[0].lower()}{raw[1:]}",
                ".",
                { ri_self: { "source_evt": ri_inability } },
                {}
            )
        )

        return

def execute_command(agent, action_spec):
    """
    Execute a command (that was deemed executable before by `attempt_command`
    method) by appropriate planning, based on the designated action type and
    parameters provided as arguments.
    """
    action_type, action_params = action_spec

    # Currently considered commands: some long-term commands that requiring
    # long-horizon planning (e.g., 'build'), and some primitive actions that
    # are to be executed---that is, signaled to Unity environment---immediately
    action_name = agent.lt_mem.lexicon.d2s[("arel", action_type)][0][1]
    match action_name:
        case "build":
            _execute_build(agent, action_params)
            return

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
        3) Solving the compiled ASP program with clingo, pushing the obtained
            sequence of atomic actions onto agenda stack
    """
    # Convert current visual scene (from ensemble prediction) into ASP fact
    # literals (note difference vs. agent.lt_mem.kb.visual_evidence_from_scene
    # method, which is for probabilistic reasoning by LP^MLN)
    threshold = 0.25; observations = set()
    likelihood_values = np.stack([
        obj["pred_cls"] for obj in agent.vision.scene.values()
    ])
    min_val = max(likelihood_values.min(), threshold)
    max_val = likelihood_values.max()
    val_range = max_val - min_val

    for oi, obj in agent.vision.scene.items():
        # Heuristic: consider predictions of only the highest likelihood
        # values for each object, instead of listing all predictions with
        # values above threshold
        ci = obj["pred_cls"].argmax().item()
        val = obj["pred_cls"].max().item()
        # Ignore any proposal not worth tracking
        if val < threshold: continue

        # Normalize & discretize within [0,20]; the more bins we
        # use for discrete approximation, the more time it takes to
        # solve the program
        nrm_val = (val-min_val) / val_range
        dsc_val = int(nrm_val * 20)

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
                        f"connection_signature",
                        wrap_args(sa_conc, ti, u, v, sgn_u, sgn_v, cp_u, cp_v)
                    )
                    assembly_pieces.add(conn_sgn_lit)

    # Encode assembly target info into ASP fact literal
    build_target = list(action_params.values())[0][0].split("_")
    target_lit = Literal("build_target", wrap_args(int(build_target[1])))

    # Load ASP encoding for selecting & committing to a goal structure
    lp_path = os.path.join(agent.cfg.paths.assets_dir, "planning_encoding")
    commit_ctl = Control(["--warn=none"])
    commit_ctl.configuration.solve.models = 0
    commit_ctl.configuration.solve.opt_mode = "opt"
    commit_ctl.load(os.path.join(lp_path, "goal_selection.lp"))

    # Add and ground all the fact literals obtained above
    all_lits = sorted(
        observations | assembly_pieces | {target_lit}, key=lambda x: x.name
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
    min_chunk_size = 6; chunks_assembled = {}
    invalid_chunk_sequences = []; current_chunk_sequence = []
    sa_ind = 0
    join_sequence = []
    # Record every 'assembly scope implication' such that if a set of certain
    # atomic objects are to be included in a chunk, some other object must
    # be included in the set as well, so as to avoid deadends due to violations
    # of 'precedence constraints'.
    scope_implications = { n: set() for n in connection_graph }

    replan_attempts = 0
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
            print(f"Deadend reached; starting from scratch... ({replan_attempts})")
            continue

        chunk_subgraph = compression_graph.subgraph(chunk_selected)

        # Load ASP encoding for planning sequence of atomic actions
        collision_checker = _PhysicalAssemblyPropagator(
            (connection_graph, connect_edges),
            (dict(assembly_hierarchy.nodes(data=True)), nonobj_ids),
            (collision_table, part_names, cp_oracle)
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
            print(f"Deadend reached; starting from scratch... ({replan_attempts})")
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

            # Keep the joining direction as same as previous join
            direction = occupied_side
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

            # Random join direction
            direction = random.randint(0, 1)

        # Join s1 and s2 at appropriate contact point
        node_o1 = obj2node_map[o1]; node_o2 = obj2node_map[o2]
        cp_o1, cp_o2 = connect_edges[(node_o1, node_o2)]
        cp_o1 = re.findall(r"p_(.*)_(.*)$", cp_o1)
        cp_o2 = re.findall(r"p_(.*)_(.*)$", cp_o2)
        cp_o1 = (int(cp_o1[0][0]), int(cp_o1[0][1]))
        cp_o2 = (int(cp_o2[0][0]), int(cp_o2[0][1]))
        action_sequence.append((
            assemble_action[direction], (s1, s2, o1, o2, cp_o1, cp_o2, res)
        ))

        # Update hands status
        hands = [None, None]
        hands[direction] = res

    # Drop the finished product on the table
    empty_side = hands.index(None)
    occupied_side = list({0, 1} - {empty_side})[0]
    action_sequence.append((drop_action[occupied_side], None))

    # Push appropriate agenda items and finish
    agent.planner.agenda += [
        ("execute_command", (action_type, action_params))
        for action_type, action_params in action_sequence
    ]

    # Dict storing current progress (from agent's viewpoint)
    agent.assembly_progress = {
        "manipulator_states": [(None, None, None), (None, None, None)],
        # Resp. left & right held object, manipulator pose, component masks
        "committed_parts": {
            oi: assembly_hierarchy.nodes[obj2node_map[oi]]["choice_conc"]
            for oi in connection_graph
        },
        "subassembly_components": {}
    }

class _PhysicalAssemblyPropagator:
    def __init__(self, connections, objs_data, cheat_sheet):
        connection_graph, contacts = connections
        nodes, nonobj_ids = objs_data
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
        # cannot be joined
        self.invalid_joins = set()

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

            # Bipartite atomic-atomic collision check, then union across
            # the pairwise checks to obtain collision test result
            objs_s1 = frozenset(status["parts"][t][s1])
            objs_s2 = frozenset(status["parts"][t][s2])
            
            if len(objs_s1) == 0 or len(objs_s2) == 0:
                # Vacuous join, always treat as 'not impossible' (i.e.,
                # doesn't rule out model due to physical collision)
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
                    # Test against table needed
                    collision_directions = set.union(*[
                        self.collision_table.get((o_s1, o_s2), set())
                        for o_s1, o_s2 in product(objs_s1, objs_s2)
                    ])
                    # Collision-free join is possible when the resultant
                    # union does not have six members, standing for each
                    # direction of assembly, hence a feasible join path
                    join_possible = len(collision_directions) < 6

            if not join_possible:
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
    target_info = agent.vision.scene[target]

    if "pred_mask" in target_info:
        # Existing object, provide the segmentation mask as action parameter
        # to the Unity environment
        agent_action = [
            (action_name, {
                "parameters": ("str|@DemRef",),     # Signals "provided as mask"
                "pointing": { (4, 11): target_info["pred_mask"] }
            })
        ]
        manip_ind = 0 if action_name.endswith("left") else 1
        manip_state = agent.assembly_progress["manipulator_states"][manip_ind]
        agent.assembly_progress["manipulator_states"][manip_ind] = \
            (action_params[0],) + manip_state[1:]
        agent.assembly_progress["subassembly_components"][target] = {target}
    else:
        # Non-object, need to match an object labelled to be an instance of
        # the targetted concept
        if False:
            agent_action = []
        else:
            agent_action = [
                ("generate", { "utterance": "foo", "pointing": {} })
            ]

    return agent_action

def _execute_drop(agent, action_name):
    """
    Drop whatever is held in the manipulator on the designated side. Not a lot
    of complications, no action parameters to consider; just signal the Unity
    environment.
    """
    print(0)

def _execute_assemble(agent, action_name, action_params):
    """
    Assemble two subassemblies held in each manipulator at the designated objects
    and contact points. Agent must provide transformation (3D pose difference)
    of a manipulator that will achieve such join as action parameters to Unity,
    as opposed to the user who knows the ground-truth name handles of objects
    and contact points in the environment. Perform pose estimation of relevant
    objects from agent's visual input and compute pose difference. Necessary
    information must have been provided from Unity environment as 'effects' of
    previous actions.
    """
    state_left = agent.assembly_progress["manipulator_states"][0]
    state_right = agent.assembly_progress["manipulator_states"][1]
    obj_left, pose_left, masks_left = state_left
    obj_right, pose_right, masks_right = state_right
    components_left = agent.assembly_progress["subassembly_components"][obj_left]
    components_right = agent.assembly_progress["subassembly_components"][obj_right]
    types_left = {agent.assembly_progress["committed_parts"][c] for c in components_left}
    types_right = {agent.assembly_progress["committed_parts"][c] for c in components_right}

    # Classify 'reference' components with the masks with the largest sizes on resp.
    # left & right, then estimate poses
    ref_mask_left = sorted(masks_left, reverse=True, key=lambda msk: msk.sum())[0]
    ref_mask_right = sorted(masks_right, reverse=True, key=lambda msk: msk.sum())[0]
    # Extracting visual features
    scene_img = agent.vision.latest_inputs[-1]
    visual_prompts = visual_prompt_by_mask(
        scene_img, blur_and_grayscale(scene_img), [ref_mask_left, ref_mask_right]
    )
    vis_model = agent.vision.model; vis_model.eval()
    with torch.no_grad():
        f_vecs = []
        for vp in visual_prompts:
            vp_processed = vis_model.dino_processor(images=vp, return_tensors="pt")
            vp_pixel_values = vp_processed.pixel_values.to(vis_model.dino.device)
            vp_dino_out = vis_model.dino(pixel_values=vp_pixel_values, return_dict=True)
            f_vecs.append(vp_dino_out.pooler_output.cpu().numpy()[0])
    # Few-shot classification, with choices constrained among concepts which are
    # known (as agent deems it) to be included in the left/right objects
    fs_pred_left = agent.vision.fs_conc_pred(agent.lt_mem.exemplars, f_vecs[0], "pcls")
    fs_pred_right = agent.vision.fs_conc_pred(agent.lt_mem.exemplars, f_vecs[1], "pcls")
    fs_pred_left = max(
        probs := { c: pr for c, pr in enumerate(fs_pred_left) if c in types_left },
        key=probs.get
    )
    fs_pred_right = max(
        probs := { c: pr for c, pr in enumerate(fs_pred_right) if c in types_right },
        key=probs.get
    )

    # Estimate poses of the reference components on each side
    structure_3d_left = agent.lt_mem.exemplars.object_3d[fs_pred_left][:3]
    structure_3d_right = agent.lt_mem.exemplars.object_3d[fs_pred_right][:3]
    with torch.no_grad():
        pose_estimation_per_view_left = pose_estimation_with_mask(
            scene_img, ref_mask_left, vis_model, structure_3d_left,
            agent.vision.camera_intrinsics
        )
        pose_estimation_per_view_right = pose_estimation_with_mask(
            scene_img, ref_mask_right, vis_model, structure_3d_right,
            agent.vision.camera_intrinsics
        )
    best_estimation_left = sorted(
        pose_estimation_per_view_left, key=lambda x: x[1]+x[2], reverse=True
    )[0]
    best_estimation_right = sorted(
        pose_estimation_per_view_right, key=lambda x: x[1]+x[2], reverse=True
    )[0]
    best_estimation_left = transformation_matrix(*best_estimation_left[0])
    best_estimation_right = transformation_matrix(*best_estimation_right[0])

    # Relation among manipulator, object & contact point transformations:
    # [Tr. of contact point in global coordinate]
    # = [Tr. of part instance in global coordinate] *
    #   [Tr. of contact point in part local coordinate]
    # = [Tr. of manipulator in global coordinate] *
    #   [Tr. of part instance in subassembly local coordinate] *
    #   [Tr. of contact point in part local coordinate]
    #
    # Based on these relations, we first obtain [Tr. of contact point in global
    # coordinate], then [Tr. of part instance in subassembly local coordinate],
    # finally [Desired Tr. of manipulator in global coordinate] by equating
    # transforms of the two contact points of interest in global coordinate.
    print(0)

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

    referents = agent.lang.dialogue.referents       # Shortcut var

    if action_name.startswith("pick_up"):
        # Pick-up action effects: pose of moved manipulator, Unity gameObject
        # name of the pick-up target (*not used by learner agent), masks of
        # all atomic parts included in the pick-up target subassembly
        manip_rotation, manip_translation, _, *component_objs = [
            arg for arg, _ in effect_lit.args[2:]
        ]
        manip_rotation = flip_quaternion_y(xyzw2wxyz([
            float(val)
            for val in referents["dis"][manip_rotation]["name"].split("/")
        ]))
        manip_translation = flip_position_y([
            float(val)
            for val in referents["dis"][manip_translation]["name"].split("/")
        ])
        manip_pose = (manip_rotation, manip_translation)
        component_masks = [
            agent.vision.scene[obj]["pred_mask"] for obj in component_objs
        ]

        manip_ind = 0 if action_name.endswith("left") else 1
        manip_state = agent.assembly_progress["manipulator_states"][manip_ind]
        agent.assembly_progress["manipulator_states"][manip_ind] = \
            (manip_state[0],) + (manip_pose, component_masks)

    if action_name.startswith("assemble"):
        print(1)

def _answer_domain_Q(agent, utt_pointer, translated):
    """
    Helper method factored out for computation of answers to an in-domain question;
    i.e. having to do with the current states of affairs of the task domain.
    """
    ti, ci = utt_pointer
    presup, question = translated[ti][1][ci][0]
    assert question is not None

    q_vars, (q_cons, q_ante) = question

    reg_gr_v, _ = agent.symbolic.concl_vis

    # New dialogue turn & clause index for the answer to be provided
    ti_new = len(agent.lang.dialogue.record)
    ci_new = 0

    # Mapping from predicate variables to their associated entities
    pred_var_to_ent_ref = {
        ql.args[0][0]: ql.args[1][0] for ql in q_cons
        if ql.name == "sp_isinstance"
    }

    qv_to_dis_ref = {
        qv: f"x{ri}t{ti_new}c{ci_new}" for ri, (qv, _) in enumerate(q_vars)
    }
    conc_type_to_pos = { "pcls": "n" }

    # Process any 'concept conjunctions' provided in the presupposition into a more
    # legible format, for easier processing right after
    if presup is None:
        restrictors = {}
    else:
        conc_conjs = defaultdict(set)
        for lit in presup[0]:
            conc_conjs[lit.args[0][0]].add(lit.name)

        # Extract any '*_subtype' statements and cast into appropriate query restrictors
        restrictors = {
            lit.args[0][0]: agent.lt_mem.kb.find_entailer_concepts(conc_conjs[lit.args[1][0]])
            for lit in q_cons if lit.name=="sp_subtype"
        }
        # Remove the '*_subtype' statements from q_cons now that they are processed
        q_cons = tuple(lit for lit in q_cons if lit.name!="sp_subtype")
        question = (q_vars, (q_cons, q_ante))

    # Ensure it has every ingredient available needed for making most informed judgements
    # on computing the best answer to the question. Specifically, scene graph outputs from
    # vision module may be omitting some entities, whose presence and properties may have
    # critical influence on the symbolic sensemaking process. Make sure such entities, if
    # actually present, are captured in scene graphs by performing visual search as needed.
    if len(agent.lt_mem.kb.entries) > 0:
        search_specs = _search_specs_from_kb(agent, question, restrictors, reg_gr_v)
        if len(search_specs) > 0:
            agent.vision.predict(None, agent.lt_mem.exemplars, specs=search_specs)

            # If new entities is registered as a result of visual search, update env
            # referent list
            new_ents = set(agent.vision.scene) - set(agent.lang.dialogue.referents["env"][-1])
            for ent in new_ents:
                mask = agent.vision.scene[ent]["pred_mask"]
                agent.lang.dialogue.referents["env"][-1][ent] = {
                    "mask": mask,
                    "area": mask.sum().item()
                }
                agent.lang.dialogue.referent_names[ent] = ent

            #  ... and another round of sensemaking
            exported_kb = agent.lt_mem.kb.export_reasoning_program()
            visual_evidence = agent.lt_mem.kb.visual_evidence_from_scene(agent.vision.scene)
            agent.symbolic.sensemake_vis(exported_kb, visual_evidence)
            agent.lang.dialogue.sensemaking_v_snaps[ti_new] = agent.symbolic.concl_vis

            agent.symbolic.resolve_symbol_semantics(agent.lang.dialogue, agent.lt_mem.lexicon)

            reg_gr_v, _ = agent.symbolic.concl_vis

    # Compute raw answer candidates by appropriately querying compiled region graph
    answers_raw, _ = agent.symbolic.query(reg_gr_v, q_vars, (q_cons, q_ante), restrictors)

    # Pick out an answer to deliver; maximum confidence
    if len(answers_raw) > 0:
        max_score = max(answers_raw.values())
        answer_selected = random.choice([
            a for (a, s) in answers_raw.items() if s == max_score
        ])
        ev_prob = answers_raw[answer_selected]
    else:
        answer_selected = (None,) * len(q_vars)
        ev_prob = None

    # From the selected answer, prepare ASP-friendly logical form of the response to
    # generate, then translate into natural language
    # (Parse the original question utterance, manipulate, then generate back)
    if len(answer_selected) == 0:
        # Yes/no question
        raise NotImplementedError
        if ev_prob < SC_THRES:
            # Positive answer
            ...
        else:
            # Negative answer
            ...
    else:
        # Wh- question
        for (qv, is_pred), ans in zip(q_vars, answer_selected):
            # Referent index in the new answer utterance
            ri = qv_to_dis_ref[qv]

            # Value to replace the designated wh-quantified referent with
            if is_pred:
                # Predicate name; fetch from lexicon
                if ans is None:
                    # No answer predicate to "What is X" question; let's simply generate
                    # "I am not sure" as answer for these cases
                    agent.lang.dialogue.to_generate.append(
                        # Will just pass None as "logical form" for this...
                        (None, "I am not sure.", {})
                    )
                    return
                else:
                    ans = ans.split("_")
                    ans = (int(ans[1]), ans[0])

                    pred_name = agent.lt_mem.lexicon.d2s[ans][0][0]

                    # Update cognitive state w.r.t. value assignment and word sense
                    agent.symbolic.value_assignment[ri] = \
                        pred_var_to_ent_ref[qv]
                    tok_ind = (f"t{ti_new}", f"c{ci_new}", "rc", "0")
                    agent.symbolic.word_senses[tok_ind] = \
                        ((conc_type_to_pos[ans[1]], pred_name), f"{ans[1]}_{ans[0]}")

                    answer_logical_form = (
                        ((pred_name, conc_type_to_pos[ans[1]], (ri,), False),), ()
                    )

                    # Split camelCased predicate name
                    splits = re.findall(
                        r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", pred_name
                    )
                    splits = [w[0].lower()+w[1:] for w in splits]
                    answer_nl = f"This is a {' '.join(splits)}."
            else:
                # Need to give some referring expression as answer; TODO? implement
                raise NotImplementedError

    # Fetch segmentation mask for the demonstratively referenced entity
    dem_mask = agent.lang.dialogue.referents["env"][-1][pred_var_to_ent_ref[qv]]["mask"]

    # Push the translated answer to buffer of utterances to generate
    agent.lang.dialogue.to_generate.append(
        ((answer_logical_form, None), answer_nl, { (0, 4): dem_mask })
    )

def _answer_nondomain_Q(agent, utt_pointer, translated):
    """
    Helper method factored out for computation of answers to a question that needs
    procedures different from those for in-domain questions for obtaining. Currently
    includes why-questions.
    """
    ti, ci = utt_pointer
    _, question = translated[ti][1][ci][0]
    assert question is not None

    _, (q_cons, _) = question

    # New dialogue turn & clause index for the answer to be provided
    ti_new = len(agent.lang.dialogue.record)
    ci_new = 0

    conc_type_to_pos = { "pcls": "n" }

    if any(lit.name=="sp_expl" for lit in q_cons):
        # Answering why-question

        # Extract target event to explain from the question; fetch the explanandum
        # statement
        expl_lits = [lit for lit in q_cons if lit.name=="sp_expl"]
        expd_utts = [
            re.search("^t(\d+)c(\d+)$", lit.args[1][0])
            for lit in expl_lits
        ]
        expd_utts = [
            (int(clause_ind.group(1)), int(clause_ind.group(2)))
            for clause_ind in expd_utts
        ]
        expd_contents = [
            translated[ti_exd][1][ci_exd][0]
            for ti_exd, ci_exd in expd_utts
        ]

        # Can treat "why ~" and "why did you think ~" as the same for our purpose
        self_think_lits = [
            [
                lit for lit in expd_cons[0]
                if lit.name=="sp_think" and lit.args[0][0]=="_self"
            ]
            for expd_cons, _ in expd_contents
        ]
        st_expd_utts = [
            [
                re.search("^t(\d+)c(\d+)$", st_lit.args[1][0])
                for st_lit in per_expd
            ]
            for per_expd in self_think_lits
        ]
        st_expd_utts = [
            [
                (int(clause_ind.group(1)), int(clause_ind.group(2)))
                for clause_ind in per_expd
            ]
            for per_expd in st_expd_utts
        ]
        # Assume all explananda consist of consequent-only rules
        st_expd_contents = [
            translated[ti_exd][1][ci_exd][0][0][0]
            for per_expd in st_expd_utts
            for ti_exd, ci_exd in per_expd
        ]
        nst_expd_contents = [
            tuple(
                lit for lit in expd_cons[0]
                if not (lit.name=="sp_think" and lit.args[0][0]=="_self")
            )
            for expd_cons, _ in expd_contents
        ]
        target_events = nst_expd_contents + st_expd_contents

        # Fetch the region graph based on which the explanandum (i.e. explanation
        # target) utterance have been made; that is, shouldn't use region graph
        # compiled *after* agent has uttered the explanandum statement
        latest_reasoning_ind = max(
            snap_ti for snap_ti in agent.lang.dialogue.sensemaking_v_snaps
            if snap_ti < ti
        )
        reg_gr_v, (kb_prog, _) = agent.lang.dialogue.sensemaking_v_snaps[latest_reasoning_ind]
        kb_prog_analyzed = KnowledgeBase.analyze_exported_reasoning_program(kb_prog)
        scene_ents = {
            a[0] for atm in reg_gr_v.graph["atoms_map"]
            for a in atm.args if isinstance(a[0], str)
        }

        # Collect previous factual statements and questions made during this dialogue
        prev_statements = []; prev_Qs = []
        for ti, (spk, turn_clauses) in enumerate(translated):
            for ci, ((rule, ques), _) in enumerate(turn_clauses):
                # Factual statement
                if rule is not None and len(rule[0])==1 and rule[1] is None:
                    prev_statements.append(((ti, ci), (spk, rule)))
                
                # Question
                if ques is not None:
                    # Here, `rule` represents presuppositions included in `ques`
                    prev_Qs.append(((ti, ci), (spk, ques, rule)))

        # Fetch teacher's last correction, which is the expected ground truth
        prev_statements_U = [
            stm for (ti, ci), (spk, stm) in prev_statements
            if spk=="Teacher" and \
                agent.lang.dialogue.clause_info[f"t{ti}c{ci}"]["domain_describing"] and \
                not agent.lang.dialogue.clause_info[f"t{ti}c{ci}"]["irrealis"]
        ]
        expected_gt = prev_statements_U[-1][0][0]

        # Fetch teacher's last probing question, which restricts the type of the answer
        # predicate anticipated by taxonomy entailment
        prev_Qs_U = [
            (ques, presup) for _, (spk, ques, presup) in prev_Qs
            if spk=="Teacher" and "sp_isinstance" in {ql.name for ql in ques[1][0]}
        ]
        probing_Q, probing_presup = prev_Qs_U[-1]
        # Process any 'concept conjunctions' provided in the presupposition into a more
        # legible format, for easier processing right after
        conc_conjs = defaultdict(set)
        for lit in probing_presup[0]:
            conc_conjs[lit.args[0][0]].add(lit.name)
        # Extract any '*_subtype' statements and cast into appropriate query restrictors
        restrictors = {
            lit.args[0][0]: agent.lt_mem.kb.find_entailer_concepts(conc_conjs[lit.args[1][0]])
            for lit in probing_Q[1][0] if lit.name=="sp_subtype"
        }

        # Find valid templates of potential explanantia for the expected ground-truth
        # answer based on the inference program
        exps_templates_gt = _find_explanans_templates(expected_gt, kb_prog_analyzed)

        for tgt_ev in target_events:
            if len(tgt_ev) == 0: continue

            # Only considers singleton events right now
            assert len(tgt_ev) == 1
            tgt_lit = tgt_ev[0]
            v_tgt_lit = Literal(f"v_{tgt_lit.name}", tgt_lit.args)

            # Find valid templates of potential explanantia for the agent answer based
            # on the inference program
            exps_templates_ans = _find_explanans_templates(tgt_lit, kb_prog_analyzed)

            # Asymmetric set difference for selecting distinguishing properties; based
            # on the notion that shared explanantia shouldn't be considered as valid
            # explanations (though they often are selected if not explicitly filtered...)
            shared_template_pairs = [
                (tpl1, tpl2)
                for tpl1, tpl2 in product(exps_templates_ans[1:], exps_templates_gt[1:])
                if Literal.entailing_mapping_btw(tpl1, tpl2)[0] == 0
            ]
            shared_templates_ans = {frozenset(tpl1) for tpl1, _ in shared_template_pairs}
            distinguishing_templates_ans = [
                tpl for tpl in exps_templates_ans[1:]
                if frozenset(tpl) not in shared_templates_ans
            ]
            selected_templates = exps_templates_ans[:1] + distinguishing_templates_ans

            # Obtain every possible instantiations of the discovered templates, then
            # flatten to a set of (grounded) potential evidence atoms
            exps_instances = _instantiate_templates(selected_templates, scene_ents)
            evidence_atoms = {
                Literal(f"v_{c_lit.name}", c_lit.args)
                for conjunction in exps_instances for c_lit in conjunction
                if c_lit.name.startswith("pcls")
            }       # Considering visual evidence for class & attributes concepts only

            # Manually select 'competitor' events that could've been the answer
            # in place of the true one (not necessarily mutually exclusive)
            possible_answers = [
                atm for atm in reg_gr_v.graph["atoms_map"]
                if (
                    atm.name in restrictors[probing_Q[0][0][0]] and
                    atm.args==tgt_lit.args
                )
            ]
            competing_evts = [
                (atm,) for atm in possible_answers if atm.name!=tgt_lit.name
            ]

            # Obtain a sufficient explanations by causal attribution (greedy search);
            # veto dud explanations like 'Because it looked liked one'
            suff_expl = agent.symbolic.attribute(
                reg_gr_v, tgt_ev, evidence_atoms, competing_evts, vetos=[v_tgt_lit]
            )

            if suff_expl is not None:
                # Found some sufficient explanations; report the first one as the
                # answer using the template "Because {}, {} and {}."
                answer_logical_form = []
                answer_nl = "Because I thought "
                dem_refs = {}; dem_offset = len(answer_nl)

                for i, exps_lit in enumerate(suff_expl):
                    # For each explanans literal, add the string "this is a X"
                    conc_pred = exps_lit.name.strip("v_")
                    conc_type, conc_ind = conc_pred.split("_")
                    pred_name = agent.lt_mem.lexicon.d2s[(int(conc_ind), conc_type)][0][0]
                    # Split camelCased predicate name
                    splits = re.findall(
                        r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", pred_name
                    )
                    splits = [w[0].lower()+w[1:] for w in splits]
                    conc_nl = ' '.join(splits)

                    # Update cognitive state w.r.t. value assignment and word sense
                    ri = f"x{i}t{ti_new}c{ci_new}"
                    agent.symbolic.value_assignment[ri] = exps_lit.args[0][0]
                    tok_ind = (f"t{ti_new}", f"c{ci_new}", "rc", "0")
                    agent.symbolic.word_senses[tok_ind] = \
                        ((conc_type_to_pos[conc_type], pred_name), conc_pred)

                    answer_logical_form.append(
                        (pred_name, conc_type_to_pos[conc_type], (ri,), False)
                    )

                    # Realize the reason as natural language utterance
                    reason_prefix = f"this is a "

                    # Append a suffix appropriate for the number of reasons
                    if i == len(suff_expl)-1:
                        # Last entry, period
                        reason_suffix = "."
                    elif i == len(suff_expl)-2:
                        # Next to last entry, comma+and
                        reason_suffix = ", and "
                    else:
                        # Otherwise, comma
                        reason_suffix = ", "

                    reason_nl = f"{reason_prefix}{conc_nl}{reason_suffix}"
                    answer_nl += reason_nl

                    # Fetch mask for demonstrative reference and shift offset
                    dem_refs[(dem_offset, dem_offset+4)] = \
                        agent.lang.dialogue.referents["env"][-1][exps_lit.args[0][0]]["mask"]
                    dem_offset += len(reason_nl)

                # Wrapping logical form (consequent part, to be precise) as needed
                answer_logical_form = (tuple(answer_logical_form), ())

                # Push the translated answer to buffer of utterances to generate
                agent.lang.dialogue.to_generate.append(
                    ((answer_logical_form, None), answer_nl, dem_refs)
                )

                # Push another record for the rhetorical relation as well, namely the
                # fact that the provided answer clause explains the agent's previous
                # question. Not to be uttered explicitly (already uttered, as a matter
                # of fact), but for bookkeeping purpose.
                rrel_logical_form = (
                    "expl", "sp", (f"t{ti_new}c{ci_new}", q_cons[0].args[1][0]), False
                )
                rrel_logical_form = ((rrel_logical_form,), ())
                agent.lang.dialogue.to_generate.append(
                    (
                        (rrel_logical_form, None),
                        "# rhetorical relation due to 'because'",
                        {}
                    )
                )

            else:
                # No 'meaningful' (i.e., 'Because it looked like one') sufficient
                # explanations found, try finding a good counterfactual explanation
                # that would've led to the expected ground truth answer
                selected_cf_expl = None

                if agent.cfg.exp.strat_feedback != "maxHelpExpl2":
                    # Short circuit, just give a dud explanation
                    agent.lang.dialogue.to_generate.append(
                        # Will just pass None as "logical form" for this...
                        (None, "I cannot explain.", {})
                    )
                    continue

                # GT side of the distinguishing properties
                shared_templates_gt = {frozenset(tpl2) for _, tpl2 in shared_template_pairs}
                distinguishing_templates_gt = [
                    tpl for tpl in exps_templates_gt[1:]
                    if frozenset(tpl) not in shared_templates_gt
                ]
                # Not including the dud explanation ('I would've said that if it looked like one')
                # unlike above
                selected_templates = distinguishing_templates_gt

                # len(selected_templates) > 0 iff there's any KB rule that can be leveraged
                # to abduce the expected ground-truth answer
                gt_inferrable_from_kb = len(selected_templates) > 0

                if gt_inferrable_from_kb:
                    # Try to find some meaningful counterfactual explanation that would
                    # increase the likelihood of the expected answer to a sufficiently
                    # high value. Basically an existence test; try every instantiation
                    # of every template discovered until some valid counterfactual is
                    # found. If found one, answer with that; otherwise, fall back to
                    # dud explanation.

                    # Test each template one-by-one
                    for template in selected_templates:
                        # Find every possible instantiation of the template
                        exps_instances = _instantiate_templates([template], scene_ents)

                        # Considering visual evidence for class & attributes concepts,
                        # as above
                        evidence_atoms = {
                            Literal(f"v_{c_lit.name}", c_lit.args)
                            for conjunction in exps_instances for c_lit in conjunction
                            if c_lit.name.startswith("pcls")
                        }
                        evidence_atoms = {
                            evd_atm for evd_atm in evidence_atoms
                            if evd_atm in reg_gr_v.graph["atoms_map"]
                        }       # Can try replacement only if `evd_atm` is registered in graph

                        if len(evidence_atoms) > 0:
                            # Potential explanation exists in scene, albeit with a
                            # probability value not high enough. Try raising likelihood
                            # of relevant evidence atoms and querying the updated region
                            # graph for ground truth event probability.

                            # Obtain a modified graph where the likelihoods are raised to
                            # 'sufficiently high' values. (Note we are assuming evidence
                            # literals occur in rules with positive polarity only, which
                            # will suffice within our current scope.)
                            replacements = { evd_atm: HIGH for evd_atm in evidence_atoms }
                            backups = {
                                evd_atm: rgr_extract_likelihood(reg_gr_v, evd_atm)
                                for evd_atm in evidence_atoms
                            }           # For rolling back to original values
                            rgr_replace_likelihood(reg_gr_v, replacements)

                            # Query the updated graph for the event probabilities
                            max_prob_evt = (None, float("-inf"))
                            for atm in possible_answers:
                                evt = (atm,)
                                _, prob_scores = query(reg_gr_v, None, (evt, None), {})

                                # Update max probability event if applicable
                                evt_prob = [
                                    prob for prob, is_evt in prob_scores[()].values() if is_evt
                                ][0]
                                if evt_prob > max_prob_evt[1]:
                                    max_prob_evt = (evt, evt_prob)
                            rgr_replace_likelihood(reg_gr_v, backups)

                            assert max_prob_evt[0] is not None
                            if max_prob_evt[0] == (expected_gt,):
                                # The counterfactual case successfully subverts the ranking
                                # of answers, making the expected ground truth as the most
                                # likely event

                                # Provide the evidence atoms with current likelihood values
                                # as data needed for generating the counterfactual explanation.
                                selected_cf_expl = (backups, template)
                                break

                        else:
                            # Potential explanation doesn't exist in scene. Try adding
                            # hypothetical entities into the scene with appropriate
                            # likelihood values based on the info contained in template.
                            # Compile a new region graph then query for ground truth
                            # event probability.
                            occurring_vars = {
                                arg for t_lit in template for arg, is_var in t_lit.args
                                if is_var
                            }
                            hyp_ents = [f"h{i}" for i in range(len(occurring_vars))]
                            hyp_subs = {
                                (v, True): (e, False)
                                for v, e in zip(occurring_vars, hyp_ents)
                            }

                            # Template instance grounded with the hypothetical entities
                            hyp_instance = [
                                t_lit.substitute(terms=hyp_subs) for t_lit in template
                            ]

                            # Deepcopy vision.scene for counterfactual manipulation
                            scene_new = deepcopy(agent.vision.scene)
                            scene_new = {
                                **scene_new,
                                **{h: {} for h in hyp_ents}
                            }

                            # Add hypothetical likelihood values as designated by the
                            # template instance
                            for h_lit in hyp_instance:
                                conc_type, conc_ind = h_lit.name.split("_")
                                conc_ind = int(conc_ind)
                                field = f"pred_{conc_type}"
                                C = getattr(agent.vision.inventories, conc_type)

                                match conc_type:
                                    case "pcls":
                                        arg1 = h_lit.args[0][0]
                                        if field not in scene_new[arg1]:
                                            scene_new[arg1][field] = np.zeros(C)
                                        scene_new[arg1][field][conc_ind] = HIGH

                                    case "prel":
                                        assert len(h_lit.args) == 2

                                        arg1 = h_lit.args[0][0]; arg2 = h_lit.args[1][0]
                                        if field not in scene_new[arg1]:
                                            scene_new[arg1][field] = {}
                                        if arg2 not in scene_new[arg1][field]:
                                            scene_new[arg1][field][arg2] = np.zeros(C)
                                        scene_new[arg1][field][arg2][conc_ind] = HIGH
                                    
                                    case _:
                                        raise ValueError("Invalid concept type")

                            hyp_evidence = agent.lt_mem.kb.visual_evidence_from_scene(scene_new)
                            reg_gr_hyp = (kb_prog + hyp_evidence).compile()

                            # Query the hypothetical region graph for the event probabilities
                            max_prob_evt = (None, float("-inf"))
                            for atm in possible_answers:
                                evt = (atm,)
                                _, prob_scores = query(reg_gr_hyp, None, (evt, None), {})

                                # Update max probability event if applicable
                                evt_prob = [
                                    prob for prob, is_evt in prob_scores[()].values() if is_evt
                                ][0]
                                if evt_prob > max_prob_evt[1]:
                                    max_prob_evt = (evt, evt_prob)

                            assert max_prob_evt[0] is not None
                            if max_prob_evt[0] == (expected_gt,):
                                # The counterfactual case successfully subverts the ranking
                                # of answers, making the expected ground truth as the most
                                # likely event

                                # Provide the hypothetical evidence atoms (denoted with None
                                # as 'current' likelihood) as data needed for generating the
                                # counterfactual explanation.
                                evidence_likelihoods = {
                                    Literal(f"v_{h_lit.name}", h_lit.args): None
                                    for h_lit in hyp_instance
                                    if h_lit.name.startswith("pcls")
                                }
                                selected_cf_expl = (evidence_likelihoods, template)
                                break

                # Generate appropriate agent response
                if selected_cf_expl is None:
                    # Agent couldn't find any meaningful explanations that could be
                    # provided verbally; answer "I cannot explain."
                    agent.lang.dialogue.to_generate.append(
                        # Will just pass None as "logical form" for this...
                        (None, "I cannot explain.", {})
                    )

                else:
                    # Found data needed for generating some counterfactual explanation,
                    # in the form of dict { [potential_evidence]: [current_likelihood] },
                    # and the template that yielded the explanans instance
                    evidence_likelihoods, template = selected_cf_expl

                    answer_nl = "Because I thought "
                    dem_refs = {}; dem_offset = len(answer_nl)

                    reasons = {}
                    for evd_atom, pr_val in evidence_likelihoods.items():
                        # For each counterfactual explanation, add appropriate string
                        conc_pred = evd_atom.name.strip("v_")
                        conc_type, conc_ind = conc_pred.split("_")
                        pred_name = agent.lt_mem.lexicon.d2s[(int(conc_ind), conc_type)][0][0]
                        # Split camelCased predicate name
                        splits = re.findall(
                            r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", pred_name
                        )
                        splits = [w[0].lower()+w[1:] for w in splits]
                        conc_nl = ' '.join(splits)

                        if pr_val is not None and pr_val >= 0.5:
                            # Had right reason, but wasn't confident enough
                            reason_prefix = "this might not be a "

                            # Demonstrative "this" refers to the (potential) part
                            dem_ref = evd_atom.args[0][0]

                        else:
                            # Wasn't aware of any instance of the potential explanans
                            # template in the scene

                            # This will need a more principled treatment if we ever get
                            # to handle relations other than "have"
                            reason_prefix = "this doesn't have a "

                            # Demonstrative "this" refers to the whole object
                            dem_ref = tgt_lit.args[0][0]

                            # Avoid redundancy
                            if conc_nl in reasons: continue

                        reasons[conc_nl] = (reason_prefix, dem_ref)

                    for i, (conc_nl, (reason_prefix, dem_ref)) in enumerate(reasons.items()):
                        # Compose NL explanation utterance string

                        # Append a suffix appropriate for the number of reasons
                        if i == len(reasons)-1:
                            # Last entry, period
                            reason_suffix = "."
                        elif i == len(reasons)-2:
                            # Next to last entry, comma+and
                            reason_suffix = ", and "
                        else:
                            # Otherwise, comma
                            reason_suffix = ", "

                        reason_nl = f"{reason_prefix}{conc_nl}{reason_suffix}"
                        answer_nl += reason_nl

                        # Add demonstrative reference and shift offset
                        dem_refs[(dem_offset, dem_offset+4)] = \
                                agent.lang.dialogue.referents["env"][-1][dem_ref]["mask"]
                        dem_offset += len(reason_nl)

                    # Push the translated answer to buffer of utterances to generate; won't
                    # care for logical form, doesn't matter much now
                    agent.lang.dialogue.to_generate.append(
                        (None, answer_nl, dem_refs)
                    )

                    # Push another record for the rhetorical relation as well, namely the
                    # fact that the provided answer clause explains the agent's previous
                    # question. Not to be uttered explicitly (already uttered, as a matter
                    # of fact), but for bookkeeping purpose.
                    rrel_logical_form = (
                        "expl", "sp", (f"t{ti_new}c{ci_new}", q_cons[0].args[1][0]), False
                    )
                    rrel_logical_form = ((rrel_logical_form,), ())
                    agent.lang.dialogue.to_generate.append(
                        (
                            (rrel_logical_form, None),
                            "# rhetorical relation due to 'because'",
                            {}
                        )
                    )

    else:
        # Don't know how to handle other non-domain questions
        raise NotImplementedError

def _search_specs_from_kb(agent, question, restrictors, ref_reg_gr):
    """
    Helper method factored out for extracting specifications for visual search,
    based on the agent's current knowledge-base entries and some sensemaking
    result provided as a compiled region graph
    """
    q_vars, (cons, _) = question

    # Prepare taxonomy graph extracted from KB taxonomy entries; used later for
    # checking any concepts groupable by closest common supertypes
    taxonomy_graph = nx.DiGraph()
    for (r_cons, r_ante), _, _, knowledge_type in agent.lt_mem.kb.entries:
        if knowledge_type == "taxonomy":
            taxonomy_graph.add_edge(r_cons[0].name, r_ante[0].name)

    # Return value: Queries to feed into KB for fetching search specs. Represent
    # each query as a pair of predicates of interest & arg entities of interest
    kb_queries = set()

    # Inspecting literals in each q_rule for identifying search specs to feed into
    # visual search calls
    for q_lit in cons:
        if q_lit.name == "sp_isinstance":
            # Literal whose predicate is question-marked (contained for questions
            # like "What (kind of X) is this?", etc.); the first argument term,
            # standing for the predicate variable, must be contained in q_vars
            assert q_lit.args[0] in q_vars

            # Assume we are only interested in cls concepts with "What is this?"
            # type of questions
            kb_query_preds = frozenset([
                pred for pred in agent.lt_mem.kb.entries_by_pred 
                if pred.startswith("pcls")
            ])
            # Filter further by provided restrictors if applicable
            if q_lit.args[0][0] in restrictors:
                kb_query_preds = frozenset([
                    pred for pred in kb_query_preds
                    if pred in restrictors[q_lit.args[0][0]]
                ])
            kb_query_args = tuple(q_lit.args[1:])
        else:
            # Literal with fixed predicate, to which can narrow down the KB query
            kb_query_preds = frozenset([q_lit.name])
            kb_query_args = tuple(q_lit.args)
        
        kb_queries.add((kb_query_preds, kb_query_args))

    # Query the KB to collect search specs
    search_spec_cands = []
    for kb_query_preds, kb_query_args in kb_queries:

        for pred in kb_query_preds:
            # Relevant KB entries containing predicate of interest
            relevant_entries = agent.lt_mem.kb.entries_by_pred[pred]
            relevant_entries = [
                agent.lt_mem.kb.entries[entry_id]
                for entry_id in relevant_entries
                if agent.lt_mem.kb.entries[entry_id][3] != "taxonomy"
                    # Not using taxonomy knowledge as leverage for visual reasoning
            ]

            # Set of literals for each relevant KB entry
            relevant_literals = sum([
                flatten_cons_ante(*entry[0]) for entry in relevant_entries
            ], [])
            relevant_literals = [
                set(cons+ante) for cons, ante in relevant_literals
            ]
            # Depending on which literal (with matching predicate name) in literal
            # sets to use as 'anchor', there can be multiple choices of search specs
            relevant_literals = [
                { l: lits-{l} for l in lits if l.name==pred }
                for lits in relevant_literals
            ]

            # Collect search spec candidates. We will disregard 'adjectival' concepts as
            # search spec elements, noticing that it is usually sufficient and generalizable
            # to provide object class info only as specs for searching potentially relevant,
            # yet unrecognized entities in a scene. This is more of a heuristic for now --
            # maybe justify this on good grounds later...
            specs = [
                {
                    tgt_lit: (
                        # {rl for rl in rel_lits if not rl.name.startswith("att_")},
                        rel_lits,
                        {la: qa for la, qa in zip(tgt_lit.args, kb_query_args)}
                    )
                    for tgt_lit, rel_lits in lits.items()
                }
                for lits in relevant_literals
            ]
            specs = [
                {
                    tgt_lit.substitute(terms=term_map): frozenset({
                        rl.substitute(terms=term_map) for rl in rel_lits
                    })
                    for tgt_lit, (rel_lits, term_map) in spc.items()
                }
                for spc in specs
            ]
            search_spec_cands += specs

    # Merge and flatten down to a single layer dict
    def set_add_merge(d1, d2):
        for k, v in d2.items(): d1[k].add(v)
        return d1
    search_spec_cands = reduce(set_add_merge, [defaultdict(set)]+search_spec_cands)

    # Finalize set of search specs, excluding those which already have satisfying
    # entities in the current sensemaking output
    final_specs = []
    for lits_sets in search_spec_cands.values():
        for lits in lits_sets:
            # Lift any remaining function term args to non-function variable args
            all_fn_args = {
                arg for arg in set.union(*[set(l.args) for l in lits])
                if type(arg[0])==tuple
            }
            all_var_names = {
                t_val for t_val, t_is_var in set.union(*[l.nonfn_terms() for l in lits])
                if t_is_var
            }
            fn_lifting_map = {
                fa: (f"X{i+len(all_var_names)}", True)
                for i, fa in enumerate(all_fn_args)
            }

            search_vars = all_var_names | {vn for vn, _ in fn_lifting_map.values()}
            search_vars = tuple(search_vars)
            if len(search_vars) == 0:
                # Disregard if there's no variables in search spec (i.e. no search target
                # after all)
                continue

            lits = [l.substitute(terms=fn_lifting_map) for l in lits]
            lits = [l for l in lits if any(la_is_var for _, la_is_var in l.args)]

            # Disregard if there's already an isomorphic literal set
            has_isomorphic_spec = any(
                Literal.entailing_mapping_btw(lits, spc[1])[0] == 0
                for spc in final_specs
            )
            if has_isomorphic_spec:
                continue

            final_specs.append((search_vars, lits, {}))

    # See if any of the search specs can be grouped and combined by closest common
    # supertypes; continue grouping pairs of specs with matching signatures and
    # shared supertypes as much as possible
    grouping_finished = False; disj_index = 0
    while not grouping_finished:
        for si, sj in combinations(range(len(final_specs)), 2):
            # If not isomorphic after replacing all cls predicates with the same
            # dummy predicate, the pair is not groupable
            is_cls_si = [
                lit.name.startswith("pcls_") or lit.name.startswith("disj_")
                for lit in final_specs[si][1]
            ]
            is_cls_sj = [
                lit.name.startswith("pcls_")  or lit.name.startswith("disj_")
                for lit in final_specs[sj][1]
            ]
            spec_subs_si = [
                lit.substitute(preds={ lit.name: "pcls_dummy" }) if is_cls else lit
                for lit, is_cls in zip(final_specs[si][1], is_cls_si)
            ]
            spec_subs_sj = [
                lit.substitute(preds={ lit.name: "pcls_dummy" }) if is_cls else lit
                for lit, is_cls in zip(final_specs[sj][1], is_cls_sj)
            ]
            entail_dir, mapping = Literal.entailing_mapping_btw(spec_subs_si, spec_subs_sj)
            if entail_dir != 0: continue

            # If there is no one-to-one correspondence between the cls predicates
            # such that the predicates in each pair belong to the same taxonomy
            # tree, the pair is not groupable
            assert sum(is_cls_si) == sum(is_cls_sj)
            cls_inds_si = [i for i, is_cls in enumerate(is_cls_si) if is_cls]
            cls_inds_sj = [i for i, is_cls in enumerate(is_cls_sj) if is_cls]

            valid_bijections = []
            for prm in permutations(cls_inds_sj):
                # Obtain bijection between cls literals in the first spec and the second
                bijection = [(cls_inds_si[i], i_sj) for i, i_sj in enumerate(prm)]
                matched_lits = [
                    (final_specs[si][1][i_si], final_specs[sj][1][i_sj])
                    for i_si, i_sj in bijection
                ]
                matched_lits = [
                    (lit_si, lit_sj.substitute(**mapping))
                    for lit_si, lit_sj in matched_lits
                ]

                # Reject bijection if any of the pairs do not have matching args
                if any(lit_si.args!=lit_sj.args for lit_si, lit_sj in matched_lits):
                    continue

                # Find closest common supertypes for each matched pair in bijection
                grouping_supertypes = []
                for lit_si, lit_sj in matched_lits:
                    # Predicate names won't be same; duplicate isomorphic specs are
                    # filtered out above
                    assert lit_si.name != lit_sj.name

                    cls_conc_si = lit_si.name if lit_si.name.startswith("pcls_") \
                        else final_specs[si][2][lit_si.name][0]
                    cls_conc_sj = lit_sj.name if lit_sj.name.startswith("pcls_") \
                        else final_specs[sj][2][lit_sj.name][0]
                    closest_common_supertype = nx.lowest_common_ancestor(
                        taxonomy_graph, cls_conc_si, cls_conc_sj
                    )

                    if closest_common_supertype is None:
                        # Not in the same taxonomy tree, cannot group
                        break
                    else:
                        # Common supertype identified, record relevant info
                        elem_concs_si = {lit_si.name} if lit_si.name.startswith("pcls_") \
                            else final_specs[si][2][lit_si.name][1]
                        elem_concs_sj = {lit_sj.name} if lit_sj.name.startswith("pcls_") \
                            else final_specs[sj][2][lit_sj.name][1]
                        grouping_supertypes.append(
                            (closest_common_supertype, elem_concs_si | elem_concs_sj)
                        )

                # Add to list of valid bijections if supertypes successfully identified
                # for all matched pairs
                if len(grouping_supertypes) == len(bijection):
                    valid_bijections.append((bijection, grouping_supertypes))

            if len(valid_bijections) == 0: continue

            # If reached here, update the spec list accordingly by replacing the two
            # specs being processed with new grouped specs (possibly multiple, in
            # principle -- though we won't see such cases in our scope)
            grouped_specs = []
            for bijection, grouping_supertypes in valid_bijections:
                # (Arbitrarily) Select the first spec to be the 'base' of the new grouped
                # spec to be appended
                search_vars, lits, _ = final_specs[si]

                # Dict describing which set of elementary concepts is referred to by
                # each disjunction predicate
                pred_glossary = {}

                # Bijection info reshaped for easier processing
                bijection = {
                    i_si: (i_sj, gr_info)
                    for (i_si, i_sj), gr_info in zip(bijection, grouping_supertypes)
                }

                # Prepare new literal set for search spec description, starting from
                # the base and appropriately replacing the predicate names
                lits_new = []
                for i_si, lit in enumerate(lits):
                    if i_si in bijection:
                        # Need to be processed before being added to the literal set
                        i_sj, grouping_info = bijection[i_si]

                        lit_si = final_specs[si][1][i_si]
                        lit_sj = final_specs[sj][1][i_sj]
                        is_elem_si = lit_si.name.startswith("pcls_")
                        is_elem_sj = lit_sj.name.startswith("pcls_")

                        if is_elem_si and is_elem_sj:
                            # Case 1: Both predicates elementary concepts, need to
                            # introduce a fresh disjunction predicate
                            disj_name = f"disj_{disj_index}"
                            disj_index += 1
                        elif is_elem_si and not is_elem_sj:
                            # Case 2: First predicate refers to elementary concept, while
                            # second refers to a disjunction; 'absorb' former to latter
                            disj_name = lit_sj.name
                        elif not is_elem_si and is_elem_sj:
                            # Symmetric case of the above Case 2, treat similarly
                            disj_name = lit_si.name
                        else:
                            # Case 3: Both predicates refer to disjunctions; (arbitrarily)
                            # select the first to absorb the second
                            disj_name = lit_si.name

                        pred_glossary[disj_name] = grouping_info
                        lits_new.append(Literal(disj_name, lit.args))

                    else:
                        # Nothing to do, add as-is
                        lits_new.append(lit)

                grouped_specs.append((search_vars, lits_new, pred_glossary))

            # Don't forget to take the remaining specs not being processed
            final_specs = [
                spc for i, spc in enumerate(final_specs) if i not in (si, sj)
            ]
            final_specs += grouped_specs

            # Then break to find any possible grouping, from the top with the updated list
            break

        else:
            # No more groupable spec pairs; terminate while loop
            grouping_finished = True

    # # Check if the agent is already (visually) aware of the potential search
    # # targets; if so, disregard this one
    # check_result, _ = agent.symbolic.query(
    #     ref_reg_gr, tuple((v, False) for v in search_vars), (lits, None)
    # )
    # if len(check_result) > 0:
    #     continue

    return final_specs


def _find_explanans_templates(tgt_lit, kb_prog_analyzed):
    """
    Helper method factored out for finding templates of potential explanantia
    (causal chains possibly not fully grounded, which could raise the possibility
    of the target explanandum when appropriately grounded), based on the inference
    program exported from some specific version of (exported) KB.

    Start from rules containing the target event predicate and continue spanning
    along (against?) the abductive direction until all potential evidence atoms
    are identified.
    """
    exps_templates = [[tgt_lit]]; frontier = {tgt_lit}
    while len(frontier) > 0:
        expd_atm = frontier.pop()       # Atom representing explanandum event

        for rule_info in kb_prog_analyzed.values():
            # Disregard rules without abductive force
            if not rule_info.get("abductive"): continue

            # Check if rule is relevant; i.e. whether the popped explanandum
            # event is included in rule antecedent
            entail_dir, mapping = Literal.entailing_mapping_btw(
                rule_info["ante"], [expd_atm]
            )

            if entail_dir is not None and entail_dir >= 0:
                # Rule relevant, target event may be abduced from consequent
                r_cons_subs = [
                    c_lit.substitute(**mapping) for c_lit in rule_info["cons"]
                ]

                # Add the whole substituted consequent to the list of valid
                # explanans templates
                exps_templates.append(r_cons_subs)

                # Add each literal in the substituted consequent to the search
                # frontier
                frontier |= set(r_cons_subs)

    return exps_templates

def _instantiate_templates(exps_templates, scene_ents):
    """
    Helper method factored out for enumerating every possible instantiations of
    the provided explanans templates with scene entities
    """
    exps_instances = []

    for conjunction in exps_templates:
        # Variables and constants occurring in the template conjunction
        occurring_consts = {
            arg for c_lit in conjunction for arg, is_var in c_lit.args
            if not is_var
        }
        occurring_vars = {
            arg for c_lit in conjunction for arg, is_var in c_lit.args
            if is_var
        }
        occurring_vars = tuple(occurring_vars)      # Int indexing
        remaining_ents = scene_ents - occurring_consts

        # All possible substitutions for the remaining variables; permutations()
        # will give empty list if len(occurring_vars) is larger than len(remaining_ents)
        possible_remaining_subs = permutations(remaining_ents, len(occurring_vars))
        for subs in possible_remaining_subs:
            subs = { (occurring_vars[i], True): (e, False) for i, e in enumerate(subs) }
            instance = [c_lit.substitute(terms=subs) for c_lit in conjunction]
            exps_instances.append(instance)

    return exps_instances
