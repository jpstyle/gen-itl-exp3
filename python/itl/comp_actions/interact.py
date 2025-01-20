"""
Implements composite agent interactions, environmental or dialogue, that need
coordination of different component modules
"""
import os
import re
import logging
from itertools import combinations, product, groupby
from collections import defaultdict, deque, Counter

import copy
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

    if "sp_neologism" in predicates_mentioned:
        # Question cannot be answered for spme relevant clause(s) include
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
        assert intended_ante is None and len(intended_cons) > 0
        join_conc = agent.lt_mem.lexicon.s2d[("va", "join")][0][1]
        assert {lit.name for lit in intended_cons} == {f"arel_{join_conc}"}

        # Answer by elaborate original intention of joining which part
        # instances, along with their types as recognized
        get_conc = lambda s: agent.lt_mem.lexicon.s2d[("va", s)][0][1]
        pick_up_actions = [get_conc("pick_up_left"), get_conc("pick_up_right")]
        assemble_actions = [
            get_conc("assemble_right_to_left"), get_conc("assemble_left_to_right")
        ]
        last_action_type, last_action_params = exec_state["action_history"][-1]
        if last_action_type in pick_up_actions:
            # The previous action undone was a picking up which resulted in
            # holding a pair of subassemblies that can never be joined. In this
            # case, the intended join is the next one in the planner agenda.
            intended_join = next(
                todo_args[1][2:4] for todo_type, todo_args in agent.planner.agenda
                if todo_type == "execute_command" and todo_args[0] in assemble_actions
            )
        else:
            # The previous action undone was an assembly between a (valid) pair
            # of subassemblies, yet at an incorrect pose. In this case, the
            # intended join is the undone assembly action.
            assert last_action_type in assemble_actions
            intended_join = last_action_params[2:4]

        # Getting appropriate name handles, concepts and NL symbols
        ent_left = agent.vision.scene[intended_join[0]]["env_handle"]
        ent_right = agent.vision.scene[intended_join[1]]["env_handle"]
        den_left = exec_state["recognitions"][intended_join[0]]
        den_right = exec_state["recognitions"][intended_join[1]]
        sym_left = agent.lt_mem.lexicon.d2s[("pcls", den_left)][0][1]
        sym_right = agent.lt_mem.lexicon.d2s[("pcls", den_right)][0][1]

        # Entity paths in environment, obtained differently per previous
        # action type
        if last_action_type in pick_up_actions:
            if pick_up_actions.index(last_action_type) == 0:
                # Entity was picked up with left hand in last action, which
                # is currently dropped back on table
                ent_path_left = f"/*/{ent_left}"
                ent_path_right = f"/Student Agent/Right Hand/*/{ent_right}"
            else:
                # Entity was picked up with right hand in last action, which
                # is currently dropped back on table
                ent_path_left = f"/Student Agent/Left Hand/*/{ent_left}"
                ent_path_right = f"/*/{ent_right}"
        else:
            ent_path_left = f"/Student Agent/Left Hand/*/{ent_left}"
            ent_path_right = f"/Student Agent/Right Hand/*/{ent_right}"

        # NL surface forms and corresponding logical forms
        surface_form_0 = f"I was trying to join this {sym_left} and this {sym_right}."
        surface_form_1 = f"# to-infinitive phrase ('to join this {sym_left} and this {sym_right}')"
        gq_0 = gq_1 = None; bvars_0 = bvars_1 = set(); ante_0 = ante_1 = []
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
            ("n", sym_left, ["x1"]),
            ("n", sym_right, ["x2"])
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
            "x0": { "entity": None, "rf_info": {} },
            "x1": { "entity": intended_join[0], "rf_info": {} },
            "x2": { "entity": intended_join[1], "rf_info": {} }
        }
        predicates_0 = {
            "pc0": (("sp", "intend"), "sp_intend"),
            "pc1": (("sp", "pronoun1"), "sp_pronoun1")
        }
        predicates_1 = {
            "pc0": (("va", "join"), f"arel_{join_conc}"),
            "pc1": (("n", sym_left), f"pcls_{den_left}"),
            "pc2": (("n", sym_right), f"pcls_{den_right}")
        }

        # Point to each part involved in the originally intended join
        offset_left = surface_form_0.find(sym_left)
        offset_right = surface_form_0.find(sym_right)
        dem_refs_0 = {
            (offset_left, offset_left+len(sym_left)): (ent_path_left, False),
            (offset_right, offset_right+len(sym_right)): (ent_path_right, False)
                # Note: False values in the tuples specify that the demonstrative
                # references are conveyed via string name handles instead of masks
        }

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

        # Referents & predicates info
        referents = {
            "e": { "mood": "." },       # Indicative
            "x0": { "entity": "_self", "rf_info": {} }
        }
        predicates = { "pc0": (("sp", "unable"), "sp_unable") }

        agent.lang.dialogue.unexecuted_commands.remove(utt_pointer)
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

    # Check if (the remainder) of the plan being executed need to be examined,
    # to see whether if it is still valid after agent's knowledg update, if any.
    xb_updated, kb_updated = exec_state.get("knowledge_updated", (False, False))
    exec_state["knowledge_updated"] = (False, False)    # Clear flags

    # Note to self: One could imagine we could skip re-planning from scratch
    # after the user's labeling feedback if it did not involve an instance
    # that is already referenced in the remainder of the plan (i.e., when the
    # corrective feedback does not refute the premise based on agent's previous
    # object recognition output)... However, it turned out it caused too much
    # headache to juice such opportunities, and it's much easier to simply
    # re-plan after any form of knowledge update (after dropping all held
    # objects in hands).

    if xb_updated or kb_updated:
        # Plan again for the remainder of the assembly based on current execution
        # state, belief and knowledge, then return without return value
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
        labeling_map = dict(labeling_feedback)

        # Update execution state to reflect recognitions hitherto committed
        # and those certified by user feedback
        exec_state["recognitions"] = {
            n: exec_state["recognitions"][n]
            for gr in exec_state["connection_graphs"].values()
            for n in gr
        } | labeling_map                           # Update with feedback

        # Finally, re-plan
        action_sequence, recognitions = _plan_assembly(agent, exec_state["plan_goal"][1])

        # Record how the agent decided to recognize each (non-)object
        exec_state["recognitions"] = recognitions

        # Enqueue appropriate agenda items and finish
        agent.planner.agenda = deque(
            ("execute_command", action_step) for action_step in action_sequence
        )       # Whatever steps remaining, replace
        agent.planner.agenda.append(("utter_simple", ("Done.", { "mood": "." })))

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
        "connection_graphs": {},
        "recognitions": {},
        "join_validities": (set(), set()),
        "knowledge_updated": (False, False),
        "action_history": []
    }

    # Plan towards building valid target structure
    action_sequence, recognitions = _plan_assembly(agent, build_target)

    # Record how the agent decided to recognize each (non-)object
    agent.planner.execution_state["recognitions"] = recognitions

    # Enqueue appropriate agenda items and finish
    agent.planner.agenda = deque(
        ("execute_command", action_step) for action_step in action_sequence
    )
    agent.planner.agenda.append(("utter_simple", ("Done.", { "mood": "." })))

def _plan_assembly(agent, build_target):
    """
    Helper method factored out for planning towards the designated target
    structure. May be used for replanning after user's belief/knowledge status
    changed, picking up from current assembly progress state recorded.
    """
    exec_state = agent.planner.execution_state      # Shortcut var

    # Fetching action concept indices
    get_conc = lambda s: agent.lt_mem.lexicon.s2d[("va", s)][0][1]
    pick_up_actions = [get_conc("pick_up_left"), get_conc("pick_up_right")]
    drop_actions = [get_conc("drop_left"), get_conc("drop_right")]
    assemble_actions = [
        get_conc("assemble_right_to_left"), get_conc("assemble_left_to_right")
    ]

    # Convert current visual scene (from ensemble prediction) into ASP fact
    # literals (note difference vs. agent.lt_mem.kb.visual_evidence_from_scene
    # method, which is for probabilistic reasoning by LP^MLN)
    threshold = 0.25; dsc_bin = 20
    observations = set()
    likelihood_values = np.stack([
        obj["pred_cls"] for obj in agent.vision.scene.values()
    ])
    min_val = max(likelihood_values.min(), threshold)
    max_val = likelihood_values.max()
    val_range = max_val - min_val

    for oi, obj in agent.vision.scene.items():
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
                    case "atomic":
                        options = data["parts"]
                        for opt_conc in options:
                            part_lit = Literal("atomic", wrap_args(opt_conc))
                            assembly_pieces.add(part_lit)
                    case "sa":
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
            ext_conc = exec_state["recognitions"][ext_node]
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
    node_unifications = {}
    rename_node = lambda n: f"n_{n.number}" if n.type == SymbolType.Number \
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

            case "unify_node":
                # Unification between objects already used as subassembly
                # component in environment vs. matching nodes in hierarchy
                ex_obj = atm.arguments[0].name
                node_rn = rename_node(atm.arguments[1])
                node_unifications[ex_obj] = node_rn

    # Top node
    assembly_hierarchy.add_node("n_0", node_type="sa")
    # Note: This assembly hierarchy structure object also serves to provide
    # explanation which part is required for building which subassembly
    # (that is direct parent in the hierarchy)

    # Pool of available instances for each part type, as recognized by agent
    atomic_node_concs = {
        n: data["choice_conc"]
        for n, data in assembly_hierarchy.nodes(data=True)
        if data["node_type"] == "atomic"
    }
    atomic_nodes_by_conc = sorted(
        assembly_hierarchy.nodes(data=True), key=lambda x: x[1]["choice_conc"]
    )
    atomic_nodes_by_conc = [
        (data["choice_conc"], n)
        for n, data in atomic_nodes_by_conc if data["node_type"] == "atomic"
    ]
    atomic_nodes_by_conc = {
        k: set(n for _, n in v)
        for k, v in groupby(atomic_nodes_by_conc, key=lambda x: x[0])
    }

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

    # Turns out if a structure consists of too many atomic parts (say, more
    # than 6 or so), ASP planner performance is significantly affected. We
    # handle this by breaking the planning problem down to multiple smaller
    # ones... In principle, this may risk planning failure when physical
    # collision check is involved, depending on how the target structure is
    # partitioned. Note that agents that are aware of semantically valid
    # substructures are entirely free from such concerns.
    connection_graph = nx.Graph()
    for u, v in connect_edges: connection_graph.add_edge(u, v)

    # Initial planning problem state, obtain once at the beginning and deepcopy
    # for manipulation at the very beginning or after reaching deadend. Reflects
    # current progress actually made in environment so far.
    def initial_planning_state():
        # Graph representation of how selected chunks are compressed into
        # subassembly nodes
        compression_graph = connection_graph.copy()
        # Chunks assembled so far, with their component objects
        chunks_assembled = {}
        # Chunking sequence made throughout the piecewise planning process
        current_chunk_sequence = []
        # Next integer index to assign to new subassembly; ensure the newly
        # assigned index doesn't overlap with any of the previously assigned
        # ones by looking through action history
        sa_ind = max([
            int(re.findall(r"[si](\d+)(_\d+)?", action_params[-1])[0][0])
            for action_type, action_params in exec_state["action_history"]
            if action_type in assemble_actions
        ] + [-1]) + 1           # [-1] ensures max value is always obtained
        # Joining action sequence, which abstracts from pick-up & drop steps
        # and choices of left/right manipulators
        join_sequence = []
        # For counting remaining available instances per type
        inst_pool = copy.deepcopy(atomic_nodes_by_conc)

        # Account for any progress made so far
        for sa, sa_graph in exec_state["connection_graphs"].items():
            if len(sa_graph) == 1: continue     # No processing for singleton parts
            chunk_nodes = {
                node_unifications[f"{sa}_{ex_obj}"] for ex_obj in sa_graph.nodes
            }
            compress_chunk(compression_graph, chunk_nodes, sa)
            chunks_assembled[sa] = chunk_nodes
            for n in chunk_nodes:
                conc_n = atomic_node_concs[n]
                if n in inst_pool[conc_n]: inst_pool[conc_n].remove(n)

        initial_state = (
            compression_graph, chunks_assembled, current_chunk_sequence,
            sa_ind, join_sequence, inst_pool
        )
        return initial_state

    # Helper method for updating compression graph by replacing the collection
    # of nodes with a new node representing the subassembly newly assembled
    # from the specified
    def compress_chunk(compression_graph, chunk, name):
        compression_graph.add_node(name)
        for u, v in list(compression_graph.edges):
            if u in chunk:
                if u in compression_graph: compression_graph.remove_node(u)
                if v not in chunk: compression_graph.add_edge(name, v)
            if v in chunk:
                if v in compression_graph: compression_graph.remove_node(v)
                if u not in chunk: compression_graph.add_edge(u, name)

    # Compress the connection graph part by part, randomly selecting chunks
    # of parts or subassemblies to form smaller planning problems
    min_chunk_size = 5
    initial_state = initial_planning_state()
    initial_state_cp = copy.deepcopy(initial_state)
    compression_graph, chunks_assembled = initial_state_cp[:2]
    current_chunk_sequence, sa_ind = initial_state_cp[2:4]
    join_sequence, inst_pool = initial_state_cp[4:]
    # Remembering sequences of chunking that proved to reach deadend
    invalid_chunk_sequences = []
    # Record every 'assembly scope implication' such that if a set of certain
    # atomic objects are to be included in a chunk, some other object must
    # be included in the set as well, so as to avoid deadends due to violations
    # of 'precedence constraints'.
    scope_implications = { n: set() for n in connection_graph }
    # Remembering valid & invalid joins of subassemblies, so as to avoid
    # calling oracle (be it cheat sheet or external motion planner)
    valid_joins_cached = exec_state["join_validities"][0]
    invalid_joins_cached = exec_state["join_validities"][1]
    # Tracking how many piecewise planning attempts and oracle calls were made
    planning_attempts = 0; query_count = 0
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
                            n for s in chunk if s in chunks_assembled
                            for n in chunks_assembled[s]
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

            conc_counts = Counter(
                sorted(atomic_node_concs.get(u, -1) for u in chunk)
            )
            shortage_count = 0
            for conc, count in conc_counts.items():
                # conc of -1 stands for non-atomic subassembly
                if conc == -1: continue
                if count > len(inst_pool[conc]):
                    shortage_count += abs(count > len(inst_pool[conc]))
            chunk_candidates.append((chunk, shortage_count))

        chunk_candidates = sorted(chunk_candidates, key=lambda x: x[1])
        for chunk, _ in chunk_candidates:
            # Select chunk with smallest part instance shortage possible,
            # while ensuring that the selection doesn't result in an
            # invalid chunk sequence (remembered so far)
            potential_sequence = current_chunk_sequence + [chunk]
            if potential_sequence not in invalid_chunk_sequences:
                chunk_selected = chunk
                break
        else:
            # All possible options would lead to invalid sequence. Remember
            # the current sequence as invalid.
            # Update tracker states
            invalid_chunk_sequences.append(current_chunk_sequence)
            planning_attempts += 1
            valid_joins_cached |= collision_checker.valid_joins
            invalid_joins_cached |= collision_checker.invalid_joins
            # Then start again
            initial_state_cp = copy.deepcopy(initial_state)
            compression_graph, chunks_assembled = initial_state_cp[:2]
            current_chunk_sequence, sa_ind = initial_state_cp[2:4]
            join_sequence, inst_pool = initial_state_cp[4:]
            continue

        chunk_subgraph = compression_graph.subgraph(chunk_selected)

        # Remove instances in the selected chunk from instance pool
        for n in chunk_selected:
            if n not in atomic_node_concs: continue     # Non-atomic
            conc_n = atomic_node_concs[n]
            if n in inst_pool[conc_n]: inst_pool[conc_n].remove(n)

        # Load ASP encoding for planning sequence of atomic actions
        collision_checker = _PhysicalAssemblyPropagator(
            (connection_graph, connect_edges),
            dict(assembly_hierarchy.nodes(data=True)),
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
            if step > len(chunk_selected):
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
        for insts1, insts2 in collision_checker.invalid_joins:
            nodes1 = frozenset(collision_checker.inst2node[i] for i in insts1)
            nodes2 = frozenset(collision_checker.inst2node[i] for i in insts2)

            # Process each direction only if the 'consequent' side
            # of the implication has one object in the set
            if len(nodes1) == 1:
                for n in nodes2:
                    scope_implications[n].add((nodes2, list(nodes1)[0]))
            if len(nodes2) == 1:
                for n in nodes1:
                    scope_implications[n].add((nodes1, list(nodes2)[0]))

        # Tracking sequence of selected chunks, so as to remember invalid
        # sequences that led to deadend states and prevent following the
        # exact same sequence again
        current_chunk_sequence.append(chunk_selected)

        query_count += collision_checker.query_count
        if optimal_model is None:
            # Planning failure, start again from scratch
            # Update tracker states
            invalid_chunk_sequences.append(current_chunk_sequence)
            planning_attempts += 1
            valid_joins_cached |= collision_checker.valid_joins
            invalid_joins_cached |= collision_checker.invalid_joins
            # Then start again
            initial_state_cp = copy.deepcopy(initial_state)
            compression_graph, chunks_assembled = initial_state_cp[:2]
            current_chunk_sequence, sa_ind = initial_state_cp[2:4]
            join_sequence, inst_pool = initial_state_cp[4:]
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
            else f"i{sa_ind}_{a.number}"
        join_sequence += [
            (
                arg_val(s1), arg_val(s2),   # Involved subassemblies
                arg_val(n1), arg_val(n2),   # Involved nodes
                f"s{sa_ind}" if i==len(projection)-1 \
                    else f"i{sa_ind}_{i}"   # Naming resultant subassembly
            )
            for i, (s1, s2, n1, n2) in enumerate(projection)
        ]

        # Update compression graph accordingly
        compress_chunk(compression_graph, chunk_selected, f"s{sa_ind}")

        # Remember which components constitute the new subassembly, while
        # removing old subassemblies in chunk that have been merged into
        # new subassembly
        chunks_assembled[f"s{sa_ind}"] = {
            n for s in chunk_selected if s in chunks_assembled
            for n in chunks_assembled[s]
        } | {
            n for n in chunk_selected if n not in chunks_assembled
        }
        for s in chunk_selected:
            if s in chunks_assembled:
                del chunks_assembled[s]

        sa_ind += 1

    planning_attempts += 1
    log_msg = f"Working plan found after {planning_attempts} (re)planning attempts "
    log_msg += f"({query_count} calls total)"
    logger.info(log_msg)

    # Instantiate the high-level plan into an actual action sequence, with
    # all the necessary information fleshed out (to be passed to the Unity
    # environment as parameters)
    action_sequence = []

    # Stipulate the unified node-to-object mappings
    node2obj_map = {}
    for sa, sa_graph in exec_state["connection_graphs"].items():
        for obj in sa_graph.nodes:
            assert f"{sa}_{obj}" in node_unifications
            unified_node = node_unifications[f"{sa}_{obj}"]
            conc_n = atomic_node_concs[unified_node]
            node2obj_map[unified_node] = (obj, conc_n)
            if obj in part_candidates[conc_n]:
                part_candidates[conc_n].remove(obj)

    # Drop any currently held objects for headache-free re-planning
    for side in [0, 1]:
        if exec_state["manipulator_states"][side][0] is not None:
            action_sequence.append((drop_actions[side], None))

    nonobj_ind = 0; hands = [None, None]
    for s1, s2, n1, n2, res in join_sequence:
        # Convert assembly hierarchy nodes into randomly sampled objects with
        # matching part type, on first encounter. If ran out of objects in
        # pool, start assigning non-object indices.
        for n in [n1, n2]:
            if n in atomic_node_concs and n not in node2obj_map:
                conc_n = atomic_node_concs[n]
                if len(part_candidates[conc_n]) > 0:
                    o = part_candidates[conc_n].pop()
                else:
                    o = f"n{nonobj_ind}"
                    nonobj_ind += 1
                node2obj_map[n] = (o, conc_n)

        # Fetch matching (non-)object and unify with subassembly names if
        # necessary
        o1 = node2obj_map[n1][0]; o2 = node2obj_map[n2][0]
        if s1 == n1: s1 = o1
        if s2 == n2: s2 = o2

        if s1 in hands or s2 in hands:
            # Immediately using the subassembly built in the previous join
            empty_side = hands.index(None)
            occupied_side = list({0, 1} - {empty_side})[0]

            # Pick up the other involved subassembly with the empty hand
            pick_up_target = s2 if hands[occupied_side] == s1 else s1
            action_sequence.append((
                pick_up_actions[empty_side], (pick_up_target,)
            ))
        else:
            if any(held is not None for held in hands):
                # One hand occupied, but held object needs to be dropped as
                # it is not used in this step
                empty_side = hands.index(None)
                occupied_side = list({0, 1} - {empty_side})[0]

                action_sequence.append((drop_actions[occupied_side], None))

            # Boths hand are (now) empty; pick up s1 and s2 on the left and
            # the right side each
            action_sequence.append((pick_up_actions[0], (s1,)))
            action_sequence.append((pick_up_actions[1], (s2,)))

        # Determine joining direction by comparing the degrees of the involved
        # part nodes: smaller to larger. If the degrees are equal, break tie
        # by comparing average neighbor degree.
        deg_handle = lambda n: (
            connection_graph.degree[n],
            nx.average_neighbor_degree(connection_graph)[n]
        )
        if deg_handle(n1) >= deg_handle(n2):
            # n1 is more 'central', n2 is more 'peripheral'; join n2 to n1
            if s1 in hands:
                direction = 1 if hands.index(None) == 0 else 0
            elif s2 in hands:
                direction = 0 if hands.index(None) == 0 else 1
            else:
                direction = 0
        else:
            # n2 is more 'central', n1 is more 'peripheral'; join n2 to n1
            if s1 in hands:
                direction = 0 if hands.index(None) == 0 else 1
            elif s2 in hands:
                direction = 1 if hands.index(None) == 0 else 0
            else:
                direction = 1

        # Join s1 and s2 at appropriate contact point
        cp_n1, cp_n2 = connect_edges[(n1, n2)]
        cp_n1 = re.findall(r"p_(.*)_(.*)$", cp_n1)
        cp_n2 = re.findall(r"p_(.*)_(.*)$", cp_n2)
        cp_n1 = (int(cp_n1[0][0]), int(cp_n1[0][1]))
        cp_n2 = (int(cp_n2[0][0]), int(cp_n2[0][1]))
        if s1 in hands:
            if hands.index(None) == 0:
                # s1 held in right hand
                join_params = (s2, s1, o2, o1, cp_n2, cp_n1, res)
            else:
                # s1 held in left hand
                join_params = (s1, s2, o1, o2, cp_n1, cp_n2, res)
        elif s2 in hands:
            if hands.index(None) == 0:
                # s2 held in right hand
                join_params = (s1, s2, o1, o2, cp_n1, cp_n2, res)
            else:
                # s2 held in left hand
                join_params = (s2, s1, o2, o1, cp_n2, cp_n1, res)
        else:
            # Use join parameter order as-is
            join_params = (s1, s2, o1, o2, cp_n1, cp_n2, res)
        action_sequence.append((assemble_actions[direction], join_params))

        # Update hands status
        hands = [None, None]
        hands[direction] = res

    # Drop the finished product on the table
    empty_side = hands.index(None)
    occupied_side = list({0, 1} - {empty_side})[0]
    action_sequence.append((drop_actions[occupied_side], None))

    # Agent's 'decisions' as to how each object is classified as instances
    # of (known) visual concepts. Merge given assumptions and newly made
    # recognition decisions according to the generated plan.
    recognitions = exec_state["recognitions"] | {
        oi: conc for oi, conc in node2obj_map.values()
    }

    return action_sequence, recognitions

class _PhysicalAssemblyPropagator:
    def __init__(self, connections, node_data, cheat_sheet, test_result_cached):
        connection_graph, contacts = connections
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

        # (Injective) Mapping of individual nodes to corresponding atomic part
        # instances in the collision table
        atomic_nodes = {
            n: data["choice_conc"] for n, data in node_data.items()
            if data["node_type"] == "atomic"
        }
        self.node2inst = {}; self.inst2node = {}
        while len(self.node2inst) != len(atomic_nodes):
            for n in atomic_nodes:
                # If already mapped
                if n in self.node2inst: continue

                # See if uniquely determined by part concept alone
                conc_n = atomic_nodes[n]
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

                    cp_u = re.findall(r"p_(.*)_(.*)$", cp_u)
                    cp_v = re.findall(r"p_(.*)_(.*)$", cp_v)
                    cp_u = (int(cp_u[0][0]), int(cp_u[0][1]))
                    cp_v = (int(cp_v[0][0]), int(cp_v[0][1]))
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
                n1 = arg_val(arg1.arguments[2])
                n2 = arg_val(arg1.arguments[3])

                self.join_actions_a2l[(s1, s2, n1, n2, t)] = lit
                self.join_actions_l2a[lit].add((s1, s2, n1, n2, t))

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
                for s1, s2, n1, n2, t in self.join_actions_l2a[lit]:
                    status["joins"][t] = (s1, s2, n1, n2)
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
        for t, (s1, s2, n1, n2) in status["joins"].items():
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
            for cached_s1, cached_s2 in self.valid_joins:
                if insts_s1 <= cached_s1 and insts_s2 <= cached_s2:
                    verified_by_cache = True
                    break
                if insts_s1 <= cached_s2 and insts_s2 <= cached_s1:
                    verified_by_cache = True
                    break
            else:
                verified_by_cache = False

            if verified_by_cache:
                join_possible = True
            else:
                subset_cached_as_invalid = any(
                    (insts_s1 >= s1 and insts_s2 >= s2) or \
                        (insts_s1 >= s2 and insts_s2 >= s1)
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
                        self.collision_table.get((i_s1, i_s2), set())
                        for i_s1, i_s2 in product(insts_s1, insts_s2)
                    ])
                    self.query_count += 1       # Update count
                    # Collision-free join is possible when the resultant
                    # union does not have six members, standing for each
                    # direction of assembly, hence a feasible join path
                    join_possible = len(collision_directions) < 6

            if join_possible:
                # Cache valid join
                self.valid_joins.add(frozenset([insts_s1, insts_s2]))
            else:
                # Join of this subassembly pair proved to be unreachable
                # while including current object members; add nogood
                # as appropriate and return
                if not subset_cached_as_invalid:
                    minimal_pair = self.analyze_collision((nodes_s1, nodes_s2))
                    if minimal_pair is not None:
                        self.invalid_joins.add(minimal_pair)

                join_lit = self.join_actions_a2l[(s1, s2, n1, n2, t)]
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
                for s1, s2, n1, n2, t in self.join_actions_l2a[lit]:
                    if t in status["joins"]: 
                        assert status["joins"][t] == (s1, s2, n1, n2)
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

        # Referents & predicates info
        referents = {
            "e": { "mood": "?" },       # Interrogative
            "x0": { "entity": None, "rf_info": {} }
        }
        predicates = { "pc0": (("n", target_sym), f"pcls_{target_conc}") }

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
    'pick_up', 'drop' and 'assemble' actions.
    """
    effect_lit = [lit for lit in effect if lit.name.startswith("arel")][0]
    action_name = int(effect_lit.name.split("_")[-1])
    action_name = agent.lt_mem.lexicon.d2s[("arel", action_name)][0][1]

    # Shortcut vars
    referents = agent.lang.dialogue.referents
    exec_state = agent.planner.execution_state      # Shortcut var

    # Mapping from Unity-side name to scene object index
    env_handle_inv = {
        obj["env_handle"]: oi
        for oi, obj in agent.vision.scene.items()
    }

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

        # Poses of individual parts in the object picked up
        part_poses = {}
        for i in range(num_parts):
            part_uid = referents["dis"][effect_lit.args[6+3*i][0]]["name"]
            part_poses[env_handle_inv[part_uid]] = (
                flip_quaternion_y(xyzw2wxyz(parse_floats(6+3*i+1))),
                flip_position_y(parse_floats(6+3*i+2))
            )

        exec_state["manipulator_states"][manip_ind] = \
            (manip_state[0],) + (manip_pose, part_poses)

    if action_name.startswith("drop"):
        # Drop action effects: not much, just clear the corresponding manipulator
        manip_ind = 0 if action_name.endswith("left") else 1
        exec_state["manipulator_states"][manip_ind] = (None, None, None)

    if action_name.startswith("assemble"):
        # Assemble action effects: closest contact point pairs after the joining
        # movement*, pose of target-side manipulator, poses of all atomic parts
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
        part_poses = {}
        for i in range(num_parts):
            start_ind = ind_offset + 1 + 3 * i
            part_uid = referents["dis"][effect_lit.args[start_ind][0]]["name"]
            part_poses[env_handle_inv[part_uid]] = (
                flip_quaternion_y(xyzw2wxyz(parse_floats(start_ind+1))),
                flip_position_y(parse_floats(start_ind+2))
            )

        exec_state["manipulator_states"][manip_ind] = \
            (manip_state[0],) + (manip_pose, part_poses)

    if action_name.startswith("disassemble"):
        # Disassemble action effects: poses of both manipulators, poses of all
        # atomic parts included in each disassembly resultants objects
        manip_ind = 0 if action_name.endswith("left") else 1

        # (Former) names of joined subassemblies, obtained from action history,
        # used as names of the disassembly results again
        prev_left, prev_right = exec_state["action_history"][-1][1][:2]

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
        graph_left = sa_graph.subgraph(part_poses_left)
        graph_right = sa_graph.subgraph(part_poses_right)
        exec_state["connection_graphs"][prev_left] = graph_left
        exec_state["connection_graphs"][prev_right] = graph_right
