"""
Simulated user which takes part in dialogue with rule-based pattern matching
-- no cognitive architecture ongoing within the user
"""
import re
import copy
import random
from collections import defaultdict
from itertools import combinations, groupby

import yaml
import inflect
import networkx as nx
from clingo import Control, Function


singularize = inflect.engine().singular_noun
pluralize = inflect.engine().plural

MATCH_THRES = 0.9
VALID_JOINS = {
    (("wheel", "bolt"), ("bolt", "bolt")),
    (("fl_fender", "wheel"), ("wheel", "bolt")),
    (("fl_fender", "wheel"), ("bolt", "bolt")),
    (("fr_fender", "wheel"), ("wheel", "bolt")),
    (("fr_fender", "wheel"), ("bolt", "bolt")),
    (("bl_fender", "wheel"), ("wheel", "bolt")),
    (("bl_fender", "wheel1"), ("wheel", "bolt")),
    (("bl_fender", "wheel1"), ("bolt", "bolt")),
    (("bl_fender", "wheel2"), ("wheel", "bolt")),
    (("bl_fender", "wheel2"), ("bolt", "bolt")),
    (("br_fender", "wheel"), ("wheel", "bolt")),
    (("br_fender", "wheel1"), ("wheel", "bolt")),
    (("br_fender", "wheel1"), ("bolt", "bolt")),
    (("br_fender", "wheel2"), ("wheel", "bolt")),
    (("br_fender", "wheel2"), ("bolt", "bolt")),
    (("chassis_front", "cabin1"), ("cabin", "front1")),
    (("chassis_front", "cabin1"), ("bolt", "bolt")),
    (("chassis_front", "cabin2"), ("bolt", "bolt")),
    (("chassis_front", "lfw"), ("fl_fender", "wheel")),
    (("chassis_front", "rfw"), ("fr_fender", "wheel")),
    (("chassis_back", "load"), ("load", "back")),
    (("chassis_back", "load"), ("bolt", "bolt")),
    (("chassis_back", "lfw0"), ("bl_fender", "wheel")),
    (("chassis_back", "lfw1"), ("bl_fender", "wheel1")),
    (("chassis_back", "rfw0"), ("bl_fender", "wheel")),
    (("chassis_back", "rfw1"), ("br_fender", "wheel1")),
    (("chassis_front", "center"), ("chassis_center", "front")),
    (("chassis_front", "center"), ("bolt", "bolt")),
    (("chassis_center", "back"), ("chassis_back", "center")),
    (("chassis_center", "back"), ("bolt", "bolt")),
}
VALID_JOINS |= {(p2, p1) for p1, p2 in VALID_JOINS}     # Symmetric

class SimulatedTeacher:
    
    def __init__(self, cfg):
        self.cfg = cfg

        # History of ITL episode records
        self.episode_records = []

        # Pieces of generic constrastive knowledge taught across episodes
        self.taught_diffs = set()

        # Teacher's strategy on how to give feedback upon student's wrong answer
        # (provided the student has taken initiative for extended ITL interactions
        # by asking further questions after correct answer feedback)
        self.strat_feedback = cfg.exp.strat_feedback

        # Whether the agent is running in test mode
        self.agent_test_mode = cfg.agent.test_mode

        # Queue of actions to execute, if any demonstration is ongoing
        self.ongoing_demonstration = []

        # Load appropriate domain knowledge stored as yamls in assets dir
        self.domain_knowledge = {}
        knowledge_path = f"{cfg.paths.assets_dir}/domain_knowledge"
        with open(f"{knowledge_path}/definitions.yaml") as yml_f:
            self.domain_knowledge["definitions"] = yaml.safe_load(yml_f)
        with open(f"{knowledge_path}/constraints.yaml") as yml_f:
            self.domain_knowledge["constraints"] = yaml.safe_load(yml_f)
        with open(f"{knowledge_path}/ontology_taxonomy.yaml") as yml_f:
            self.domain_knowledge["taxonomy"] = yaml.safe_load(yml_f)
        with open(f"{knowledge_path}/ontology_partWhole.yaml") as yml_f:
            self.domain_knowledge["part_whole"] = yaml.safe_load(yml_f)

        # Tabulate and store ontology knowledge in graph forms, so that transitive
        # closures are easily obtained
        taxonomy_graph = nx.DiGraph()
        for supertype, data in self.domain_knowledge["taxonomy"].items():
            is_proper = data["relation_type"] == "proper"
            for subtype in data["subtypes"]:
                taxonomy_graph.add_edge(supertype, subtype, is_proper=is_proper)
        for definiendum, definiens in self.domain_knowledge["definitions"].items():
            taxonomy_graph.add_edge(definiens["supertype"], definiendum, is_proper=True)
        self.domain_knowledge["taxonomy"] = taxonomy_graph

        part_whole_graph = nx.DiGraph()
        for whole, data in self.domain_knowledge["part_whole"].items():
            for part, count in data["parts"].items():
                part_whole_graph.add_edge(whole, part, count=count)
        self.domain_knowledge["part_whole"] = part_whole_graph

    def setup_episode(self, target_task, all_subtypes):
        """
        Preparation of a new interaction episode, comprising random initialization
        of the task for the episode and queueing of target concepts to teach
        """
        self.current_episode_record = {
            # Track state of ongoing assembly progress, either observed or demonstrated
            "assembly_state": {
                "left": None, "right": None,
                "subassemblies": defaultdict(set),
                "aliases": {}
            }
        }
        self.target_task = target_task

        # These constraints are always included in order to account for physically
        # infeasible part combinations...
        constraints = [
            ("forall", "truck", (["spares_chassis_center", "dumper"], None), False),
            ("forall", "truck", (["staircase_chassis_center", "dumper"], None), False),
            ("forall", "truck", (["staircase_chassis_center", "rocket_launcher"], None), False)
        ]

        if target_task == "build_truck_supertype":
            # More 'basic' experiment suite invested on learning part types, valid
            # structures of trucks (& subassemblies) and contact pairs & points
            self.target_concept = "truck"

            # Sampling with minimal constraints, just so enough that a valid truck
            # structure can be built; no distractors added
            num_distractors = 0

        else:
            assert target_task == "build_truck_subtype"

            # 'Advanced' stage invested on learning definitions of truck subtypes,
            # along with rules and constraints that influence trucks in general
            # or specific truck subtypes
            self.target_concept = random.sample([
                "base_truck", "dumper_truck", "missile_truck", "fire_truck"
            ], 1)[0]

            # Sampling with full consideration of constraints in domain knowledge;
            # add distractors specifically selected to allow mistakes
            num_distractors = 4
            constraints += [
                ("exists", definiendum, (list(definiens["parts"].values()), []), True)
                for definiendum, definiens in self.domain_knowledge["definitions"].items()
            ] + [
                (rule_type, scope_class, tuple(entry), True)
                for scope_class, rules in self.domain_knowledge["constraints"].items()
                for rule_type, entries in rules.items()
                for entry in entries
            ]

        # Leverage ASP to obtain all possible samples of valid part set subject to
        # a set of constraints
        sampled_inits = _sample_ASP(
            self.target_concept, self.domain_knowledge,
            constraints, all_subtypes, num_distractors, self.cfg.seed
        )

        # Extract and store sampled part subtype info for the current scene
        sampled_parts = {}
        for spec, subtype_id in sampled_inits.items():
            supertype, attribute_type, obj_id = spec.split("/")
            if attribute_type != "type": continue
            if not obj_id.startswith("t"): continue

            obj_id = int(obj_id.replace("t", ""))
            sampled_parts[(supertype, obj_id)] = all_subtypes[supertype][subtype_id]
        self.current_episode_record["sampled_parts"] = sampled_parts

        # Return sampled parts to pass as environment parameters
        return sampled_inits

    def initiate_dialogue(self):
        """
        Prepare an opening line for starting a new thread of dialogue based on
        the current target concept (unprompted dialogue starter, as opposed to
        self.react method below, which is invoked only when prompted by an agent
        reaction)
        """
        return [{
            "utterance": f"Build a {self.target_concept}.",
            "pointing": {}
        }]

    def react(self, agent_reactions):
        """ Rule-based pattern matching for handling agent responses """
        response = []
        step_demonstrated = False
        sampled_parts = self.current_episode_record["sampled_parts"]
        for utt, dem_refs in agent_reactions:
            if re.match(r"I don't know what '(.*)' means\.$", utt):
                # Agent reported it does not know what the target concept is
                # (and thus unable to build it). If target concept is the general
                # 'truck' supertype, demonstrate how to build one from existing
                # parts by sampling and executing a work plan. If target concept
                # is a fine-grained subtype of truck, provide NL definition.
                reported_neologism = re.findall(r"I don't know what '(.*)' means\.$", utt)[0]
                assert reported_neologism == self.target_concept

                # Sample a valid workplan for demonstration and load for execution
                sampled_plan = _sample_demo_plan(copy.deepcopy(sampled_parts))
                self.ongoing_demonstration = sampled_plan

                # Notify agent that user will demonstrate how to build one
                response.append((
                    "generate",
                    {
                        "utterance": f"I will demonstrate how to build a {self.target_concept}.",
                        "pointing": {}
                    }
                ))

            elif utt == "# Observing":
                # Agent has signaled it is paying attention to user's demonstration

                # Demonstrate at most one step per turn; discard any additional
                # observation signals
                if step_demonstrated: break
                step_demonstrated = True

                if len(self.ongoing_demonstration) == 0:
                    # Last action executed, demonstration finished
                    response.append((
                        "generate",
                        {
                            "utterance": f"This is a {self.target_concept}.",
                            "pointing": { (0, 4): "/truck" }
                        }
                    ))

                else:
                    # Keep popping and executing plan actions until plan is empty
                    act_type, act_params = self.ongoing_demonstration.pop(0)
                    # Shortcuts
                    assem_st = self.current_episode_record["assembly_state"]
                    subassems = assem_st["subassemblies"]

                    # Annotate physical action to be executed for this step, communicated
                    # to agent along with the action
                    act_str_prefix = f"# Action: {act_type}"
                    offset = len(act_str_prefix)

                    act_dscr = None
                    if act_type.startswith("pick_up"):
                        # For pick_up~ actions, provide demonstrative reference by string path
                        # to target GameObject (which will be converted into binary mask)
                        target = act_params[0]
                        crange = (offset+1, offset+1+len(target))
                        act_anno = {
                            "utterance": f"{act_str_prefix}({target})",
                            "pointing": { crange: "/" + act_params[0] }
                        }
                        assem_st["left" if act_type.endswith("left") else "right"] = target
                        if target not in subassems:
                            subassems[target].add(target)

                        atomic_parts = {
                            f"t_{inst[0]}_{inst[1]}": part_type
                            for inst, part_type in sampled_parts.items()
                        }
                        if target in atomic_parts:
                            target_label = f"a {atomic_parts[target]}"
                        else:
                            target_label = "the subassembly"
                        act_dscr = {
                            "utterance": f"Pick up {target_label}.",
                            "pointing": {}
                        }

                    elif act_type.startswith("assemble"):
                        # For assemble~ actions, provide contact point info, specified by
                        # (atomic part supertype, point identifier string) pair. Parameters
                        # after the third represent requests for ground truth masks of
                        # subassembly component parts.
                        subassembly, target_l, target_r = act_params
                        # Building action spec description string
                        num_components = \
                            len(subassems[assem_st["left"]] | subassems[assem_st["right"]])
                        act_str = act_str_prefix
                        act_str += f"({subassembly},{target_l},{target_r},{num_components},"
                        offset = len(act_str)
                        mask_requests = []; pointings = []
                        for topmost_held in [assem_st["left"], assem_st["right"]]:
                            for part in subassems[topmost_held]:
                                mask_requests.append(part)
                                part_type = re.findall(r"^t_(.*)_\d+$", part)[0]
                                pointings.append((
                                    (offset, offset+len(part)),
                                    f"/Teacher Agent/*/{topmost_held}/{part_type}"
                                ))
                                offset += len(part) + 1
                        act_str += ",".join(mask_requests) + ")"
                        act_anno = { "utterance": act_str, "pointing": dict(pointings) }
                        subassems[subassembly] = \
                            subassems.pop(assem_st["left"]) | subassems.pop(assem_st["right"])
                        assem_st["left"] = subassembly if act_type.endswith("left") else None
                        assem_st["right"] = None if act_type.endswith("left") else subassembly

                    elif act_type.startswith("inspect"):
                        # For inspect~ actions, provide integer index of (relative) viewpoint
                        hand = "Left" if act_type.endswith("left") else "Right"
                        path_prefix = f"/Teacher Agent/{hand} Hand"
                        target, view_ind = act_params
                        crange = (offset+1, offset+1+len(target))
                        act_anno = {
                            "utterance": f"{act_str_prefix}({target},{view_ind})",
                            "pointing": {
                                crange: "/".join([path_prefix, "*"])
                            }
                        }

                    else:
                        # No parameter info to communicate, just annotate action type
                        act_anno = {
                            "utterance": f"{act_str_prefix}()",
                            "pointing": {}
                        }

                    # Execute physical action
                    act_params_serialized = tuple(
                        f"{type(prm).__name__}|{prm}" for prm in act_params
                    )
                    response.append((act_type, { "parameters": act_params_serialized }))
                    # Provide additional action annotations
                    response.append(("generate", act_anno))
                    if act_dscr is not None: response.append(("generate", act_dscr))

            elif utt.startswith("# Action:"):
                # Agent action intent; somewhat like reading into the agent's 'mind'
                # for convenience's sake
                assem_st = self.current_episode_record["assembly_state"]

                if utt.startswith("# Action: pick_up"):
                    # Obtain agent-side naming of the targetted object
                    side, target = re.findall(r"# Action: pick_up_(.*)\((.*)\)$", utt)[0]
                    assem_st[side] = target

                if utt.startswith("# Action: assemble"):
                    # Obtain intent of joining the target subassembly pairs at which
                    # component parts
                    _, params = re.findall(r"# Action: assemble_(.*)\((.*)\)$", utt)[0]
                    _, _, part_left, part_right = params.split(",")
                    assem_st["join_intent"] = (part_left, part_right)

                response.append((None, None))

            elif utt.startswith("# Effect:"):
                # Action effect feedback from Unity environment, determine whether the
                # latest action was incorrect and agent needs to be interrupted
                assem_st = self.current_episode_record["assembly_state"]

                interrupted = False
                if utt.startswith("# Effect: pick_up"):
                    # Effect of pick up action
                    side, effects = re.findall(r"# Effect: pick_up_(.*)\((.*)\)$", utt)[0]
                    obj_picked_up = effects.split(",")[2]

                    agent_side_alias = assem_st.pop(side)
                    assem_st["aliases"][agent_side_alias] = obj_picked_up
                    assem_st[side] = obj_picked_up

                    if False:
                        # Interrupt if agent has picked up an object not needed in the
                        # desired goal structure, or if agent is holding a pair of objects
                        # that should not be assembled
                        interrupted = True

                if utt.startswith("# Effect: assemble"):
                    # Effect of assemble action; interested in closest contact points
                    direction, effects = re.findall(r"# Effect: assemble_(.*)\((.*)\)$", utt)[0]
                    if direction.endswith("left"):
                        part_tgt, part_src = assem_st.pop("join_intent")
                    else:
                        part_src, part_tgt = assem_st.pop("join_intent")
                    part_src = assem_st["aliases"][part_src]
                    part_tgt = assem_st["aliases"][part_tgt]
                    part_src, _ = re.findall(r"t_(.*)_(\d+)$", part_src)[0]
                    part_tgt, _ = re.findall(r"t_(.*)_(\d+)$", part_tgt)[0]
                    valid_joins = [
                        ((part1, cp1), (part2, cp2))
                        for (part1, cp1), (part2, cp2) in VALID_JOINS
                        if part1 == part_src and part2 == part_tgt
                    ]

                    effects = effects.split(",")
                    num_cp_pairs = int(effects[0])
                    cp_pairs = effects[1:1+4*num_cp_pairs]
                    cp_pairs = [cp_pairs[4*i:4*(i+1)] for i in range(num_cp_pairs)]
                    cp_pairs = [
                        (tuple(cp_src.split("/")), tuple(cp_tgt.split("/")))
                        for cp_src, cp_tgt, _, _ in cp_pairs
                    ]

                    if False:
                        # Interrupt if agent has assembled a pair of objects at incorrect
                        # contact point, which is determined by checking whether top 3
                        # contact point pairs contain a valid join for this episode's
                        # goal structure.
                        # (Do we also need to threshold the difference value? Do we want
                        # to be more strict?)
                        interrupted = True

                    print(0)

                if not interrupted:
                    # No interruption needs to be made, keep observing...
                    response.append(("generate", { "utterance": "# Observing", "pointing": {} }))

            elif utt == "OK.":
                # No further interaction needed; effectively terminates current episode
                pass

            else:
                # Cannot parse this pattern of reaction
                pass

        return response


def _sample_ASP(
        goal_object, domain_knowledge, constraints, all_subtypes, num_distractors, seed
    ):
    """
    Helper method factored out for writing & running ASP program for controlled
    sampling, subject to provided domain knowledge. (If some other person ever gets
    to read this part, sorry in advance :s)
    """
    base_prg_str = "hasPart(O0, O2) :- hasPart(O0, O1), hasPart(O1, O2).\n"

    # Add rules for covering whole-to-part derivations
    pw_graph = domain_knowledge["part_whole"]
    pw_rules = [(n, pw_graph.out_edges(n, data="count")) for n in pw_graph.nodes]
    pw_rules = [(n, r) for n, r in pw_rules if len(r) > 0]
    i = 0; occurring_subassemblies = set()
    for hol, per_holonym in pw_rules:
        for _, mer, count in per_holonym:
            occurring_subassemblies.add(mer)
            for _ in range(count):
                base_prg_str += f"{mer}(f_{i}(O)) :- {hol}(O).\n"
                base_prg_str += f"hasPart(O, f_{i}(O)) :- {hol}(O).\n"
                i += 1

    # Add rules for covering supertype-subtype relations and choices
    tx_graph = domain_knowledge["taxonomy"]
    tx_rules = [(n, tx_graph.out_edges(n)) for n in tx_graph.nodes]
    tx_rules = [(n, r) for (n, r) in tx_rules if len(r) > 0]
    for hyper, per_hypernym in tx_rules:
        if hyper == "color": continue

        for _, hypo in per_hypernym:
            base_prg_str += f"{hyper}(O) :- {hypo}(O).\n"
        if hyper not in occurring_subassemblies: continue

        base_prg_str += "1{ "
        base_prg_str += "; ".join(f"{hypo}(O)" for _, hypo in per_hypernym)
        base_prg_str += " }1 :- "
        base_prg_str += f"{hyper}(O).\n"

    # Add rules for covering color choices for applicable parts
    base_prg_str += "1{ hasColor(O, C) : color(C) }1 :- colored_part(O).\n"
    base_prg_str += ". ".join(f"color({c})" for c in all_subtypes["color"]) + ".\n"

    # Add rules for specifying atomic parts
    for supertype, subtypes in all_subtypes.items():
        if supertype == "color": continue
        for subtype in subtypes:
            base_prg_str += f"atomic(O, {subtype}) :- {subtype}(O).\n"

    # Add integrity constraints for filtering invalid combinations
    for ci, (rule_type, scope_class, entry, _) in enumerate(constraints):
        base_prg_str += f"rule({ci}).\n"
        parts, attributes = entry

        full_scope_str = f"{scope_class}(O), "
        full_scope_str += ", ".join(f"{p}(O{i})" for i, p in enumerate(parts)) + ", "
        full_scope_str += ", ".join(f"hasPart(O, O{i})" for i in range(len(parts)))
        if len(parts) > 1:
            obj_pairs = combinations(range(len(parts)), 2)
            full_scope_str += ", "
            full_scope_str += ", ".join(f"O{i} != O{j}" for i, j in obj_pairs)
        base_prg_str += f"{rule_type}_applicable({ci}) :- {scope_class}(O).\n"

        if attributes is None:
            base_prg_str += f"exception_found({ci}) :- {full_scope_str}.\n"
        else:
            attr_args_str = ", ".join(f"O{a}" for a in range(len(parts)))
            piece_strs = []

            for attr, pol, arg_inds in attributes:
                if attr == "_same_color":
                    assert len(arg_inds) == 2
                    piece_str = f"hasColor(O{arg_inds[0]}, C0), "
                    piece_str += f"hasColor(O{arg_inds[1]}, C1), "
                    piece_str += "C0 = C1" if pol else "C0 != C1"
                else:
                    piece_args_str = ", ".join(f"O{a}" for a in arg_inds)
                    piece_str = "" if pol else "not "
                    if attr in all_subtypes["color"]:
                        piece_str += f"hasColor({piece_args_str}, {attr})"
                    else:
                        piece_str += f"{attr}({piece_args_str})"
                piece_strs.append(piece_str)

            match rule_type:
                case "exists":
                    base_prg_str += \
                        ", ".join([f"case_found({ci}) :- {full_scope_str}"]+piece_strs) + ".\n"
                case "forall":
                    base_prg_str += \
                        ", ".join([f"attr_{ci}({attr_args_str}) :- {full_scope_str}"]+piece_strs) + ".\n"
                    base_prg_str += \
                        f"exception_found({ci}) :- {full_scope_str}, not attr_{ci}({attr_args_str}).\n"
                case _:
                    raise ValueError("Invalid rule type")

    # 'exists' rules are violated if required attribute case is not found; 'forall' rules
    # if any exception to required attribute is found
    base_prg_str += "rule_violated(R) :- rule(R), exists_applicable(R), not case_found(R).\n"
    base_prg_str += "rule_violated(R) :- rule(R), forall_applicable(R), exception_found(R).\n"

    # Control the number of rules to be violated (0 or 1) by an external query atom
    base_prg_str += ":- violate_none, rule_violated(R).\n"
    base_prg_str += ":- violate_one, not 1{ rule_violated(R) : rule(R) }1.\n"

    # Finally add a grounded instance representing the whole truck sample
    base_prg_str += f"{goal_object}(o).\n"

    # Program fragment defining the external control atoms
    base_prg_str += "#external violate_none.\n"
    base_prg_str += "#external violate_one.\n"

    ctl = Control(["--warn=none"])
    ctl.configuration.solve.models = 0
    ctl.configuration.solver.seed = seed
    ctl.add("base", [], base_prg_str)
    ctl.ground([("base", [])])

    models = []
    ctl.assign_external(Function("violate_none", []), True)
    with ctl.solve(yield_=True) as solve_gen:
        for m in solve_gen:
            models.append(m.symbols(atoms=True))
            if len(models) > 30000: break       # Should be enough...

    # Randomly select a sampled model
    sampled_model = random.sample(models, 1)[0]

    # Inverse map from subtype to supertype
    all_subtypes_inv_map = {x: k for k, v in all_subtypes.items() for x in v}

    # Filter out unnecessary atoms, to extract atomic parts and colors sampled
    tgt_parts_by_fname = {}; tgt_colors_by_fname = {}
    for atm in sampled_model:
        # Collate part types first, indexing by skolem function name
        if atm.name == "atomic":
            tgt_parts_by_fname[atm.arguments[0].name] = (
                atm.arguments[1].name, all_subtypes_inv_map[atm.arguments[1].name]
            )
    for atm in sampled_model:
        # Then fetch matching colors for colorable parts
        if atm.name == "hasColor":
            tgt_colors_by_fname[atm.arguments[0].name] = atm.arguments[1].name

    # Group by part supertype
    tgt_objs_grouped_by_supertype = {
        k: sorted([f for f, _ in v], key=lambda x: int(x.replace("f_", "")))
        for k, v in groupby(tgt_parts_by_fname.items(), key=lambda x: x[1][1])
    }

    if num_distractors > 0:
        # For distractors, we want to sample additional part groups that violate
        # exactly one of the constraints
        ctl.assign_external(Function("violate_none", []), False)
        ctl.assign_external(Function("violate_one", []), True)

        # Part groups to consider resampling as distractors
        part_groups_by_type = [
            [["cabin"]],
            [["load"]],
            [["chassis_center"]],
            [["fl_fender", "fr_fender", "bl_fender", "br_fender"], ["wheel"]]
        ]
        part_groups_by_obj = [
            [
                sum([tgt_objs_grouped_by_supertype[p] for p in group], [])
                for group in part_supertypes
            ]
            for part_supertypes in part_groups_by_type
        ]
        part_specs = {
            obj: (tgt_parts_by_fname[obj][0], tgt_colors_by_fname.get(obj))
            for groups in part_groups_by_obj
            for group in groups
            for obj in group
        }

        # Remove already sampled parts in this group so that exact same part
        # specs (part type and color if applicable) cannot be sampled again,
        # while fixing all other parts as identical; implement as additional
        # clingo program fragment
        sampled_distractor_groups = []
        dtr_parts_by_fname = {}; dtr_colors_by_fname = {}

        i = 0; sample_pool = list(range(len(part_groups_by_obj)))
        while len(sampled_distractor_groups) < num_distractors:
            if len(sample_pool) == 0:
                # No more distractor sampling possible, terminate here
                break

            gi = random.sample(sample_pool, 1)[0]
            sample_pool.remove(gi)

            # Program fragment representing additional temporary constraints
            alt_prg_str = f"#external active_{i}.\n"

            # Ban current specs of the resampling target group
            for group in part_groups_by_obj[gi]:
                for obj in group:
                    part_type_str = f"{part_specs[obj][0]}(O)"
                    part_color_str = "" if part_specs[obj][1] is None \
                        else f", hasColor(O, {part_specs[obj][1]})"
                    alt_prg_str += f":- {part_type_str}{part_color_str}, active_{i}.\n"

            # Enforce current specs for the remainder
            piece_strs = []; oi = 0
            for gj, groups in enumerate(part_groups_by_obj):
                if gi == gj: continue

                for group in groups:
                    for obj in group:
                        part_type_str = f"{part_specs[obj][0]}(O{oi})"
                        part_color_str = "" if part_specs[obj][1] is None \
                            else f", hasColor(O{oi}, {part_specs[obj][1]})"

                        piece_strs.append(part_type_str + part_color_str)
                        oi += 1

            piece_strs += [f"O{oi} != O{oj}" for oi, oj in combinations(range(oi), 2)]

            alt_prg_str += f"remainder_preserved_{i} :- "
            alt_prg_str += ", ".join(piece_strs)
            alt_prg_str += f", active_{i}.\n"

            alt_prg_str += f":- not remainder_preserved_{i}, active_{i}.\n"

            # Manage solver control by appropriate processing of external atoms and
            # grounding
            if i > 0:
                ctl.release_external(Function(f"active_{i-1}", []))
                ctl.cleanup()
            ctl.add(f"distractor_sampling_{i}", [], alt_prg_str)
            ctl.ground([(f"distractor_sampling_{i}", [])])
            ctl.assign_external(Function(f"active_{i}", []), True)

            models = []
            with ctl.solve(yield_=True) as solve_gen:
                for m in solve_gen:
                    if len(models) > 1000: break
                    models.append(m.symbols(atoms=True))

            if len(models) == 0:
                # No possible sampling of distractor that violate exactly one rule
                i += 1
                continue

            # Some rules have more violating distractor samples than others; sample
            # by rule-basis, not model-basis
            violated_constraints = [
                [atm.arguments[0].number for atm in m if atm.name == "rule_violated"][0]
                for m in models
            ]
            model_inds_by_constraint = {
                ci: [mi for mi, cj in enumerate(violated_constraints) if ci==cj]
                for ci in range(len(constraints))
            }
            sampled_violated_constraint = random.sample([
                ci for ci, (_, _, _, can_violate) in enumerate(constraints)
                if can_violate and len(model_inds_by_constraint[ci]) > 0
            ], 1)[0]
            sampled_model = random.sample(
                model_inds_by_constraint[sampled_violated_constraint]
            , 1)[0]
            sampled_model = models[sampled_model]

            # Filter out unnecessary atoms, to extract atomic parts and colors sampled
            types_to_collect = sum(part_groups_by_type[gi], [])
            for atm in sampled_model:
                # Collate part types in resampled group
                if atm.name == "atomic":
                    part_supertype = all_subtypes_inv_map[atm.arguments[1].name]
                    if part_supertype in types_to_collect:
                        dtr_parts_by_fname[atm.arguments[0].name] = (
                            atm.arguments[1].name, part_supertype
                        )
            for atm in sampled_model:
                # Then fetch matching colors for colorable parts
                if atm.name == "hasColor" and atm.arguments[0].name in dtr_parts_by_fname:
                    dtr_colors_by_fname[atm.arguments[0].name] = atm.arguments[1].name

            i += 1
            sampled_distractor_groups += part_groups_by_type[gi]

        dtr_objs_grouped_by_supertype = defaultdict(list)
        for obj, (_, part_supertype) in dtr_parts_by_fname.items():
            dtr_objs_grouped_by_supertype[part_supertype].append(obj)
        dtr_objs_grouped_by_supertype = dict(dtr_objs_grouped_by_supertype)

    # Organize into final return value dict
    sampled_inits = {}
    for obj, (part_subtype, part_supertype) in tgt_parts_by_fname.items():
        oi = tgt_objs_grouped_by_supertype[part_supertype].index(obj)
        sampled_inits[f"{part_supertype}/type/t{oi}"] = \
            all_subtypes[part_supertype].index(part_subtype)

        if obj in tgt_colors_by_fname:
            sampled_inits[f"{part_supertype}/color/t{oi}"] = \
                all_subtypes["color"].index(tgt_colors_by_fname[obj])

    if num_distractors > 0:
        for i in range(min(num_distractors, len(sampled_distractor_groups))):
            dtr_group = sampled_distractor_groups[i]
            for part_supertype in dtr_group:
                for oi, obj in enumerate(dtr_objs_grouped_by_supertype[part_supertype]):
                    part_subtype = dtr_parts_by_fname[obj][0]
                    sampled_inits[f"{part_supertype}/type/d{oi}"] = \
                        all_subtypes[part_supertype].index(part_subtype)

                    if obj in dtr_colors_by_fname:
                        sampled_inits[f"{part_supertype}/color/d{oi}"] = \
                            all_subtypes["color"].index(dtr_colors_by_fname[obj])

    return sampled_inits


def _sample_demo_plan(sampled_parts):
    """
    Helper method factored out for sampling a valid assembly plan that builds
    a valid instance of truck structure. Since we (and the teacher) already know
    the desired structure, let's specify a partial order plan by a set of subgoals
    with precedence constraints and sample a valid instantiation accordingly.
    """
    # Represent partial plans with (action sequence, preconditions, name of resultant
    # subassembly) tuples
    partial_plan = [
        # Front-left fender-wheel-bolt unit
        (
            [
                ("pick_up_left", ("fl_fender",)),
                ("pick_up_right", ("wheel",)),
                ("assemble_right_to_left", ("fl_fw", "fl_fender/wheel", "wheel/bolt")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("fl_fw_unit", "fl_fender/wheel", "bolt/bolt")),
                ("drop_left", ())
            ],
            [],
            "fl_fw_unit"
        ),

        # Front-right fender-wheel-bolt unit
        (
            [
                ("pick_up_left", ("fr_fender",)),
                ("pick_up_right", ("wheel",)),
                ("assemble_right_to_left", ("fr_fw", "fr_fender/wheel", "wheel/bolt")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("fr_fw_unit", "fr_fender/wheel", "bolt/bolt")),
                ("drop_left", ())
            ],
            [],
            "fr_fw_unit"
        ),

        # Back-left fender-wheel-bolt unit
        (
            [
                ("pick_up_left", ("bl_fender",)),
            ] + ([
                ("pick_up_right", ("wheel",)),
                ("assemble_right_to_left", ("bl_fw", "bl_fender/wheel", "wheel/bolt")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("bl_fw_unit", "wheel/bolt", "bolt/bolt")),
            ] if not sampled_parts[("bl_fender", 0)].startswith("double") else [
                ("pick_up_right", ("wheel",)),
                ("assemble_right_to_left", ("bl_fw1", "bl_fender/wheel1", "wheel/bolt")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("bl_fw2", "bl_fender/wheel1", "bolt/bolt")),
                ("pick_up_right", ("wheel",)),
                ("assemble_right_to_left", ("bl_fw3", "bl_fender/wheel2", "wheel/bolt")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("bl_fw_unit", "bl_fender/wheel2", "bolt/bolt")),
            ]) + [
                ("drop_left", ())
            ],
            [],
            "bl_fw_unit"
        ),

        # Back-right fender-wheel-bolt unit
        (
            [
                ("pick_up_left", ("br_fender",)),
            ] + ([
                ("pick_up_right", ("wheel",)),
                ("assemble_right_to_left", ("br_fw", "br_fender/wheel", "wheel/bolt")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("br_fw_unit", "wheel/bolt", "bolt/bolt")),
            ] if not sampled_parts[("br_fender", 0)].startswith("double") else [
                ("pick_up_right", ("wheel",)),
                ("assemble_right_to_left", ("br_fw1", "br_fender/wheel1", "wheel/bolt", )),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("br_fw2", "br_fender/wheel1", "bolt/bolt")),
                ("pick_up_right", ("wheel",)),
                ("assemble_right_to_left", ("br_fw3", "br_fender/wheel2", "wheel/bolt")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("br_fw_unit", "br_fender/wheel2", "bolt/bolt")),
            ]) + [
                ("drop_left", ())
            ],
            [],
            "br_fw_unit"
        ),

        # Truck front
        (
            [
                ("pick_up_left", ("chassis_front",)),
                ("pick_up_right", ("cabin",)),
                ("assemble_right_to_left", ("cf0", "chassis_front/cabin1", "cabin/front1")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("cf1", "chassis_front/cabin1", "bolt/bolt")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("cf2", "chassis_front/cabin2", "bolt/bolt")),
                ("pick_up_right", ("fl_fw_unit",)),
                ("assemble_right_to_left", ("cfl", "chassis_front/lfw", "fl_fender/wheel")),
                ("pick_up_right", ("fr_fw_unit",)),
                ("assemble_right_to_left", ("truck_front", "chassis_front/rfw", "fr_fender/wheel")),
                ("drop_left", ())
            ],
            ["fl_fw_unit", "fr_fw_unit"],
            "truck_front"
        ),

        # Truck back
        (
            [
                ("pick_up_left", ("chassis_back",)),
                ("pick_up_right", ("load",)),
                ("assemble_right_to_left", ("lb0", "chassis_back/load", "load/back")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("lb1", "chassis_back/load", "bolt/bolt"))
            ] + ([
                ("pick_up_right", ("bl_fw_unit",)),
                ("assemble_right_to_left", ("lb2", "chassis_back/lfw0", "bl_fender/wheel"))
            ] if not sampled_parts[("bl_fender", 0)].startswith("double") else [
                ("pick_up_right", ("bl_fw_unit",)),
                ("assemble_right_to_left", ("lb2", "chassis_back/lfw1", "bl_fender/wheel1"))
            ]) + ([
                ("pick_up_right", ("br_fw_unit",)),
                ("assemble_right_to_left", ("truck_back", "chassis_back/rfw0", "bl_fender/wheel"))
            ] if not sampled_parts[("br_fender", 0)].startswith("double") else [
                ("pick_up_right", ("br_fw_unit",)),
                ("assemble_right_to_left", ("truck_back", "chassis_back/rfw1", "br_fender/wheel1"))
            ]) + [
                ("drop_left", ())
            ],
            ["bl_fw_unit", "br_fw_unit"],
            "truck_back"
        ),

        # Whole truck
        (
            [
                ("pick_up_left", ("truck_front",)),
                ("pick_up_right", ("chassis_center",)),
                ("assemble_right_to_left", ("tfc0", "chassis_front/center", "chassis_center/front")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("tfc1", "chassis_front/center", "bolt/bolt")),
                ("pick_up_right", ("truck_back",)),
                ("assemble_right_to_left", ("tfcb", "chassis_center/back", "chassis_back/center")),
                ("pick_up_right", ("bolt",)),
                ("assemble_right_to_left", ("truck", "chassis_center/back", "bolt/bolt")),
                ("drop_left", ())
            ],
            ["truck_front", "truck_back"],
            "truck"
        )
    ]

    # Now sample to instantiate the partial plan
    plan = []; introduced_subtypes = set()
    atomic_supertypes = {k[0] for k in sampled_parts}
    while len(partial_plan) > 0:
        # Sample (the index of) a subgoal whose preconditions are cleared
        subgoal_ind = random.sample([
            i for i, (_, preconditions, _) in enumerate(partial_plan)
            if len(preconditions) == 0
        ], 1)[0]

        act_seq, _, result = partial_plan.pop(subgoal_ind)

        # For manual check of fender-wheel compatibility...
        if result.endswith("_fw_unit"):
            large_wheel_needed = sampled_parts[(f"{result[:2]}_fender", 0)].startswith("large")
        else:
            large_wheel_needed = None

        # Each object argument of pick_up_left/Right arguments in the action
        # sequence is somewhat lifted, as they only specify type; 'ground'
        # them by sampling from existing part instances, then append to fully
        # instantiated plan
        for action in act_seq:
            if action[0].startswith("pick_up") and action[1][0] in atomic_supertypes:
                compatible_parts = []
                for instance, subtype in sampled_parts.items():
                    if instance[0] != action[1][0]: continue
                    if action[1][0] == "wheel" and large_wheel_needed is not None:
                        if subtype != "large_wheel" and large_wheel_needed: continue
                        if subtype == "large_wheel" and not large_wheel_needed: continue

                    compatible_parts.append(instance)

                instance = random.sample(compatible_parts, 1)[0]
                grounded_action = (action[0], (f"t_{instance[0]}_{instance[1]}",))
                subtype = sampled_parts.pop(instance)

                plan.append(grounded_action)

                # Interleave the sampled plan with 'inspect~' actions to allow 3D scanning
                # of newly introduced parts, for each first occurrence of pick_up~ action
                # involving a part subtype. Note that this implies any pick_up~ action not
                # followed by an inspect~ action involves a part subtype that is already
                # introduced before in a previously executed pick_up~ action.
                if subtype not in introduced_subtypes:
                    introduced_subtypes.add(subtype)
                    hand = "left" if action[0].endswith("left") else "right"
                    plan += [
                        (f"inspect_{hand}", (f"t_{instance[0]}_{instance[1]}", i))
                        for i in range(41)
                    ]

            else:
                plan.append(action)

        # Remove the resultant subassembly from the preconditions of the
        # remaining subgoals in partial_plan
        partial_plan = [
            (subgoal[0], [cond for cond in subgoal[1] if cond != result], subgoal[2])
            for subgoal in partial_plan
        ]

    # All sampled parts that build target object are used
    assert len(sampled_parts) == 0

    return plan
