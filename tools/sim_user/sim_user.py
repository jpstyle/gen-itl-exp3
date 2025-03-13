"""
Simulated user which takes part in dialogue with rule-based pattern matching
-- no cognitive architecture ongoing within the user
"""
import re
import copy
import random
import logging
from collections import defaultdict
from itertools import product, combinations, groupby

import yaml
import networkx as nx
from clingo import Control, Function, Number

from python.itl.comp_actions.interact import _divide_and_conquer


logger = logging.getLogger(__name__)

MATCH_THRES = 0.9
VALID_JOINS = {
    (("fl_fender", "wheel"), ("wheel", "bolt")),
    (("fl_fender", "wheel"), ("bolt", "bolt")),
    (("fr_fender", "wheel"), ("wheel", "bolt")),
    (("fr_fender", "wheel"), ("bolt", "bolt")),
    (("bl_fender", "wheel"), ("wheel", "bolt")),
    (("bl_fender", "wheel"), ("bolt", "bolt")),
    (("br_fender", "wheel"), ("wheel", "bolt")),
    (("br_fender", "wheel"), ("bolt", "bolt")),
    (("chassis_front", "cabin1"), ("cabin", "front1")),
    (("chassis_front", "cabin1"), ("bolt", "bolt")),
    (("chassis_front", "cabin2"), ("bolt", "bolt")),
    (("chassis_front", "lfw"), ("fl_fender", "wheel")),
    (("chassis_front", "rfw"), ("fr_fender", "wheel")),
    (("chassis_back", "load"), ("load", "back")),
    (("chassis_back", "load"), ("bolt", "bolt")),
    (("chassis_back", "lfw"), ("bl_fender", "wheel")),
    (("chassis_back", "rfw"), ("br_fender", "wheel")),
    (("chassis_front", "center"), ("chassis_center", "front")),
    (("chassis_front", "center"), ("bolt", "bolt")),
    (("chassis_center", "back"), ("chassis_back", "center")),
    (("chassis_center", "back"), ("bolt", "bolt")),
}

class SimulatedTeacher:
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.next_seed = cfg.seed

        # History of ITL episode records
        self.episode_records = []

        # Pieces of generic constrastive knowledge taught across episodes
        self.taught_diffs = set()

        # Teacher's strategy on how to give feedback upon student's wrong answer
        # (provided the student has taken initiative for extended ITL interactions
        # by asking further questions after correct answer feedback)
        self.player_type = cfg.exp.player_type

        # Queue of actions to execute, if any demonstration is ongoing
        self.ongoing_demonstration = (None, [])

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
        with open(f"{knowledge_path}/collision_table.yaml") as yml_f:
            self.collision_table = yaml.safe_load(yml_f)

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
                "left": None, "right": None, "subassemblies": {},
                "joins_remaining": {frozenset(join) for join in VALID_JOINS},
                "aliases": {}
            },
            "metrics": {
                "num_search_failure": 0,
                "num_invalid_pickup": 0,
                "num_invalid_join": 0,
                "num_planning_forfeiture": 0,
                "episode_discarded": False
            } if target_task != "inject_color" else {}
        }
        self.target_task = target_task

        # These constraints are always included in order to account for physically
        # infeasible part combinations...
        constraints = [
            ("forall", "truck", (["spares_chassis_center", "dumper"], None), False),
            ("forall", "truck", (["spares_chassis_center", "rocket_launcher"], None), False),
            ("forall", "truck", (["staircase_chassis_center", "dumper"], None), False)
        ]

        if target_task == "build_truck_supertype":
            # More 'basic' experiment suite invested on learning part types, valid
            # structures of trucks (& subassemblies) and contact pairs & points
            if any(
                any(st is None for st in subtypes) for subtypes in all_subtypes.values()
            ):
                # Now already constrained after the first sampling
                sampled_truck_subtype = "truck"
            else:
                # First sampling, sample uniformly across truck subtypes (since naive
                # samples per ASP models are not evenly distributed)
                sampled_truck_subtype = random.sample([
                    "base_truck", "dump_truck", "container_truck", "missile_truck", "fire_truck"
                ], 1)[0]
            self.target_concept = "truck"       # Always denoted as 'truck' in any case

            # Sampling with minimal constraints, just so enough that a valid truck
            # structure can be built; no distractors added
            num_distractors = 0
            constraints += [
                ("exists", definiendum, (list(definiens["parts"].values()), []), True)
                for definiendum, definiens in self.domain_knowledge["definitions"].items()
            ] + [
                ("forall", "truck", (["normal_wheel", "large_wheel"], None), False),
            ]   # Keep the sizes of the wheels identical

        else:
            assert target_task in ["build_truck_subtype", "inject_color"]

            # 'Advanced' stage invested on learning definitions of truck subtypes,
            # along with rules and constraints that influence trucks in general
            # or specific truck subtypes
            sampled_truck_subtype = random.sample([
                "base_truck", "dump_truck", "container_truck", "missile_truck", "fire_truck"
            ], 1)[0]
            self.target_concept = sampled_truck_subtype

            # Sampling with full consideration of constraints in domain knowledge;
            # add distractors specifically selected to allow mistakes
            num_distractors = 3
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
        random.seed(self.next_seed)
        sampled_inits = self._sample_ASP(
            sampled_truck_subtype, self.domain_knowledge,
            constraints, all_subtypes, num_distractors
        )
        self.next_seed = random.randint(1, 1000000)
            # Set and record randomization seed at this point in order to ensure
            # planning problems are kept constant across experiments sharindg the
            # starting seed

        # Extract and store sampled part subtype info for the current scene
        sampled_parts = defaultdict(dict)
        for field, value in sampled_inits.items():
            supertype, obj_id, attribute_type = field.split("/")
            obj_id = re.findall(r"([td])(\d+)$", obj_id)[0]
            obj_id = (obj_id[0], int(obj_id[1]))

            inst_id = (supertype, obj_id)
            match attribute_type:
                case "type":
                    sampled_parts[inst_id]["type"] = \
                        all_subtypes[supertype][value]
                case "color":
                    sampled_parts[inst_id]["color"] = \
                        all_subtypes["color"][value]
                case "viol":
                    sampled_parts[inst_id]["violated_constraint"] = value

        self.current_episode_record["sampled_parts"] = dict(sampled_parts)

        # Return sampled parts to pass as environment parameters
        return sampled_inits

    def initiate_dialogue(self):
        """
        Prepare an opening line for starting a new thread of dialogue based on
        the current target concept (unprompted dialogue starter, as opposed to
        self.react method below, which is invoked only when prompted by an agent
        reaction)
        """
        if self.target_task == "inject_color":
            # Color injection pre-task; provide all positive color labels
            return [
                {
                    "utterance": f"This is {col}.",
                    "pointing": {
                        (0, 4): (f"/{inst[1][0]}_{inst[0]}_{inst[1][1]}", True)
                    }
                }
                for inst, info in self.current_episode_record["sampled_parts"].items()
                if (col := info.get("color")) is not None
            ]
        else:
            # Main task of building the truck supertype or a truck subtype
            return [{
                "utterance": f"Build a {self.target_concept}.",
                "pointing": {}
            }]

    def react(self, agent_reactions):
        """ Rule-based pattern matching for handling agent responses """
        response = []
        step_demonstrated = False

        # Shortcuts
        assem_st = self.current_episode_record["assembly_state"]
        subassems = assem_st["subassemblies"]
        sampled_parts = self.current_episode_record["sampled_parts"]
        ep_metrics = self.current_episode_record["metrics"]

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
                sampled_plan = self._sample_demo_plan()
                self.ongoing_demonstration = ("full", sampled_plan)

                # Notify agent that user will demonstrate how to build one
                response.append((
                    "generate",
                    {
                        "utterance": f"I will demonstrate how to build a {self.target_concept}.",
                        "pointing": {}
                    }
                ))

            elif utt == "# Observing":
                # Agent has signaled it is paying attention to user's action

                # Demonstrate at most one step per turn; discard any additional
                # observation signals
                if step_demonstrated: break
                step_demonstrated = True

                demo_type, demo_plan = self.ongoing_demonstration
                if len(demo_plan) == 0:
                    if demo_type == "full":
                        # Demonstration finished (including end-of-demo message),
                        # wait w/o terminating until receiving agent's acknowledgement
                        # message "OK" (unless already included)
                        if ("OK.", {}) not in agent_reactions:
                            response.append((None, None))
                        continue
                    elif demo_type == "frag":
                        # Tell the agent to resume its task execution
                        response.append(
                            ("generate", { "utterance": "Continue.", "pointing": {} })
                        )
                        self.ongoing_demonstration = (None, [])     # Demo over
                        continue
                    else:
                        # No-op; agent has presumably now exited pause mode
                        assert demo_type is None
                        response.append((None, None))
                        continue

                # Keep popping and executing plan actions until plan is empty
                act_type, act_params = demo_plan.pop(0)
                side = act_type.split("_")[-1] if act_type is not None else None

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
                        "pointing": { crange: ("/" + act_params[0], True) }
                    }
                    assem_st[side] = target
                    if target not in subassems:
                        singleton_gr = nx.Graph()
                        singleton_gr.add_node(target)
                        subassems[target] = singleton_gr

                    if self.player_type in ["label", "full"]:
                        inst = re.findall(r"^t_(.*)_(\d+)$", target)
                        inst = (inst[0][0], ("t", int(inst[0][1]))) if len(inst) == 1 else None
                        if inst in sampled_parts:
                            target_label = f"a {inst[0]}"
                        else:
                            if self.player_type == "label":
                                target_label = "the subassembly"
                            else:
                                target_label = f"the {target}"
                        act_dscr = {
                            "utterance": f"Pick up {target_label}.",
                            "pointing": {}
                        }

                elif act_type.startswith("drop"):
                    # No parameter info to communicate, just annotate action type
                    act_anno = {
                        "utterance": f"{act_str_prefix}()",
                        "pointing": {}
                    }
                    # Provide NL label for the subassembly type (not a 'description'
                    # of the action as such though) only for full feedback player
                    # types for holonymy/meronymy info involving meaningful units
                    # of substructures --- except the final product "truck", which
                    # must be provided for all player types
                    dropped_sa = assem_st[side]
                    if self.player_type == "full" or dropped_sa == "truck":
                        act_dscr = {
                            "utterance": f"This is a {dropped_sa}.",
                            "pointing": {
                                (0, 4): (f"/Student Agent/*/{dropped_sa}", True)
                            }
                        }

                elif act_type.startswith("assemble"):
                    # For assemble~ actions, provide contact point info, specified by
                    # (atomic part supertype, point identifier string) pair
                    subassembly, target_l, target_r = act_params
                    # Selecting a specific instance matching the target condition. One
                    # (ugly) assumption made here is that the contact point of interest
                    # can be uniquely specified by the part type alone.
                    type_l, cp_l = target_l.split("/")
                    type_r, cp_r = target_r.split("/")
                    inst_l = next(
                        inst for inst in subassems[assem_st["left"]] if type_l in inst
                    )
                    inst_r = next(
                        inst for inst in subassems[assem_st["right"]] if type_r in inst
                    )
                    target_l = f"{inst_l}/{cp_l}"
                    target_r = f"{inst_r}/{cp_r}"
                    act_params = (subassembly, target_l, target_r)    # Rewrite params
                    # Building action spec description string
                    num_components = \
                        len(subassems[assem_st["left"]]) + len(subassems[assem_st["right"]])
                    act_str = act_str_prefix
                    act_str += f"({subassembly},{target_l},{target_r},{num_components})"
                    act_anno = { "utterance": act_str, "pointing": {} }
                    subassems[subassembly] = nx.union(
                        subassems.pop(assem_st["left"]), subassems.pop(assem_st["right"])
                    )
                    subassems[subassembly].add_edge(
                        inst_l, inst_r, contact={
                            inst_l: (type_l, cp_l), inst_r: (type_r, cp_r)
                        }
                    )
                    assem_st["left"] = subassembly if side == "left" else None
                    assem_st["right"] = None if side == "left" else subassembly

                    if self.player_type in ["label", "full"]:
                        # Linguistic annotation (though this utterance doesn't provide
                        # any additional learning signals in our scope)
                        act_dscr = {
                            "utterance": f"Join the {type_l} and the {type_r}.",
                            "pointing": {}
                        }

                elif act_type.startswith("inspect"):
                    # For inspect~ actions, provide integer index of (relative) viewpoint
                    ent_path = f"/Student Agent/{side.capitalize()} Hand/{assem_st[side]}"
                    target, view_ind = act_params
                    crange = (offset+1, offset+1+len(target))
                    act_anno = {
                        "utterance": f"{act_str_prefix}({target},{view_ind})",
                        "pointing": { crange: (ent_path, True) }
                    }

                if act_type is not None:
                    # Execute physical action
                    act_params_serialized = tuple(
                        f"{type(prm).__name__}|{prm}" for prm in act_params
                    )
                    response.append((act_type, { "parameters": act_params_serialized }))
                    if demo_type == "full":
                        # Provide additional action annotations
                        response.append(("generate", act_anno))
                        if act_dscr is not None:
                            response.append(("generate", act_dscr))

            elif utt.startswith("# Action:"):
                # Agent action intent; somewhat like reading into the agent's 'mind'
                # for convenience's sake
                if utt.startswith("# Action: assemble"):
                    # Obtain intent of joining the target subassembly pairs at which
                    # component parts
                    _, params = re.findall(r"# Action: assemble_(.*)\((.*)\)$", utt)[0]
                    _, part_left, _, part_right, product_name = params.split(",")
                    assem_st["join_intent"] = (part_left, part_right, product_name)

                response.append((None, None))       # Then no-op

            elif utt.startswith("# Effect:"):
                # Action effect feedback from Unity environment, determine whether the
                # latest action was incorrect and agent needs to be interrupted

                interrupted = False
                if utt.startswith("# Effect: pick_up"):
                    # Effect of pick up action
                    side, effects = re.findall(r"# Effect: pick_up_(.*)\((.*)\)$", utt)[0]
                    obj_picked_up = effects.split(",")[0]

                    # Update ongoing execution state
                    assem_st[side] = obj_picked_up
                    if obj_picked_up not in subassems:
                        singleton_gr = nx.Graph()
                        singleton_gr.add_node(obj_picked_up)
                        subassems[obj_picked_up] = singleton_gr

                    pair_invalid = False
                    if assem_st["left"] is not None and assem_st["right"] is not None:
                        # Test if a valid join can be achieved with the two subassembly
                        # objects held in each hand
                        compatible_part_pairs = {
                            frozenset([type1, type2])
                            for (type1, _), (type2, _) in assem_st["joins_remaining"]
                        }
                        sa_left = subassems[assem_st["left"]]
                        sa_right = subassems[assem_st["right"]]
                        for part_left, part_right in product(sa_left, sa_right):
                            part_left = re.findall(r"t_(.*)_\d+$", part_left)[0]
                            part_right = re.findall(r"t_(.*)_\d+$", part_right)[0]
                            part_pair = frozenset([part_left, part_right])
                            # Pass if part pair not compatible after all
                            compatible = part_pair in compatible_part_pairs
                            if not compatible: continue
                            # Pair valid if reached here
                            break
                        else:
                            # If reached here, no compatible bipartite pairing of
                            # parts exists with held subassemblies
                            pair_invalid = True

                    # Interrupt if agent has picked up an object not needed in the
                    # desired goal structure, or if agent is holding a pair of objects
                    # that should not be assembled
                    if pair_invalid:
                        # Interrupt agent and undo the last pick up (by dropping the
                        # object just picked up)
                        assem_st[side] = None       # Pick-up undone
                        response += [
                            ("generate", { "utterance": "Stop.", "pointing": {} }),
                            (f"drop_{side}", { "parameters": () })
                        ]
                        if self.player_type == "bool":
                            # Minimal help; no additional learning signal than the fact
                            # that the undone pick-up action is invalid
                            self.ongoing_demonstration = ("frag", [])
                        elif self.player_type == "demo":
                            # Language-less player type with assistance by demonstration;
                            # sample a next valid join and proceed to demonstrate
                            demo_segment = self._sample_demo_step()
                            if demo_segment is None:
                                # Terminate episode; see what return value of None means
                                # in _sample_demo_step().
                                ep_metrics["episode_discarded"] = True
                                return []
                            else:
                                self.ongoing_demonstration = ("frag", demo_segment)
                        else:
                            # Languageful player types; directly inquire the intent
                            # of the undone pick-up action
                            assert self.player_type in ["label", "full"]
                            response += [
                                ("generate", {
                                    "utterance": "What were you trying to join?",
                                    "pointing": {}
                                })
                            ]
                        
                        interrupted = True
                        ep_metrics["num_invalid_pickup"] += 1

                if utt.startswith("# Effect: drop"):
                    # Effect of drop action; no args, just update hand states
                    side = re.findall(r"# Effect: drop_(.*)\(\)$", utt)[0]
                    assem_st[side] = None

                if utt.startswith("# Effect: assemble"):
                    # Effect of assemble action; interested in closest contact points
                    direction, effects = re.findall(r"# Effect: assemble_(.*)\((.*)\)$", utt)[0]
                    src_side, _, tgt_side = direction.split("_")
                    if tgt_side == "left":
                        part_tgt, part_src, product_name = assem_st.pop("join_intent")
                    else:
                        part_src, part_tgt, product_name = assem_st.pop("join_intent")
                    part_src = assem_st["aliases"][part_src]
                    part_tgt = assem_st["aliases"][part_tgt]
                    type_src = re.findall(r"t_(.*)_\d+$", part_src)[0]
                    type_tgt = re.findall(r"t_(.*)_\d+$", part_tgt)[0]
                    valid_joins = [
                        frozenset(((type1, cp1), (type2, cp2)))
                        for (type1, cp1), (type2, cp2) in VALID_JOINS
                        if (type1 == type_src and type2 == type_tgt) or
                            (type2 == type_src and type1 == type_tgt)
                    ]

                    effects = effects.split(",")
                    num_cp_pairs = int(effects[2])
                    cp_pairs = effects[3:3+4*num_cp_pairs]
                    cp_pairs = [cp_pairs[4*i:4*(i+1)] for i in range(num_cp_pairs)]
                    cp_pairs = [
                        (tuple(cp_src.split("/")), tuple(cp_tgt.split("/")))
                        for cp_src, cp_tgt, diff_pos, diff_rot in cp_pairs
                        if float(diff_pos) + float(diff_rot) > 1.95
                            # Threshold for whether pair counts as match by sum of diffs
                    ]
                    cp_pairs = [
                        (
                            (re.findall(r"t_(.*)_\d+$", part1)[0], cp1),
                            (re.findall(r"t_(.*)_\d+$", part2)[0], cp2),
                        )
                        for (part1, cp1), (part2, cp2) in cp_pairs
                    ]
                    cp_pairs = [
                        frozenset(((type1, cp1), (type2, cp2)))
                        for (type1, cp1), (type2, cp2) in cp_pairs
                        if type1 == type_src and type2 == type_tgt
                    ]
                    valid_pairs_achieved = set(cp_pairs) & set(valid_joins)

                    # Update ongoing execution state
                    sa_left = assem_st["left"]
                    sa_right = assem_st["right"]
                    graph_left = subassems[sa_left]
                    graph_right = subassems[sa_right]

                    # Interrupt if agent has assembled a pair of objects at incorrect
                    # contact point, which is determined by checking whether top 3
                    # contact point pairs contain a valid join for this episode's
                    # goal structure
                    join_invalid = len(valid_pairs_achieved) == 0
                    if join_invalid:
                        # Interrupt agent and undo the last join (by disassembling
                        # the product just assembled, at the exact same part pair)
                        if tgt_side == "right":
                            takeaway_parts = list(graph_left)
                        else:
                            takeaway_parts = list(graph_right)
                        response += [
                            ("generate", { "utterance": "Stop.", "pointing": {} }),
                            (f"disassemble_{direction.split('_')[-1]}", {
                                "parameters": (
                                    f"str|{sa_left}", f"str|{sa_right}",
                                    f"int|{len(takeaway_parts)}",
                                ) + tuple(
                                    f"str|{part}" for part in takeaway_parts
                                )
                            })
                        ]
                        if self.player_type == "bool":
                            # Minimal help; no additional learning signal than the fact
                            # that the undone pick-up action is invalid
                            self.ongoing_demonstration = ("frag", [])
                        elif self.player_type == "demo":
                            # Language-less player type with assistance by demonstration;
                            # sample a next valid join and proceed to demonstrate
                            demo_segment = self._sample_demo_step()
                            if demo_segment is None:
                                # Terminate episode; see what return value of None means
                                # in _sample_demo_step().
                                ep_metrics["episode_discarded"] = True
                                return []
                            else:
                                self.ongoing_demonstration = ("frag", demo_segment)
                        else:
                            # Languageful player types; directly inquire the intent
                            # of the undone pick-up action
                            assert self.player_type in ["label", "full"]
                            response += [
                                ("generate", {
                                    "utterance": "What were you trying to join?",
                                    "pointing": {}
                                })
                            ]
                        interrupted = True
                        ep_metrics["num_invalid_join"] += 1
                    else:
                        # If join not invalid, update ongoing execution state
                        assert len(valid_pairs_achieved) == 1
                        pair_joined = list(valid_pairs_achieved)[0]
                        cp_src = next(
                            cp for part_type, cp in pair_joined if part_type == type_src
                        )
                        cp_tgt = next(
                            cp for part_type, cp in pair_joined if part_type == type_tgt
                        )
                        del subassems[sa_left]; del subassems[sa_right]
                        subassems[product_name] = nx.union(graph_left, graph_right)
                        subassems[product_name].add_edge(
                            part_src, part_tgt, contact={
                                part_src: (type_src, cp_src), part_tgt: (type_tgt, cp_tgt)
                            }
                        )
                        assem_st[src_side] = None
                        assem_st[tgt_side] = product_name
                        # Check the achieved join off the set
                        assem_st["joins_remaining"] -= valid_pairs_achieved

                if not interrupted:
                    # No interruption needs to be made, keep observing...
                    response.append(("generate", { "utterance": "# Observing", "pointing": {} }))

            elif utt == "I cannot find a part I need on the table." or \
                utt.startswith("I couldn't plan further"):
                # Two possible cases that need user feedback by partial demo:
                #    1) Agent has failed to ground an instance of some part
                #       type; however, due to lack of shared vocabulary, agent
                #       was not able to directly query whether an instance of
                #       the type exists on the table (which is guaranteed by
                #       design).
                #    2) Agent has failed to plan until the end product due to
                #       lack of accurate part type info on some atomic parts
                #       that are used and included in some subassembly. This
                #       happens because user feedback by demo doesn't provide
                #       immediately available part type info.

                # Provide a partial demonstration, up to a next join valid in
                # current progress
                demo_segment = self._sample_demo_step()
                if demo_segment is None:
                    # Terminate episode; see what return value of None means
                    # in _sample_demo_step().
                    ep_metrics["episode_discarded"] = True
                    return []
                else:
                    self.ongoing_demonstration = ("frag", demo_segment)

                # Notify agent that user will demonstrate a valid join
                response.append((
                    "generate",
                    {
                        "utterance": "I will demonstrate a valid join.",
                        "pointing": {}
                    }
                ))

                if utt == "I cannot find a part I need on the table.":
                    metric = "num_search_failure"
                else:
                    assert utt.startswith("I couldn't plan further")
                    metric = "num_planning_forfeiture"
                ep_metrics[metric] += 1

            elif utt.startswith("Is there a "):
                # Agent has failed to ground an instance of some part type, which
                # is guaranteed to exist on tabletop by design
                queried_type = re.findall(r"Is there a (.*)\?$", utt)[0]

                # Fetch list of matching instances and select one that hasn't
                # been used yet (i.e., still on the tabletop)
                matching_insts = [
                    f"t_{supertype}_{inst_id[1]}"
                    for (supertype, inst_id), info in sampled_parts.items()
                    if queried_type == supertype or queried_type == info["type"]
                ]
                consumed_insts = [
                    inst for sa_graph in subassems.values() for inst in sa_graph
                    if len(sa_graph) > 1
                ]
                available_insts = [
                    inst for inst in matching_insts
                    if inst not in consumed_insts or        # Not used in some subassembly, or
                        len(subassems.get(inst, {})) == 1   # Picked up once but not consumed yet
                ]
                selected_inst = random.sample(available_insts, 1)[0]

                # Selected instance may already have been picked up and held
                # in left or right hand
                if selected_inst == assem_st["left"]:
                    ent_path = f"/Student Agent/Left Hand/{selected_inst}"
                elif selected_inst == assem_st["right"]:
                    ent_path = f"/Student Agent/Right Hand/{selected_inst}"
                else:
                    ent_path = f"/{selected_inst}"
                response.append((
                    "generate",
                    {
                        "utterance": f"This is a {queried_type}.",
                        "pointing": { (0, 4): (ent_path, False) }
                            # `False`: Pass by string name instead of seg mask
                    }
                ))
                ep_metrics["num_search_failure"] += 1

            elif utt.startswith("I was trying to "):
                # Agent reported its originally intended join of two part instances
                # which is based on incorrect grounding
                for crange, inst_name in dem_refs.items():
                    inst = re.findall(r"([td])_(.*)_(\d+)$", inst_name)[0]
                    inst = (inst[1], (inst[0], int(inst[2])))
                    reported_grounding = utt[slice(*crange)]

                    # Determine correctness according to the expected knowledge level
                    # (i.e., specificity between supertype vs. subtype), as determined
                    # by the current task
                    if self.target_task == "build_truck_supertype":
                        gt_type = inst[0]
                    else:
                        gt_type = sampled_parts[inst]["type"]
                    if reported_grounding == gt_type: continue

                    # For incorrect groundings results, provide appropriate corrective
                    # feedback
                    if inst_name in subassems.get(assem_st["left"], set()):
                        ent_path = f"/Student Agent/Left Hand/*/{inst_name}"
                    elif inst_name in subassems.get(assem_st["right"], set()):
                        ent_path = f"/Student Agent/Right Hand/*/{inst_name}"
                    else:
                        ent_path = f"/*/{inst_name}"        # On tabletop
                    response += [
                        ("generate", {
                            "utterance": f"This is not a {reported_grounding}.",
                            "pointing": { (0, 4): (ent_path, False) }
                        }),
                        ("generate", {
                            "utterance": f"This is a {gt_type}.",
                            "pointing": { (0, 4): (ent_path, False) }
                        })
                    ]

                # Finally tell the agent to resume its task execution (based on
                # the modified knowledge)
                response.append(
                    ("generate", { "utterance": "Continue.", "pointing": {} })
                )

            elif utt == "OK." or utt == "Done.":
                # No further interaction needed; effectively terminates current episode
                pass

            else:
                # Cannot parse this pattern of reaction
                pass

        return response

    def _sample_ASP(
            self, goal_object, domain_knowledge, constraints, all_subtypes, num_distractors
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
            base_prg_str += "; ".join(
                f"{hypo}(O)" for _, hypo in per_hypernym
                if not (hyper in all_subtypes and hypo not in all_subtypes[hyper])
            )       # All possible subtype options that are valid (i.e., non-None)
            base_prg_str += " }1 :- "
            base_prg_str += f"{hyper}(O).\n"

        # Add rules for covering color choices for applicable parts
        base_prg_str += "1{ hasColor(O, C) : color(C) }1 :- colored_part(O).\n"
        base_prg_str += ". ".join(f"color({c})" for c in all_subtypes["color"]) + ".\n"

        for supertype, subtypes in all_subtypes.items():
            if supertype == "color": continue

            # Add integrity constraints for avoiding unavailable choices
            # (after fixing subtypes in `build_truck_supertype` task)
            base_prg_str += "1{ "
            base_prg_str += "; ".join(
                f"{subtype}(O)" for subtype in subtypes if subtype is not None
            )
            base_prg_str += " }1 :- "
            base_prg_str += f"{supertype}(O).\n"

            # Add rules for reifying selected part types
            for subtype in subtypes:
                if subtype is None: continue
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
                        piece_str = f"hasColor(O{arg_inds[0]},C0), "
                        piece_str += f"hasColor(O{arg_inds[1]},C1), "
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

        # Control the number of rules to be violated (0) by an external query atom
        base_prg_str += "#external violate_none.\n"
        base_prg_str += ":- violate_none, rule_violated(R).\n"

        # Finally add a grounded instance representing the whole truck sample
        base_prg_str += f"{goal_object}(o).\n"

        ctl = Control(["--warn=none"])
        ctl.configuration.solve.models = 0
        ctl.configuration.solver.seed = self.cfg.seed
        ctl.add("base", [], base_prg_str)
        ctl.ground([("base", [])])

        models = []
        ctl.assign_external(Function("violate_none", []), True)
        with ctl.solve(yield_=True) as solve_gen:
            for m in solve_gen:
                models.append(m.symbols(atoms=True))
                if len(models) > 10000: break       # Should be enough...

        # Randomly select a sampled model
        sampled_model = random.sample(models, 1)[0]

        # Inverse map from subtype to supertype
        all_subtypes_inv_map = {
            x: k for k, v in all_subtypes.items() for x in v if x is not None
        }

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
            # exactly one of the constraints. Sampling is on rule-basis; i.e., we
            # first select a rule to violate with resampling, then select among
            # possible rule-violating models.
            ctl.release_external(Function("violate_none", []))
            ctl.cleanup()

            # Additional program fragment for controlling the number of rules to be
            # violated (1) by an external query atom
            resample_prg_str = "#external violate_one(R) : rule(R).\n"
            resample_prg_str += ":- violate_one(R), not rule_violated(R).\n"
            resample_prg_str += ":- violate_one(_), not 1{ rule_violated(R) : rule(R) }1.\n"

            # Part groups to consider resampling as distractors; types may not
            # be mutually exclusive as with the case of fenders
            part_groups_by_type = [
                [["cabin"]],
                [["load"]],
                [["chassis_center"]],
                [["wheel"]],
                [["fl_fender", "fr_fender"]],
                [["bl_fender", "br_fender"]],
                [["fl_fender", "fr_fender", "bl_fender", "br_fender"], ["wheel"]]
            ]
            part_groups_by_obj = [
                [
                    sum([tgt_objs_grouped_by_supertype[p] for p in group], [])
                    for group in part_supertypes
                ]
                for part_supertypes in part_groups_by_type
            ]
            # Track which part groups overlap with which other groups by type
            as_flat_set = lambda groups: set.union(*[set(group) for group in groups])
            part_group_overlaps = {
                (gi, gj) for (gi, groups_gi), (gj, groups_gj) in combinations(
                    enumerate(part_groups_by_type), 2
                )
                if len(as_flat_set(groups_gi) & as_flat_set(groups_gj)) > 0
            }
            part_group_overlaps |= {(gj, gi) for gi, gj in part_group_overlaps}
            part_group_overlaps = {
                gi: {gj for _, gj in gjs} for gi, gjs in groupby(
                    sorted(part_group_overlaps), key=lambda x: x[0]
                )
            }

            # One and only one of the part groups may be resampled as distractor
            # to violate one and only one constraint. If a group is selected to be
            # resampled, none of the already sampled spec should be sampled again.
            # If a group is not selected and thus specified to be 'preserved', all
            # of the already sampled specs should be reproduced, except those belonging
            # to the resampled group
            resample_prg_str += "#external resample_group(G) : part_group(G).\n"
            resample_prg_str += "{ reproduce_slot_with(S,O) } :- may_fill(O,S).\n"
            resample_prg_str += "resample_slot(S) :- " \
                "part_slot(S), slot_group(S,G), resample_group(G).\n"
            resample_prg_str += "reproduce_slot(S) :- " \
                "part_slot(S), not resample_slot(S).\n"
            resample_prg_str += ":- resample_slot(S), may_fill(_,S).\n"
            resample_prg_str += \
                ":- reproduce_slot(S), #count { O : reproduce_slot_with(S,O) } != 1.\n"
            resample_prg_str += \
                ":- reproduce_slot_with(S1,O), reproduce_slot_with(S2,O), S1 != S2.\n"

            # Recognizing each 'part slot' that could be replaced with a part
            # with newly sampled spec; each slot is identified with the fname
            sample_slots = {
                obj for groups in part_groups_by_obj
                for group in groups
                for obj in group
            }
            for slot in sample_slots:
                resample_prg_str += f"part_slot({slot}).\n"

            # Encoding how the identical resampling would be achieved based on the specs
            # of the parts already sampled as 'ground-truth'
            part_specs = {
                obj: (tgt_parts_by_fname[obj][0], tgt_colors_by_fname.get(obj))
                for groups in part_groups_by_obj
                for group in groups
                for obj in group
            }
            for gi, groups in enumerate(part_groups_by_obj):
                resample_prg_str += f"part_group({gi}).\n"
                for group in groups:
                    for obj in group:
                        resample_prg_str += f"slot_group({obj},{gi}).\n"
                        part_type_str = f"{part_specs[obj][0]}(O)"
                        part_color_str = "" if part_specs[obj][1] is None \
                            else f", hasColor(O,{part_specs[obj][1]})"
                        resample_prg_str += \
                            f"may_fill(O,{obj}) :- {part_type_str}{part_color_str}.\n"

            ctl.add("resample", [], resample_prg_str)
            ctl.ground([("resample", [])])

            # Remove already sampled parts in this group so that exact same part
            # specs (part type and color if applicable) cannot be sampled again,
            # while fixing all other parts as identical; implement as additional
            # clingo program fragment
            sampled_distractor_groups = []
            sampled_distractors = {}

            # Get list of constraints that are in action in the sampled model so
            # that we can sample from them one by one
            model_predicates = {atm.name for atm in sampled_model}
            relevant_constraint_pool = [
                ci for ci, (_, scope_class, _, can_violate) in enumerate(constraints)
                if can_violate and scope_class in model_predicates
            ]
            part_group_pool = list(range(len(part_groups_by_type)))
            while len(sampled_distractor_groups) < num_distractors:
                if len(relevant_constraint_pool) == 0:
                    # No more distractor sampling possible, terminate here
                    break

                ci = random.sample(relevant_constraint_pool, 1)[0]
                relevant_constraint_pool.remove(ci)

                # Specify the sole constraint to violate
                ctl.assign_external(Function("violate_one", [Number(ci)]), True)

                # Test every part group to see if a distractor (group) can be
                # sampled to violate only the particular constraint selected
                samples_per_group = {}
                for gi in part_group_pool:
                    ctl.assign_external(Function("resample_group", [Number(gi)]), True)

                    # Obtain up to 1k samples; there may be none, in which case
                    # no resampling from the part group can violate the constraint
                    samples = set()
                    with ctl.solve(yield_=True) as solve_gen:
                        models_inspected = 0
                        for m in solve_gen:
                            if models_inspected > 100: break
                            models_inspected += 1

                            # Extract the atomic parts and colors (where applicable)
                            # sampled; part type first, then color
                            types_to_collect = sum(part_groups_by_type[gi], [])
                            sample = {}
                            for atm in m.symbols(atoms=True):
                                if atm.name != "atomic": continue
                                fname = atm.arguments[0].name
                                part_subtype = atm.arguments[1].name
                                part_supertype = all_subtypes_inv_map[part_subtype]
                                if part_supertype not in types_to_collect: continue
                                sample[fname] = (part_subtype, None)
                            for atm in m.symbols(atoms=True):
                                if atm.name != "hasColor": continue
                                fname = atm.arguments[0].name
                                color = atm.arguments[1].name
                                if fname not in sample: continue
                                sample[fname] = (sample[fname][0], color)
                            samples.add(frozenset(
                                (fname, part_subtype, color)
                                for fname, (part_subtype, color) in sample.items()
                            ))

                    if len(samples) > 0:
                        samples_per_group[gi] = samples

                    ctl.assign_external(Function("resample_group", [Number(gi)]), False)

                # Undo the sole rule violation constraint
                ctl.release_external(Function("violate_one", [Number(ci)]))
                ctl.cleanup()

                if len(samples_per_group) > 0:
                    # Valid samples found. First sample a part group, then remove
                    # self and any overlapping part groups from the pool
                    gi_sampled = random.sample(sorted(samples_per_group), 1)[0]
                    overlapping_groups = part_group_overlaps.get(gi_sampled, set())
                    part_group_pool = [
                        gi for gi in part_group_pool
                        if gi != gi_sampled and gi not in overlapping_groups
                    ]
                    sampled_distractor_groups += part_groups_by_type[gi_sampled]
                    # Finally, randomly select a sample for the group to use as
                    # distractor
                    parts_sampled = random.sample(samples_per_group[gi_sampled], 1)[0]
                    sampled_distractors.update({
                        fname: (subtype, color, ci)
                        for fname, subtype, color in parts_sampled
                    })

            dtr_objs_grouped_by_supertype = defaultdict(list)
            for obj, (part_subtype, _, _) in sampled_distractors.items():
                part_supertype = all_subtypes_inv_map[part_subtype]
                dtr_objs_grouped_by_supertype[part_supertype].append(obj)
            dtr_objs_grouped_by_supertype = dict(dtr_objs_grouped_by_supertype)

        # Organize into final return value dict
        sampled_inits = {}
        for obj, (part_subtype, part_supertype) in tgt_parts_by_fname.items():
            oi = tgt_objs_grouped_by_supertype[part_supertype].index(obj)
            subtype_ind = all_subtypes[part_supertype].index(part_subtype)
            color_ind = all_subtypes["color"].index(tgt_colors_by_fname[obj]) \
                if obj in tgt_colors_by_fname else None

            sampled_inits[f"{part_supertype}/t{oi}/type"] = subtype_ind
            if color_ind is not None:
                sampled_inits[f"{part_supertype}/t{oi}/color"] = color_ind

        if num_distractors > 0:
            i = 0; num_added_distractor_groups = 0
            while num_added_distractor_groups < num_distractors:
                if i >= len(sampled_distractor_groups): break       # Seen all
                dtr_group = sampled_distractor_groups[i]
                i += 1

                for part_supertype in dtr_group:
                    for oi, obj in enumerate(dtr_objs_grouped_by_supertype[part_supertype]):
                        part_subtype, color, ci = sampled_distractors[obj]
                        violated_constraint = constraints[ci]
                        subtype_ind = all_subtypes[part_supertype].index(part_subtype)
                        color_ind = all_subtypes["color"].index(color) \
                            if color is not None else None

                        sampled_inits[f"{part_supertype}/d{oi}/type"] = subtype_ind
                        if color_ind is not None:
                            sampled_inits[f"{part_supertype}/d{oi}/color"] = color_ind
                        sampled_inits[f"{part_supertype}/d{oi}/viol"] = violated_constraint

                num_added_distractor_groups += 1

        return sampled_inits

    def _sample_demo_plan(self):
        """
        Helper method factored out for sampling a valid assembly plan that builds
        a valid instance of truck structure. Since we (and the teacher) already know
        the desired structure, let's specify a partial order plan by a set of subgoals
        with precedence constraints and sample a valid instantiation accordingly.
        """
        sampled_parts = copy.deepcopy(self.current_episode_record["sampled_parts"])

        # Represent partial plans with (action sequence, preconditions, name of resultant
        # subassembly) tuples
        partial_plan = [
            # Front-left fender-wheel-bolt unit
            (
                [
                    ("pick_up_left", ("fl_fender", "GT")),
                    ("pick_up_right", ("wheel", "GT")),
                    ("assemble_right_to_left", ("fl_fw", "fl_fender/wheel", "wheel/bolt")),
                    ("pick_up_right", ("bolt", "GT")),
                    ("assemble_right_to_left", ("fl_fw_unit", "fl_fender/wheel", "bolt/bolt")),
                    ("drop_left", ())
                ],
                [],
                "fl_fw_unit"
            ),

            # Front-right fender-wheel-bolt unit
            (
                [
                    ("pick_up_left", ("fr_fender", "GT")),
                    ("pick_up_right", ("wheel", "GT")),
                    ("assemble_right_to_left", ("fr_fw", "fr_fender/wheel", "wheel/bolt")),
                    ("pick_up_right", ("bolt", "GT")),
                    ("assemble_right_to_left", ("fr_fw_unit", "fr_fender/wheel", "bolt/bolt")),
                    ("drop_left", ())
                ],
                [],
                "fr_fw_unit"
            ),

            # Back-left fender-wheel-bolt unit
            (
                [
                    ("pick_up_left", ("bl_fender", "GT")),
                    ("pick_up_right", ("wheel", "GT")),
                    ("assemble_right_to_left", ("bl_fw", "bl_fender/wheel", "wheel/bolt")),
                    ("pick_up_right", ("bolt", "GT")),
                    ("assemble_right_to_left", ("bl_fw_unit", "bl_fender/wheel", "bolt/bolt")),
                    ("drop_left", ())
                ],
                [],
                "bl_fw_unit"
            ),

            # Back-right fender-wheel-bolt unit
            (
                [
                    ("pick_up_left", ("br_fender", "GT")),
                    ("pick_up_right", ("wheel", "GT")),
                    ("assemble_right_to_left", ("br_fw", "br_fender/wheel", "wheel/bolt")),
                    ("pick_up_right", ("bolt", "GT")),
                    ("assemble_right_to_left", ("br_fw_unit", "br_fender/wheel", "bolt/bolt")),
                    ("drop_left", ())
                ],
                [],
                "br_fw_unit"
            ),

            # Truck front
            (
                [
                    ("pick_up_left", ("chassis_front", "GT")),
                    ("pick_up_right", ("cabin", "GT")),
                    ("assemble_right_to_left", ("cf0", "chassis_front/cabin1", "cabin/front1")),
                    ("pick_up_right", ("bolt", "GT")),
                    ("assemble_right_to_left", ("cf1", "chassis_front/cabin1", "bolt/bolt")),
                    ("pick_up_right", ("bolt", "GT")),
                    ("assemble_right_to_left", ("cf2", "chassis_front/cabin2", "bolt/bolt")),
                    ("pick_up_right", ("fl_fw_unit", "SA")),
                    ("assemble_right_to_left", ("cfl", "chassis_front/lfw", "fl_fender/wheel")),
                    ("pick_up_right", ("fr_fw_unit", "SA")),
                    ("assemble_right_to_left", ("truck_front", "chassis_front/rfw", "fr_fender/wheel")),
                    ("drop_left", ())
                ],
                ["fl_fw_unit", "fr_fw_unit"],
                "truck_front"
            ),

            # Truck back
            (
                [
                    ("pick_up_left", ("chassis_back", "GT")),
                    ("pick_up_right", ("load", "GT")),
                    ("assemble_right_to_left", ("lb0", "chassis_back/load", "load/back")),
                    ("pick_up_right", ("bolt", "GT")),
                    ("assemble_right_to_left", ("lb1", "chassis_back/load", "bolt/bolt")),
                    ("pick_up_right", ("bl_fw_unit", "SA")),
                    ("assemble_right_to_left", ("lb2", "chassis_back/lfw", "bl_fender/wheel")),
                    ("pick_up_right", ("br_fw_unit", "SA")),
                    ("assemble_right_to_left", ("truck_back", "chassis_back/rfw", "br_fender/wheel")),
                    ("drop_left", ())
                ],
                ["bl_fw_unit", "br_fw_unit"],
                "truck_back"
            ),

            # Whole truck
            (
                [
                    ("pick_up_left", ("truck_front", "SA")),
                    ("pick_up_right", ("chassis_center", "GT")),
                    ("assemble_right_to_left", ("tfc0", "chassis_front/center", "chassis_center/front")),
                    ("pick_up_right", ("bolt", "GT")),
                    ("assemble_right_to_left", ("tfc1", "chassis_front/center", "bolt/bolt")),
                    ("pick_up_right", ("truck_back", "SA")),
                    ("assemble_right_to_left", ("tfcb", "chassis_center/back", "chassis_back/center")),
                    ("pick_up_right", ("bolt", "GT")),
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
                obj_id = (f"{result[:2]}_fender", ("t", 0))
                large_wheel_needed = sampled_parts[obj_id]["type"].startswith("large")
            else:
                large_wheel_needed = None

            # Each object argument of pick_up_left/Right arguments in the action
            # sequence is somewhat lifted, as they only specify type; 'ground'
            # them by sampling from existing part instances, then append to fully
            # instantiated plan
            for action in act_seq:
                if action[0].startswith("pick_up") and action[1][0] in atomic_supertypes:
                    compatible_parts = []
                    for instance, data in sampled_parts.items():
                        if instance[0] != action[1][0]: continue
                        if instance[1][0] != "t": continue      # Don't use distractors
                        if action[1][0] == "wheel" and large_wheel_needed is not None:
                            if data["type"] != "large_wheel" and large_wheel_needed: continue
                            if data["type"] == "large_wheel" and not large_wheel_needed: continue

                        compatible_parts.append(instance)

                    inst_supertype, inst_id = random.sample(compatible_parts, 1)[0]
                    grounded_action = (
                        action[0],
                        (f"{inst_id[0]}_{inst_supertype}_{inst_id[1]}", action[1][1])
                    )
                    subtype = sampled_parts.pop((inst_supertype, inst_id))["type"]

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
                            (
                                f"inspect_{hand}",
                                (f"{inst_id[0]}_{inst_supertype}_{inst_id[1]}", i)
                            )
                            for i in range(24+1)
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
        assert len([
            inst_id for _, inst_id in sampled_parts if inst_id[0] == "t"
        ]) == 0

        return plan

    def _sample_demo_step(self):
        """
        Helper method factored out for sampling a valid assembly plan segment up to
        a very next join that is valid and achievable from current progress status.
        Use the planning procedure used by the agent to obtain a full valid plan
        until completion, then return the segment until the first assemble action.
        """
        # Shortcut vars
        sampled_parts = self.current_episode_record["sampled_parts"]
        assem_st = self.current_episode_record["assembly_state"]
        subassems = assem_st["subassemblies"]

        # Figure out a valid joining plan in the form of partial order plan,
        # using the same procedure adopted by the agent. First prepare
        # procedure input parameters from teacher's knowledge of sampled
        # instances and current progress...

        # Tabulate and assign arbitrary indices to relevant entities (parts,
        # contact points)
        all_nodes = [f"t_{supertype}_{inst[1]}" for supertype, inst in sampled_parts]
        part_names = dict(enumerate(set(supertype for supertype, _ in sampled_parts)))
        cp_names = set("/".join(cp) for cps in VALID_JOINS for cp in cps)
        cp_names = dict(enumerate(cp_names))
        part_names_inv = { v: k for k, v in part_names.items() }
        cp_names_inv = { v: k for k, v in cp_names.items() }

        # Pick a valid target structure while accounting for the current progress
        inst_pool = {
            supertype: {
                f"{inst[0]}_{supertype}_{inst[1]}"
                for _, inst in insts if inst[0] == "t"
            }
            for supertype, insts in groupby(
                sorted(sampled_parts, key=lambda x: x[0]), key=lambda x: x[0]
            )
        }
        connection_graph = nx.Graph(); atomic_node_concs = {}; contacts = {}
        connection_status = {}; node_unifications = {}
        # Here we are leveraging the very specific domain knowledge that wheels
        # and bolts in the list of valid joins are all distinct and instances
        # of other types are the same ones (one and only for each)
        remaining_joins = copy.deepcopy(VALID_JOINS)
        for sa, sa_graph in subassems.items():
            # Process existing joins first so that parameters will comply
            if len(sa_graph) == 1: continue     # Ignore atomic singletons
            for u, v, cps_uv in sa_graph.edges(data="contact"):
                connection_graph.add_edge(u, v)
                type_u = re.findall(r"t_(.*)_\d+$", u)[0]
                type_v = re.findall(r"t_(.*)_\d+$", v)[0]
                if type_u in ["wheel", "bolt"]:
                    inst_pool[type_u].remove(u)
                if type_v in ["wheel", "bolt"]:
                    inst_pool[type_v].remove(v)
                atomic_node_concs[u] = part_names_inv[type_u]
                atomic_node_concs[v] = part_names_inv[type_v]
                cp_u = cps_uv[u]; cp_v = cps_uv[v]
                if (cp_u, cp_v) in remaining_joins:
                    remaining_joins.remove((cp_u, cp_v))
                    edge = (u, v); contact = (cp_u, cp_v)
                elif (cp_v, cp_u) in remaining_joins:
                    remaining_joins.remove((cp_v, cp_u))
                    edge = (v, u); contact = (cp_v, cp_u)
                else:
                    # This really shouldn't happen... but I've witnessed it
                    # happened once. Probably an invalid join seeped through
                    # the stochastic perturbation wall implemented in Unity???
                    # Aways, if this is ever to happen, log and return null
                    # step, effectively aborting this episode
                    log_msg = "Existing join "
                    log_msg += f"{type_u}/{cp_u}~{type_v}/{cp_v} invalid, "
                    log_msg += "abort episode"
                    logger.info(log_msg)
                    return None
                contacts[edge] = (
                    "p_" + str(cp_names_inv[contact[0][0] + "/" + contact[0][1]]),
                    "p_" + str(cp_names_inv[contact[1][0] + "/" + contact[1][1]])
                )
                node_unifications[f"{sa}_{u}"] = u
                node_unifications[f"{sa}_{v}"] = v
            connection_status[sa] = sa_graph

        if len(remaining_joins) == 0:
            return []           # Task already finished

        for (type1, cp1), (type2, cp2) in remaining_joins:
            # Remaining joins to make; randomly select remaining parts where
            # there are more than one ways/orders to sample them (wheels and
            # bolts, in particular)
            if type1 in ["wheel", "bolt"]:
                inst1 = inst_pool[type1].pop()
            else:
                inst1 = list(inst_pool[type1])[0]
            if type2 in ["wheel", "bolt"]:
                inst2 = inst_pool[type2].pop()
            else:
                inst2 = list(inst_pool[type2])[0]
            connection_graph.add_edge(inst1, inst2)
            atomic_node_concs[inst1] = part_names_inv[type1]
            atomic_node_concs[inst2] = part_names_inv[type2]
            contacts[(inst1, inst2)] = (
                "p_" + str(cp_names_inv[type1 + "/" + cp1]),
                "p_" + str(cp_names_inv[type2 + "/" + cp2])
            )

        # Singleton compression sequence containing all nodes
        compression_sequence = [("truck", all_nodes)]
        # Now run the procedure, just the join tree is needed
        join_tree, _, _, _ = _divide_and_conquer(
            compression_sequence, connection_graph, {},
            atomic_node_concs, contacts, part_names, cp_names,
            (connection_status, node_unifications), set(), set(),
            self.cfg.paths.assets_dir, self.cfg.seed
        )
        if join_tree is None:
            # If no valid join tree is identified despite the teacher's full
            # domain knowledge, this implies agent somehow 'trapped itself'
            # into hard-to-recover deadend (due to collision constraint)
            # stemming from misclassification of an atomic part included in
            # an existing subassembly. We could consider backtracking and 
            # disassembling until full assembly is feasible again, but it's
            # gonna induce some headache... Let's just discard this episode
            # altogether and mark this episode as botched, for such occasions
            # happen quite rarely, which is fortunate for us.
            logger.info("Irreconcilable dead end reached, abort episode")
            return None

        # Integer index for the resultant subassembly, obtained from list of
        # any existing subassemblies; ensure the newly assigned index doesn't
        # overlap with any of the previously assigned ones
        sa_ind = max([
            int(re.findall(r"s(\d+)", sa)[0])
            for sa, sa_graph in subassems.items()
            if len(sa_graph) > 1
        ] + [-1]) + 1           # [-1] to ensure max value is always obtained

        # Now select the next available join to come up with the demo
        # fragment up until the join as return value
        demo_frag = []
        available_joins = [
            n for n in join_tree if len(nx.ancestors(join_tree, n))==2
        ]
        available_joins = [
            (join_res,) + tuple(n for n, _ in join_tree.in_edges(join_res))
            for join_res in available_joins
        ]

        current_held = (assem_st["left"], assem_st["right"])
        if assem_st["left"] is not None or assem_st["right"] is not None:
            # If holding anything, see if any of the available joins can be
            # demonstrated without dropping them
            available_without_dropping = [
                (join_res, n1, n2) for join_res, n1, n2 in available_joins
                if set(current_held) - {None} <= {n1, n2}
            ]
            if len(available_without_dropping) > 0:
                # Narrow down the options to the currently possible joins
                available_joins = available_without_dropping
            else:
                # No such joins possible, drop currently handheld objects
                # and just randomly pick a join
                if assem_st["left"] is not None:
                    assem_st["left"] = None
                    demo_frag.append(("drop_left", ()))
                if assem_st["right"] is not None:
                    assem_st["right"] = None
                    demo_frag.append(("drop_right", ()))

        # Specifying objects to be held (or already held) in left and right
        # hand, respectively
        next_join = available_joins[0]
        if assem_st["left"] is None and assem_st["right"] is None:
            # Both of agent's hands are currently empty, user can arbitrarily
            # choose a valid join to demonstrate among remaining ones
            join_res, obj_left, obj_right = next_join

        elif assem_st["left"] is not None and assem_st["right"] is not None:
            # Both of agent's hands are currently occupied, and the next
            # join to demonstrate can be achieved immediately
            join_res = next_join[0]
            obj_left = assem_st["left"]
            obj_right = assem_st["right"]

        else:
            # Either of agent's hands is empty and need to pick up the other
            # object
            join_res = next_join[0]
            if assem_st["left"] in next_join:
                obj_left = assem_st["left"]
                obj_right = next_join[2] if obj_left == next_join[1] else next_join[1]
            else:
                assert assem_st["right"] in next_join
                obj_right = assem_st["right"]
                obj_left = next_join[2] if obj_right == next_join[1] else next_join[1]

        # Pick up each object with left and right hand resp. as needed
        if assem_st["left"] is None:
            label_key_left = "SA" if len(subassems.get(obj_left, {})) > 1 else "GT"
            demo_frag.append(("pick_up_left", (obj_left, label_key_left)))
        if assem_st["right"] is None:
            label_key_right = "SA" if len(subassems.get(obj_right, {})) > 1 else "GT"
            demo_frag.append(("pick_up_right", (obj_right, label_key_right)))

        # For determining the assembly direction, exploit the valid joins
        # stored in `VALID_JOINS``, hence those in `contacts`, which are all
        # listed in tgt <- src direction
        contact_left = join_tree.edges[(obj_left, join_res)]["join_by"]
        contact_right = join_tree.edges[(obj_right, join_res)]["join_by"]
        if (contact_left, contact_right) in contacts:
            obj_tgt = obj_left
            contact_tgt, contact_src = contact_left, contact_right
        else:
            assert (contact_right, contact_left) in contacts
            obj_tgt = obj_right
            contact_tgt, contact_src = contact_right, contact_left
        cp_tgt, cp_src = contacts[(contact_tgt, contact_src)]
        cp_tgt = cp_names[int(re.findall(r"p_(\d+)$", cp_tgt)[0])]
        cp_src = cp_names[int(re.findall(r"p_(\d+)$", cp_src)[0])]

        # Join the objects at the correct connection site (i.e., part and
        # contact point) in the src->tgt direction
        if obj_left == obj_tgt:
            join_direction = "right_to_left"
            cp_left, cp_right = cp_tgt, cp_src
        else:
            join_direction = "left_to_right"
            cp_left, cp_right = cp_src, cp_tgt

        demo_frag.append(
            (f"assemble_{join_direction}", (f"s{sa_ind}", cp_left, cp_right))
        )

        return demo_frag
