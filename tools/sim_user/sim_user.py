"""
Simulated user which takes part in dialogue with rule-based pattern matching
-- no cognitive architecture ongoing within the user
"""
import re
import random
from collections import defaultdict
from itertools import combinations, groupby

import yaml
import inflect
import numpy as np
import networkx as nx
from clingo import Control, Function, Number

from python.itl.vision.utils import mask_iou


singularize = inflect.engine().singular_noun
pluralize = inflect.engine().plural

MATCH_THRES = 0.9

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
        self.current_episode_record = {}
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

        # Return sampled parts to pass as environment parameters
        return sampled_inits

    def initiate_dialogue(self):
        """
        Prepare an opening line for starting a new thread of dialogue, based on
        the current queue of target concepts (unprompted dialogue starter, as
        opposed to self.react method below, which is invoked only when prompted
        by an agent reaction)
        """
        # Dequeue from current episode's target concept queue
        self.current_target_concept = self.current_queue.pop(0)
        gameObject_path, _ = self.current_target_concept

        if gameObject_path is None:
            gameObject_handle = "/truck"
            part_type = None
        else:
            part_type, part_subtype = gameObject_path
            part_subtype_cc = "".join(
                tok.capitalize() if i>0 else tok
                for i, tok in enumerate(part_subtype.split(" "))
            )           # camelCase
            gameObject_handle = f"/truck/{part_type}/{part_type}_{part_subtype_cc}"

        if self.target_task == "prior_supertypes":
            return [{
                "utterance": "What is this?",
                "pointing": { (8, 12): gameObject_handle }
            }]
        elif self.concept_set == "prior_parts":
            return [{
                "utterance": f"What kind of {part_type} is this?",
                "pointing": { (17+len(part_type), 17+len(part_type)+4): gameObject_handle }
            }]
        elif self.concept_set == "single_fourway" or self.concept_set == "double_fiveway":
            return [{
                "utterance": "What kind of truck is this?",
                "pointing": { (22, 26): gameObject_handle }
            }]
        else:
            raise ValueError

    def react(self, agent_reactions):
        """ Rule-based pattern matching for handling agent responses """
        gameObject_path, string_name = self.current_target_concept
        if gameObject_path is None:
            gameObject_handle = "/truck"
        else:
            part_type, part_subtype = gameObject_path
            part_subtype_cc = "".join(
                tok.capitalize() if i>0 else tok
                for i, tok in enumerate(part_subtype.split(" "))
            )           # camelCase
            gameObject_handle = f"/truck/{part_type}/{part_type}_{part_subtype_cc}"

        response = []
        for utt, dem_refs in agent_reactions:
            if utt == "I am not sure.":
                # Agent answered it doesn't have any clue what the concept instance
                # is; provide correct label, even if taking minimalist strategy (after
                # all, learning cannot take place if we don't provide any)
                self.current_episode_record[string_name] = { "answer": None }

                if self.agent_test_mode:
                    # No feedback needed if agent running in test mode
                    if len(self.current_queue) > 0:
                        # Remaining target concepts to test and teach
                        response += self.initiate_dialogue()
                    else:
                        # No further interaction needed
                        pass
                else:
                    response.append({
                        "utterance": f"This is a {string_name}.",
                        "pointing": { (0, 4): gameObject_handle }
                    })

            elif utt.startswith("This is"):
                # Agent provided an answer what the instance is
                answer_content = re.findall(r"This is a (.*)\.$", utt)[0]
                self.current_episode_record[string_name] = { "answer": answer_content }

                if self.agent_test_mode:
                    # No feedback needed if agent running in test mode
                    if len(self.current_queue) > 0:
                        # Remaining target concepts to test and teach
                        response += self.initiate_dialogue()
                    else:
                        # No further interaction needed
                        pass
                else:
                    if string_name == answer_content:
                        # Correct answer

                        # # Teacher would acknowledge by saying "Correct" in the previous
                        # # project. but I think we can skip that for simplicity
                        # responses.append({
                        #     "utterances": ["Correct."],
                        #     "pointing": [{}]
                        # })

                        if len(self.current_queue) > 0:
                            # Remaining target concepts to test and teach
                            response += self.initiate_dialogue()
                    else:
                        # Incorrect answer; reaction branches here depending on teacher's
                        # strategy

                        # At all feedback level, let the agent know the answer is incorrect
                        response.append({
                            "utterance": f"This is not a {answer_content}.",
                            "pointing": { (0, 4): gameObject_handle }
                        })

                        # Correct label additionally provided if teacher strategy is 'greater'
                        # than [minimal feedback] or the concept hasn't ever been taught
                        taught_concepts = set(
                            conc for epi in self.episode_records for conc in epi
                        )
                        is_novel_concept = string_name not in taught_concepts
                        if self.strat_feedback != "minHelp" or is_novel_concept:
                            response.append({
                                "utterance": f"This is a {string_name}.",
                                "pointing": { (0, 4): gameObject_handle }
                            })

                        # Ask for an explanation why the agent have the incorrect answer,
                        # if the strategy is to take interest
                        if self.strat_feedback.startswith("maxHelpExpl"):
                            response.append({
                                "utterance": f"Why did you think this is a {answer_content}?",
                                "pointing": { (18, 22): gameObject_handle }
                            })

            elif utt == "I cannot explain.":
                # Agent couldn't provide any verbal explanations for its previous answer;
                # always provide generic rules about concept differences between agent
                # answer vs. ground truth
                assert self.strat_feedback.startswith("maxHelpExpl")

                # Log (lack of) cited reasons
                self.current_episode_record[string_name]["reason"] = "noExpl"

                conc_gt = string_name
                conc_ans = self.current_episode_record[string_name]["answer"]
                conc_diffs = _compute_concept_differences(
                    self.domain_knowledge, conc_gt, conc_ans
                )
                response += _properties_to_nl(conc_diffs)

            elif utt.startswith("Because I thought "):
                # Agent provided explanation for its previous answer; expecting the agent's
                # recognition of truck parts (or failure thereof), which it deemed relevant
                # to its reasoning
                reasons = re.findall(r"^Because I thought (.*).$", utt)[0].split(", and ")
                reasons = reasons[0].split(", ") + reasons[1:]
                reasons = [_parse_nl_reason(r) for r in reasons]

                # Log types of cited reasons
                self.current_episode_record[string_name]["reason"] = \
                    "|".join({reason_type for _, reason_type in reasons})

                dem_refs = sorted(dem_refs.items())

                # Prepare concept differences, needed for checking whether the agent's
                # knowledge (indirectly revealed through its explanations) about the
                # concepts is incomplete
                conc_gt = string_name
                conc_ans = self.current_episode_record[string_name]["answer"]
                conc_diffs = _compute_concept_differences(
                    self.domain_knowledge, conc_gt, conc_ans
                )

                # Part domain knowledge info dicts for ground truth and agent answer
                gt_parts_info = self.domain_knowledge[conc_gt]["parts"]
                ans_parts_info = self.domain_knowledge[conc_ans]["parts"]

                # Part types that actually play role for distinguishing conc_gt vs. conc_ans
                distinguishing_part_types = {p for _, p in conc_diffs["parts"]}

                has_diff_knowledge = False
                for (part, reason_type), (_, reference) in zip(reasons, dem_refs):
                    # For each part statement given as explanation, provide feedback
                    # on whether the agent's judgement was correct (will be most likely
                    # wrong... I suppose for now)

                    # Variable storing currently active pending connective
                    pending_connective = None

                    # Checking if mentioned part subtype is a property of the ground
                    # truth concept & the agent answer concept
                    part_type = self.taxonomy_knowledge[part]
                    part_gt_prop = part_type in gt_parts_info and \
                        gt_parts_info[part_type] == part
                    part_ans_prop = part_type in ans_parts_info and \
                        ans_parts_info[part_type] == part

                    # Current experiment design is such that all parts of the ground
                    # truth concept have their masks visible in the scene, not occluded
                    # or anything
                    assert part_type in self.current_gt_masks

                    if part in distinguishing_part_types:
                        # The part type cited in the reason bears relevance as to
                        # distinguishing the agent answer vs. ground truth concepts.
                        has_diff_knowledge = True

                    if isinstance(reference, str):
                        # Reference by Unity GameObject string name; test by the object
                        # name whether the agent's judgement was correct
                        # (Will reach here if the region localized by the agent's vision
                        # module has sufficiently high box-IoU with one of the existing
                        # real object... Will it ever, though?)
                        raise NotImplementedError

                    else:
                        # Reference by raw mask bitmap
                        assert isinstance(reference, np.ndarray)

                        if reason_type == "existence":
                            # Agent made an explicit part claim as sufficient explanation,
                            # which may or may not be true

                            # Sanity check: agent must be believing the mentioned part
                            # is a (distinguishing) property of the concept it provided
                            # as answer
                            assert part_ans_prop

                            if part_gt_prop:
                                # Mentioned part actually exists somewhere, obtain IoU
                                gt_mask = self.current_gt_masks[part_type]
                                match_score = mask_iou([gt_mask], [reference])[0][0]
                            else:
                                # Mentioned part absent, consider match score zero
                                match_score = 0

                            if match_score > MATCH_THRES:
                                # The part claim itself is true
                                _add_after_redundancy_check(response, {
                                    "utterance": f"It's true this is a {part}.",
                                    "pointing": { (10, 14): reference.reshape(-1).tolist() }
                                })
                                pending_connective = "but"
                            else:
                                # The part claim is false, need correction
                                _add_after_redundancy_check(response, {
                                    "utterance": f"This is not a {part}.",
                                    "pointing": { (0, 4): reference.reshape(-1).tolist() }
                                })
                                pending_connective = "and"

                            # If the mentioned part is also a property of the ground truth,
                            # and thus doesn't serve as distinguishing property, need to
                            # correct that
                            if part_gt_prop:
                                conc_gt_pl = pluralize(conc_gt).capitalize()
                                part_pl = pluralize(part)
                                _add_after_redundancy_check(response, _prepend_with_connective({
                                    "utterance": f"{conc_gt_pl} have {part_pl}, too.",
                                    "pointing": {}
                                }, pending_connective))
                                pending_connective = None

                        elif reason_type == "uncertain" or reason_type == "absence":
                            # Agent made a counterfactual excuse

                            # Sanity check: agent must be believing the mentioned part
                            # is a (distinguishing) property of the ground truth concept
                            assert part_gt_prop

                            if reason_type == "uncertain":
                                # Agent exhibited uncertainty on part identity as counterfactual
                                # excuse; the referenced entity may or may not be an instance of
                                # the part subtype

                                # Mentioned part always exists, obtain IoU
                                gt_mask = self.current_gt_masks[part_type]
                                match_score = mask_iou([gt_mask], [reference])[0][0]

                                if match_score > MATCH_THRES:
                                    # Sufficient overlap, match good enough; endorse the
                                    # proposed mask reference as correct
                                    _add_after_redundancy_check(response, {
                                        "utterance": f"This is a {part}.",
                                        "pointing": { (0, 4): reference.reshape(-1).tolist() }
                                    })
                                    pending_connective = "but"
                                else:
                                    # Not enough overlap, bad match; reject the suspected
                                    # part reference and provide correct mask
                                    _add_after_redundancy_check(response, {
                                        "utterance": f"This is not a {part}.",
                                        "pointing": { (0, 4): reference.reshape(-1).tolist() }
                                    })
                                    _add_after_redundancy_check(response, {
                                        "utterance": f"This is a {part}.",
                                        "pointing": { (0, 4): gt_mask.reshape(-1).tolist() }
                                    })
                                    pending_connective = "and"

                            elif reason_type == "absence":
                                # Agent made an explicit absence claim as counterfactual excuse,
                                # which may or may not be true
                                gt_mask = self.current_gt_masks[part_type]

                                # The absence claim is false; point to the ground-truth
                                _add_after_redundancy_check(response, {
                                    "utterance": f"This is a {part}.",
                                    "pointing": { (0, 4): gt_mask.reshape(-1).tolist() }
                                })
                                pending_connective = "and"

                            # If the mentioned part is also a property of the incorrect agent
                            # answer, and thus doesn't serve as distinguishing property, need
                            # to correct that
                            if part_ans_prop:
                                conc_ans_pl = pluralize(conc_ans).capitalize()
                                part_pl = pluralize(part)
                                _add_after_redundancy_check(response, _prepend_with_connective({
                                    "utterance": f"{conc_ans_pl} have {part_pl}, too.",
                                    "pointing": {}
                                }, pending_connective))
                                pending_connective = None

                        else:
                            # Don't know how to handle other reason types
                            raise NotImplementedError

                # If imperfect agent knowledge is revealed, provide appropriate
                # feedback regarding generic differences between conc_gt vs. conc_ans
                if not has_diff_knowledge:
                    conc_diff_feedback = [
                        _prepend_with_connective(fb, pending_connective) if i==0 else fb
                        for i, fb in enumerate(_properties_to_nl(conc_diffs))
                    ]
                    response += conc_diff_feedback
                    pending_connective = None

            elif utt.startswith("How are") and utt.endswith("different?"):
                # Agent requested generic differences between two similar concepts
                assert self.strat_feedback.startswith("maxHelp")

                # if contrast_concepts in self.taught_diffs:
                #     # Concept diffs requested again; do something? This would 'annoy'
                #     # the user if keeps happening
                #     ...

                # Extract two concepts being confused, then compute & select generic
                # characterizations that best describe how the two are different
                ques_content = re.findall(r"How are (.*) and (.*) different\?$", utt)[0]
                conc1, conc2 = singularize(ques_content[0]), singularize(ques_content[1])
                conc_diffs = _compute_concept_differences(
                    self.domain_knowledge, conc1, conc2
                )
                response += _properties_to_nl(conc_diffs)

                self.taught_diffs.add(frozenset([conc1, conc2]))

            elif utt == "OK.":
                if len(self.current_queue) > 0:
                    # Remaining target concepts to test and teach
                    response += self.initiate_dialogue()
                else:
                    # No further interaction needed
                    pass

            else:
                raise NotImplementedError

        return response


def _compute_concept_differences(domain_knowledge, conc1, conc2):
    """
    Compute differences of concepts to teach based on the domain knowledge provided.
    Compare 'properties' of the concepts by the following order:

    0) Return empty dict if entirely identical (by the domain knowledge)
    1) If supertypes are different, teach the supertype difference
    2) Else-if belonging part sets are different, teach the part set difference
    3) Else-if attributes of any parts are different, teach the part attribute difference
    """
    assert conc1 in domain_knowledge and conc2 in domain_knowledge
    conc1_props = domain_knowledge[conc1]
    conc2_props = domain_knowledge[conc2]

    conc_diffs = {}

    # Compare supertype
    if conc1_props["supertype"] != conc2_props["supertype"]:
        conc_diffs["supertype"] = [
            (conc1, conc1_props["supertype"]),
            (conc2, conc1_props["supertype"]),
        ]

        raise NotImplementedError

    elif conc1_props["parts"] != conc2_props["parts"]:
        conc_diffs["parts"] = []

        # Symmetric set difference on part sets
        part_type_union = set(conc1_props["parts"]) | set(conc2_props["parts"])
        for part_type in part_type_union:
            if conc1_props["parts"].get(part_type) == conc2_props["parts"].get(part_type):
                # No difference on part type info
                continue

            if part_type in conc1_props["parts"]:
                conc_diffs["parts"].append((conc1, conc1_props["parts"][part_type]))
            if part_type in conc2_props["parts"]:
                conc_diffs["parts"].append((conc2, conc2_props["parts"][part_type]))

    elif conc1_props["part_attributes"] != conc2_props["part_attributes"]:
        conc_diffs["part_attributes"] = []

        raise NotImplementedError

        # Symmetric set difference on part attributes for each corresponding part
        for part in conc1_props["parts"]:
            conc1_attrs = conc1_props["parts"][part]
            conc2_attrs = conc2_props["parts"][part]

            if len(conc1_attrs-conc2_attrs) > 0:
                conc_diffs["part_attributes"].append(
                    (conc1, part, list(conc1_attrs-conc2_attrs))
                )
            if len(conc2_attrs-conc1_attrs) > 0:
                conc_diffs["part_attributes"].append(
                    (conc2, part, list(conc2_attrs-conc1_attrs))
                )

    return conc_diffs


def _parse_nl_reason(nl_string):
    """
    Recognize which claim was made by the agent by means of the NL string. Return
    a tuple ()
    """
    # Case 1: Agent made 'positive statement' that the mask refers to a part type
    re_test = re.findall(r"^this is a (.*)$", nl_string)
    if len(re_test) > 0:
        return (re_test[0], "existence")

    # Case 2: Agent is not sure if the provided mask refers to a part type
    re_test = re.findall(r"^this might not be a (.*)$", nl_string)
    if len(re_test) > 0:
        return (re_test[0], "uncertain")

    # Case 3: Agent made 'negative statement' that it failed to find any occurrence
    # of a part type
    re_test = re.findall(r"^this doesn't have a (.*)$", nl_string)
    if len(re_test) > 0:
        return (re_test[0], "absence")


def _properties_to_nl(conc_props):
    """
    Helper method factored out for realizing some collection of concept properties
    to natural language feedback to agent
    """
    # Prepare user feedback based on the selected concept difference
    feedback = []

    if "supertype" in conc_props:
        # "Xs are Ys"
        for conc, super_conc in conc_props["supertype"]:
            conc_str = pluralize(conc).capitalize()
            super_conc_str = pluralize(super_conc)
            feedback.append({
                "utterance": f"{conc_str} are {super_conc_str}.",
                "pointing": {}
            })
        raise NotImplementedError       # Remove after sanity check

    if "parts" in conc_props:
        # "Xs have Ys"
        for conc, part in conc_props["parts"]:
            conc_str = pluralize(conc).capitalize()
            part_str = pluralize(part)
            feedback.append({
                "utterance": f"{conc_str} have {part_str}.",
                "pointing": {}
            })

    if "part_attributes" in conc_props:
        # "Xs have Z1, Z2, ... and Zn Ys"
        for conc, part, attrs in conc_props["part_attributes"]:
            conc_str = pluralize(conc).capitalize()
            part_str = pluralize(part)
            attrs_str = " and ".join([", ".join(attrs[:-1])] + attrs[-1:]) \
                if len(attrs) > 1 else attrs[0]
            feedback.append({
                "utterance": f"{conc_str} have {attrs_str} {part_str}.",
                "pointing": {}
            })
        raise NotImplementedError       # Remove after sanity check

    return feedback


def _prepend_with_connective(response, connective):
    """
    Helper method factored out for prepending a connective to an agent utterance
    (if and only if connective is not None).
    """
    if connective is None: return response

    utterance = response["utterance"]
    response_new = {
        "utterance": f"{connective.capitalize()} {utterance[0].lower() + utterance[1:]}",
        "pointing": response["pointing"]
    }
    return response_new


def _add_after_redundancy_check(outgoing_buffer, feedback):
    """ Add to buffer of outgoing feedback messages only if not already included """
    if feedback in outgoing_buffer: return
    outgoing_buffer.append(feedback)


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

            if rule_type == "exists":
                base_prg_str += \
                    ", ".join([f"case_found({ci}) :- {full_scope_str}"]+piece_strs) + ".\n"
            else:
                assert rule_type == "forall"
                base_prg_str += \
                    ", ".join([f"attr_{ci}({attr_args_str}) :- {full_scope_str}"]+piece_strs) + ".\n"
                base_prg_str += \
                    f"exception_found({ci}) :- {full_scope_str}, not attr_{ci}({attr_args_str}).\n"

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
    ctl.add("base", [], base_prg_str)
    ctl.ground([("base", [])])
    ctl.configuration.solve.models = 0
    ctl.configuration.solver.seed = seed

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
