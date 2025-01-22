import re

class SemanticParser:
    """
    Semantic parser that first processes free-form language inputs into MRS, and then
    translates them into ASP-friendly custom formats
    """
    def __init__(self):
        # In case the project gets to use any advanced semantic parser...
        pass

    def nl_parse(self, usr_in, pointing):
        # As opposed to predecessors, resort to simpler template matching; semantic
        # parsing is not our core focus in this part of the project
        assert isinstance(usr_in, list)

        # Parsing outputs should be of the following format:
        #   `clauses`: dict with event variables as keys and logical forms of
        #       clause contents as values. Each value is a 4-tuple of (gq, bvars,
        #       ante, cons), respectively representing appropriate general quantifier
        #       (if any), set of bounding variables, antecedent literals, consequent
        #       literals.
        #    `referents`: dict with discourse referent and event variables as keys
        #       and their associated information as values.
        #   `events`: dict with event variables as keys and natural-language
        #       provenances as values.
        parses = []       # Return value
        for utt, dem_refs in zip(usr_in, pointing):
            # Possible patterns expected from the simulated user, in the scope of
            # this experiment:
            #   1) "Build a {truck_type}."
            #   2) "I will demonstrate how to build a {truck_type}."
            #   3) "# Action/Effect: {action_type}({parameters})"
            #   4) "This is (not) a {concept_type}."
            #   5) "Pick up a {part_type}." or "Pick up the {subassembly_type}."
            #   6) "Join the {part_type_1} and the {part_type_2}."
            #   7) "# Observing"
            #   8) "What were you trying to do?"
            #   9) "Stop." & "Continue."
            if re.match(r"Build a (.*)\.$", utt):
                # Imperative command to build an instance of the specified concept
                # from parts available in the scene
                comm_target = re.findall(r"Build a (.*)\.$", utt)[0]

                clauses = {
                    "e0": (
                        None, set(), [],
                        [
                            ("va", "build", ["e0", "x0", "x1"]),
                            ("sp", "pronoun2", ["x0"]),
                            ("n", comm_target, ["x1"])
                        ]
                    )
                }
                referents = {
                    "e0": { "mood": "!" },
                    "x0": { "source_evt": "e0" },
                    "x1": { "source_evt": "e0", "is_pred": True }
                }

                source = { "e0": utt }

            elif re.match(r"I will demonstrate how to build a (.*)\.$", utt):
                # Signposting that user will be demonstrating how to build
                # an instance of the target concept
                demo_target = re.findall(r"I will demonstrate how to build a (.*)\.$", utt)[0]

                clauses = {
                    "e0": (
                        None, set(), [],
                        [
                            ("sp", "demonstrate", ["e0", "x0", "x1"]),
                            ("sp", "pronoun1", ["x0"]),
                            ("sp", "manner", ["x1", "e1"])
                        ]
                    ),
                    "e1": (
                        None, set(), [],
                        [
                            ("va", "build", ["e1", "x2", "x3"]),
                            ("n", demo_target, ["x3"])
                        ]
                    )
                }
                referents = {
                    "e0": { "mood": ".", "tense": "future" },
                    "x0": { "source_evt": "e0" },
                    "x1": { "source_evt": "e0" },
                    "e1": { "mood": "~" },        # Infinitive mood
                    "x2": { "source_evt": "e1" },
                    "x3": { "source_evt": "e1", "is_pred": True }
                }

                source = { "e0": utt, "e1": "# to-infinitive phrase" }

            elif re.match(r"# Action: (.*)\((.*)\)$", utt) or \
                re.match(r"# Effect: (.*)\((.*)\)$", utt):
                # Annotation of intent or effect of an action in the ongoing demo;
                # while not free-form NL input, abuse & exploit the formalism
                if utt.startswith("# Action:"):
                    act_anno = re.findall(r"# Action: (.*)\((.*)\)$", utt)[0]
                else:
                    act_anno = re.findall(r"# Effect: (.*)\((.*)\)$", utt)[0]
                act_type, act_params = act_anno

                # Unpack action parameters
                act_params = act_params.split(",")

                clauses = {
                    "e0": (
                        None, set(), [],
                        [
                            ("va", act_type, ["e0", "x0"] + [
                                f"x{i+1}" for i in range(len(act_params))
                            ]),
                            ("sp", "pronoun1", ["x0"]),
                        ]
                    )
                }
                referents = {
                    "e0": { "mood": "." },
                    "x0": { "source_evt": "e0" }
                }

                # Store original parameter string values and attach demonstrative
                # references where applicable
                for pi, prm in enumerate(act_params):
                    ri = pi + 1
                    referents[f"x{ri}"] = {
                        "source_evt": "e0",
                        "name": prm
                    }
                for crange in dem_refs:
                    pi = act_params.index(utt[crange[0]:crange[1]])
                    ri = pi + 1
                    referents[f"x{ri}"]["dem_ref"] = crange

                source = { "e0": utt }

            elif re.match(r"This is (not )?a (.*)\.$", utt):
                # Providing a concept labeling of the demonstratively referenced
                # object instance; if positive, may signal end of demonstration if
                # the 'target concept' (as previously signaled by 2) above) instance
                # is provided
                pol, labeled_target = re.findall(r"This is (not )?a (.*)\.$", utt)[0]

                if pol != "not ":
                    # Positive polarity; factual statement represented as a simple
                    # antecedent-less clause
                    ante = []
                    cons = [("n", labeled_target, ["x0"])]
                else:
                    # Negative polarity; factual statement represented as a simple
                    # consequent-less clause (similar to headless integrity constraint
                    # in logic programming)
                    ante = [("n", labeled_target, ["x0"])]
                    cons = []
                clauses = { "e0": (None, set(), ante, cons) }
                referents = {
                    "e0": { "mood": "." },
                    "x0": { "source_evt": "e0", "dem_ref": (0, 4) }
                }

                source = { "e0": utt }

            elif re.match(r"Pick up (.*)\.$", utt):
                # Utterance has appearance of a command, but more like a NL
                # description (with haptic-ostensive reference) of a pick-up action
                # being demonstrated by the user

                # Only consider "Pick up a {part_type}" descriptions; "Pick up
                # the {subassembly_type}" utterances don't provide any additional
                # learning signals in our scope
                pick_up_target = re.findall(r"Pick up a (.*)\.$", utt)
                if len(pick_up_target) > 0:
                    # "~ a {part_type}" case, extract the NL label
                    pick_up_target = pick_up_target[0]

                    clauses = {
                        "e0": (
                            None, set(), [],
                            [
                                ("va", "pick_up", ["e0", "x0", "x1"]),
                                ("sp", "pronoun2", ["x0"]),
                                ("n", pick_up_target, ["x1"])
                            ]
                        )
                    }
                    referents = {
                        "e0": { "mood": "." },
                        "x0": { "source_evt": "e0" },
                        "x1": { "source_evt": "e0" }
                    }

                    source = { "e0": utt }
                else:
                    # "~ the {subassembly_type}" case, null logical form
                    clauses = {}
                    referents = {}
                    source = { "e0": utt }

            elif re.match(r"Join the (.*) and the (.*)\.$", utt):
                # Similar to the case above; utterance has appearance of a command,
                # but more like a NL description (with haptic-ostensive reference)
                # of a join action being demonstrated by the user

                # Note that in our scope, these utterances won't provide any further
                # learning signals in the learner's demonstration analysis procedure
                # in addition to the '# Action/Effect: ~' annotations. Therefore,
                # we'll just return null logical form here.
                clauses = {}
                referents = {}
                source = { "e0": utt }

            elif utt == "What were you trying to join?":
                # Asking the agent's intention (which led to a questionable
                # action from the user's viewpoint)
                clauses = {
                    "e0": (
                        None, set(), [],
                        [
                            ("sp", "intend", ["e0", "x0", "e1"]),
                            ("sp", "pronoun2", ["x0"])
                        ]
                    ),
                    "e1": (
                        None, set(), [], [
                            ("va", "join", ["e1", "x1", "x2"])
                        ]
                    )
                }
                referents = {
                    "e0": { "mood": "?", "tense": "past" },
                    "x0": { "source_evt": "e0" },
                    "e1": { "mood": "~" },        # Infinitive mood
                    "x1": { "source_evt": "e1" },
                    "x2": { "source_evt": "e1" }
                }

                source = { "e0": utt, "e1": "# to-infinitive phrase" }

            elif utt == "Stop." or utt == "Continue.":
                # Raw form has all necessary info, null logical form
                clauses = {}
                referents = {}
                source = { "e0": utt }

            else:
                # Don't know how to process other patterns
                raise ValueError(f"Cannot parse input: '{utt}'")

            parses.append({
                "clauses": clauses,
                "referents": referents,
                "source": source
            })

        return parses
