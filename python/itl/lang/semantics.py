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
        #   `source`: dict with event variables as keys and natural-language
        #       provenances as values.
        parses = []       # Return value
        for utt, dem_refs in zip(usr_in, pointing):
            # Possible patterns expected from the simulated user, in the scope of
            # this experiment:
            #   1) "Build a {truck_type}."
            #   2) "I will demonstrate how to build a {truck_type}."
            #   3) "I will demonstrate a valid join."
            #   4) "# Action/Effect: {action_type}({parameters})"
            #   5) "This is (not) a {concept_type}."
            #   6) "This is [red/green/blue/gold/white]."
            #   7) "Pick up this {part_type}." or "Pick up the subassembly/{subassembly_type}."
            #   8) "Join the {part_type_1} and the {part_type_2}."
            #   9) "{part_subtype} is a type of {part_supertype}."
            #   10) "# Observing"
            #   11) "What were you trying to do?"
            #   12) "Stop." & "Continue."
            #   13) "A {truck_subtype} is a truck with {part_subtypes}."
            #   14) "Use this {gt_descriptor} instead of this {dt_descriptor}."
            if re.match(r"Build a (.*)\.$", utt):
                # Imperative command to build an instance of the specified concept
                # from parts available in the scene
                comm_target = re.findall(r"Build a (.*)\.$", utt)[0]

                clauses = {
                    "e0": (
                        (), (), [],
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
                        (), (), [],
                        [
                            ("sp", "demonstrate", ["e0", "x0", "x1"]),
                            ("sp", "pronoun1", ["x0"]),
                            ("sp", "manner", ["x1", "e1"])
                        ]
                    ),
                    "e1": (
                        (), (), [],
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

            elif utt == "I will demonstrate a valid join.":
                # Signposting that user will be demonstrating a single join
                # that is possible from current progress
                clauses = {
                    "e0": (
                        (), (), [],
                        [
                            ("sp", "demonstrate", ["e0", "x0", "e1"]),
                            ("sp", "pronoun1", ["x0"])
                        ]
                    ),
                    "e1": (
                        (), (), [],
                        [("va", "join", ["e1", "x1", "x2"])]
                    )
                }
                referents = {
                    "e0": { "mood": ".", "tense": "future" },
                    "x0": { "source_evt": "e0" },
                    "e1": { "mood": "~" },        # 'Infinitive' mood, let's say
                    "x1": { "source_evt": "e1" },
                    "x2": { "source_evt": "e1" }
                }
                source = { "e0": utt, "e1": "# event noun" }

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
                        (), (), [],
                        [
                            ("va", act_type, ["e0", "x0"] + [
                                f"x{i+1}" for i in range(len(act_params))
                            ])
                        ]
                    )
                }
                referents = {
                    "e0": { "mood": "." },
                    "x0": { "source_evt": "e0" }
                }
                source = { "e0": utt }

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

            elif re.match(r"This is (not )?a (.*)\.$", utt):
                # Providing a concept labeling of the demonstratively referenced
                # object instance; if positive, may signal end of demonstration if
                # the 'target concept' (as previously signaled by 2) above) instance
                # is provided
                pol, labeled_target = re.findall(r"This is (not )?a (.*)\.$", utt)[0]

                pol = "~" if pol == "not " else ""
                cons = [("n", f"{pol}{labeled_target}", ["x0"])]
                clauses = { "e0": ((), (), [], cons) }
                referents = {
                    "e0": { "mood": "." },
                    "x0": { "source_evt": "e0", "dem_ref": (0, 4) }
                }
                source = { "e0": utt }

            elif re.match(r"This is (red|green|blue|gold|white).", utt):
                # Providing color concept labeling of the demonstratively referenced
                # object instance, during color concept injection pre-task. Label
                # always positive; colors assumed to be mutually exclusive, so other
                # colors are inferred to be negative.
                color_label = re.findall(r"This is (red|green|blue|gold|white).", utt)[0]
                color_negatives = {"red", "green", "blue", "gold", "white"} - {color_label}

                clauses = {
                    "e0": ((), (), [], [("a", color_label, ["x0"])])
                } | {
                    f"e{i+1}": ((), (), [("a", col, [f"x{i+1}"])], [])
                    for i, col in enumerate(color_negatives)
                }       # Single positive label & inferred negative labels
                referents = {}
                for i in range(len(color_negatives)+1):
                    referents |= {
                        f"e{i}": { "mood": "." },
                        f"x{i}": { "source_evt": f"e{i}", "dem_ref": (0, 4) }
                    }
                source = { "e0": utt } | {
                    f"e{i+1}": "(Inference by domain knowledge)"
                    for i in range(len(color_negatives))
                }

            elif re.match(r"Pick up (.*)\.$", utt):
                # Utterance has appearance of a command, but more like a NL
                # description (with haptic-ostensive reference) of a pick-up action
                # being demonstrated by the user

                # Only consider "Pick up this {part_type}" descriptions; "Pick up
                # the subassembly/{subassembly_type}" utterances don't provide any
                # additional learning signals in our scope
                target = re.findall(r"Pick up this (.*)\.$", utt)
                if len(target) > 0:
                    # "~ this {part_type}" case, extract the NL label
                    target = target[0]

                    clauses = {
                        "e0": (
                            (), (), [],
                            [
                                ("va", "pick_up", ["e0", "x0", "x1"]),
                                ("sp", "pronoun2", ["x0"]),
                                ("n", target, ["x1"])
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

            elif re.match(r"Join (.*) and (.*)\.$", utt):
                # Similar to the case above; utterance has appearance of a command,
                # but more like a NL description of a join action being demonstrated
                # by the user

                # Note that a target quantified with indefinite article (e.g., a
                # wheel) signals introduction of an instance of the denoted part
                # (super)type, which specifies type requirements for the part
                # 'slot' in the target structure for the first time
                _, target_l, _, target_r = \
                    re.findall(r"Join (a|the) (.*) and (a|the) (.*)\.$", utt)[0]

                clauses = clauses = {
                    "e0": (
                        (), (), [],
                        [
                            ("va", "join", ["e0", "x0", "x1", "x2"]),
                            ("sp", "pronoun2", ["x0"]),
                            ("n", target_l, ["x1"]),
                            ("n", target_r, ["x2"])
                        ]
                    )
                }
                referents = {
                    "e0": { "mood": "." },
                    "x0": { "source_evt": "e0" },
                    "x1": { "source_evt": "e0" },
                    "x2": { "source_evt": "e0" }
                }
                source = { "e0": utt }

            elif re.match(r"(.*) is a type of (.*)\.$", utt):
                # Providing a taxonomy (hypernymy/hyponymy) relation between
                # a part subtype and a part supertype. Should be identified
                # as a lifted rule and stored in KB later.
                part_subtype, part_supertype = \
                    re.findall(r"(.*) is a type of (.*)\.$", utt)[0]

                clauses = {
                    "e0": (
                        (), (), [],
                        [
                            ("sp", "subtype", ["e0", "x0", "x1"]),
                            ("n", part_subtype, ["x0"]),
                            ("n", part_supertype, ["x1"])
                        ]
                    )
                }
                referents = {
                    "e0": { "mood": "." },
                    "x0": { "source_evt": "e0", "is_pred": True },
                    "x1": { "source_evt": "e0", "is_pred": True }
                }
                source = { "e0": utt }

            elif utt == "What were you trying to join?":
                # Asking the agent's intention (which led to a questionable
                # action from the user's viewpoint)
                clauses = {
                    "e0": (
                        (), (), [],
                        [
                            ("sp", "intend", ["e0", "x0", "e1"]),
                            ("sp", "pronoun2", ["x0"])
                        ]
                    ),
                    "e1": (
                        (), (), [], [
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

            elif re.match(r"A (.*) is a truck with (.*)\.$", utt):
                # Providing verbal definition of a truck subtype by
                # necessary parts
                truck_subtype, part_subtypes = \
                    re.findall(r"A (.*) is a truck with (.*)\.$", utt)[0]
                part_subtypes = [
                    subtype.strip("a ") for subtype in part_subtypes.split(" and ")
                ]

                # Split the definition into two rules: 1) taxonomy rule s.t.
                # subtype(O) |- supertype(O), and 2) constraint that the
                # provided property must be observed when building an instance
                # of the subtype (e.g., a firetruck must include a ladder)
                clauses = {
                    "e0": (
                        (), (), [],
                        [
                            ("sp", "subtype", ["e0", "x0", "x1"]),
                            ("n", truck_subtype, ["x0"]),
                            ("n", "truck", ["x1"])
                        ]
                    )
                } | {
                    f"e{i+1}": (
                        ("forall", "exists",),
                        (f"x{2+2*i}", f"x{2+2*i+1}"),
                        [("n", truck_subtype, [f"x{2+2*i}"])],
                        [
                            ("vs", "have", [f"x{2+2*i}", f"x{2+2*i+1}"]),
                            ("n", subtype, [f"x{2+2*i+1}"])
                        ]
                    )
                    for i, subtype in enumerate(part_subtypes)
                }
                referents = {
                    "e0": { "mood": "." },
                    "x0": { "source_evt": "e0", "is_pred": True },
                    "x1": { "source_evt": "e0", "is_pred": True }
                }
                for i in range(len(part_subtypes)):
                    referents |= {
                        f"e{i+1}": { "mood": "." },
                        f"x{2+2*i}": { "source_evt": f"e{i+1}" },
                        f"x{2+2*i+1}": { "source_evt": f"e{i+1}" },
                    }
                source = { f"e{i}": utt for i in range(len(part_subtypes)+1) }

            elif re.match(r"Use this (.*) instead of this (.*)\.$", utt):
                # Informing agent to use some ground-truth part instance
                # of a same category instead of the distractor part instance
                # previously used
                gt_descriptor, dt_descriptor = \
                    re.findall(r"Use this (.*) instead of this (.*)\.$", utt)[0]
                if gt_descriptor == "one":
                    gt_descriptor = []
                else:
                    gt_descriptor = gt_descriptor.split(" ")
                    gt_descriptor = [("n", gt_descriptor[-1], ["x1"])] + \
                        ([("a", gt_descriptor[0], ["x1"])] if len(gt_descriptor) > 1 else [])
                if dt_descriptor == "one":
                    dt_descriptor = []
                else:
                    dt_descriptor = dt_descriptor.split(" ")
                    dt_descriptor = [("n", dt_descriptor[-1], ["x2"])] + \
                        ([("a", dt_descriptor[0], ["x2"])] if len(dt_descriptor) > 1 else [])

                clauses = {
                    "e0": (
                        (), (), [],
                        [
                            ("sp", "pronoun2", ["x0"]),
                            ("va", "pick_up", ["e0", "x0", "x1"]),
                            ("va", "~pick_up", ["e0", "x0", "x2"]),
                        ] + gt_descriptor + dt_descriptor
                    )
                }
                referents = {
                    "e0": { "mood": "." },
                    "x0": { "source_evt": "e0" },
                    "x1": { "source_evt": "e0", "dem_ref": sorted(dem_refs)[0] },
                    "x2": { "source_evt": "e0", "dem_ref": sorted(dem_refs)[1] }
                }
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
