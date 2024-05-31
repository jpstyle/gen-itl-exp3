import re

import inflect


singularize = inflect.engine().singular_noun
pluralize = inflect.engine().plural

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

        parses = []       # Return value
        for utt, dem_refs in zip(usr_in, pointing):
            # Possible patterns expected from the simulated user, in the scope of
            # this experiment:
            #   1) "Build a {truck_type}."
            #   2) "I will demonstrate how to build a {truck_type}."
            #   3) "# Action: {action_type}({action_args})"
            #   4) "This is a {concept_type}"
            #   5) ...
            if re.match(r"Build a (.*)\.$", utt):
                # 1) Imperative command to build an instance of the specified concept
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
                # 2) Signposting that user will be demonstrating how to build
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

            elif re.match(r"# Action: (.*)\((.*)\)$", utt):
                # 3) User provided annotation of an action in the ongoing demonstration;
                # while not free-form NL input, abuse & exploit the formalism
                act_anno = re.findall(r"# Action: (.*)\((.*)\)$", utt)[0]
                act_type, act_params = act_anno

                # Unpack PascalCased action type and snake_case it
                splits = re.findall(
                    r"(?:[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", act_type
                )
                splits = [w[0].lower()+w[1:] for w in splits]
                act_type = "_".join(splits)
                # Unpack action parameters
                act_params = act_params.split(",")

                clauses = {
                    "e0": (
                        None, set(), [],
                        [
                            ("va", act_type, ["e0", "x0"] + [f"x{i+1}" for i in range(len(act_params))]),
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

            elif re.match(r"This is a (.*)\.$", utt):
                # 4) Providing a concept label of the demonstratively referenced
                # object instance; may signal end of demonstration if the 'target
                # concept' (as previously signaled by 2) above) instance is provided
                labeled_target = re.findall(r"This is a (.*)\.$", utt)[0]

                clauses = {
                    "e0": (
                        None, set(), [],
                        [
                            ("n", labeled_target, ["x0"])
                        ]
                    )
                }
                referents = {
                    "e0": { "mood": "." },
                    "x0": { "source_evt": "e0", "dem_ref": (0, 4) }
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
