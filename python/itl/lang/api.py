"""
Language processing module API that exposes only the high-level functionalities
required by the ITL agent: situate the embodied agent in a physical environment,
understand & generate language input in the context of the dialogue
"""
from collections import defaultdict

from .semantics import SemanticParser
from .dialogue import DialogueManager


class LanguageModule:

    def __init__(self):
        """
        Args:
            opts: argparse.Namespace, from parse_argument()
        """
        self.semantic = SemanticParser()
        self.dialogue = DialogueManager()

        self.unresolved_neologisms = set()

    def situate(self, vis_scene):
        """
        Put entities in the physical environment into domain of discourse
        """
        # No-op if no new visual input
        if vis_scene is None:
            return

        # Incorporate parsed scene graph into dialogue context
        for oi, obj in vis_scene.items():
            mask = obj["pred_mask"]
            self.dialogue.referents["env"][oi] = {
                "mask": mask,
                "area": mask.sum().item()
            }
            self.dialogue.referent_names[oi] = oi
        
        # Register these indices as names, for starters
        self.dialogue.referent_names = { i: i for i in self.dialogue.referents["env"] }

    def understand(self, parses, pointing=None):
        """
        Update dialogue state by interpreting the parsed NL input in the context
        of the ongoing dialogue.

        `pointing` (optional) is a dict summarizing the 'gesture' made along with
        the utterance, indicating the reference (represented as mask) made by the
        n'th occurrence of linguistic token. Mostly for programmed experiments.
        """
        ti = len(self.dialogue.record)      # New dialogue turn index
        new_record = []                     # New dialogue record for the turn

        # For indexing clauses in dialogue turn
        se2ci = defaultdict(lambda: len(se2ci))

        for si, parse in enumerate(parses):
            clauses = parse["clauses"]
            referents = parse["referents"]
            source = parse["source"]

            ## Mapping referents to string identifiers within dialogue
            # For indexing individual clauses
            r2i = { ei: f"t{ti}c{se2ci[(si,ei)]}" for ei in clauses }
            # For indexing referents within individual clauses
            reindex_per_clause = { ei: 0 for ei in clauses }
            for rf, v in referents.items():
                if not rf.startswith("x"): continue

                ri_src_evt = r2i[v["source_evt"]]
                referent_id = reindex_per_clause[v["source_evt"]]
                reindex_per_clause[v["source_evt"]] += 1
                r2i[rf] = f"{ri_src_evt}x{referent_id}"

            # Add to the list of discourse referents
            for rf, rf_info in referents.items():
                if rf.startswith("x"):
                    # Annotate instance referents (const terms)
                    self.dialogue.referents["dis"][r2i[rf]] = {
                        k: r2i.get(v, v) for k, v in rf_info.items()
                    }
                elif rf.startswith("e"):
                    # Annotate eventuality referents (i.e. clauses)
                    self.dialogue.clause_info[r2i[rf]] = {
                        k: r2i.get(v, v) for k, v in rf_info.items()
                    }

            # Handle certain hard assignments
            for rf, rf_info in referents.items():
                if rf_info.get("dem_ref"):
                    # Demonstrately referenced entities
                    assert pointing is not None and len(pointing) > si

                    dem_mask = pointing[si][rf_info["dem_ref"]].astype(bool)
                    pointed = self.dialogue.dem_point(dem_mask)
                    self.dialogue.assignment_hard[r2i[rf]] = pointed
            for _, _, ante, cons in clauses.values():
                literals = ante + cons
                for lit in literals:
                    # Referenced by pronouns
                    if lit[:2] == ("sp", "pronoun1"):
                        self.dialogue.assignment_hard[r2i[lit[2][0]]] = "_user"
                    if lit[:2] == ("sp", "pronoun2"):
                        self.dialogue.assignment_hard[r2i[lit[2][0]]] = "_self"

            # ASP-compatible translation
            for ev_id, (gq, bvars, ante, cons) in clauses.items():
                if len(bvars) > 0:
                    # See if any presuppositions can be extracted out of the content;
                    # any literals that do not involve bound variables specified by
                    # `bvars`
                    presup = [
                        l for l in ante if len(bvars & set(l[2])) == 0
                    ] + [
                        l for l in cons if len(bvars & set(l[2])) == 0
                    ]
                    ante = [l for l in ante if len(bvars & set(l[2])) > 0]
                    cons = [l for l in cons if len(bvars & set(l[2])) > 0]
                else:
                    # No bound variables (not any type of quantification)
                    assert gq is None
                    presup = []

                formatted_data = _map_and_format((bvars, ante, cons), referents, r2i)

                mood = referents[ev_id]["mood"]
                match mood:
                    case "." | "~":
                        # Propositions, infinitives
                        pass        # Nothing special to do here

                    case "?":
                        # Questions

                        # Record turn-clause index of unanswered question
                        self.dialogue.unanswered_Qs.add((ti, se2ci[(si, ev_id)]))

                    case "!":
                        # Commands
                        
                        # Record turn-clause index of unexecuted command
                        self.dialogue.unexecuted_commands.add((ti, se2ci[(si, ev_id)]))
                
                    case _:
                        # ???
                        raise ValueError("Invalid sentence mood")

                new_record.append(((gq,)+formatted_data, source[ev_id], mood))

                if len(presup) > 0:
                    raise NotImplementedError

        self.dialogue.record.append(("U", new_record))    # Add new record

    def acknowledge(self):
        """ Push an acknowledging utterance to generation buffer """
        self.dialogue.to_generate.append((None, "OK.", {}))

    def generate(self):
        """ Flush the buffer of utterances prepared """
        if len(self.dialogue.to_generate) > 0:
            return_val = []

            ti_new = len(self.dialogue.record)

            new_record = []
            for ci_new, data in enumerate(self.dialogue.to_generate):
                ri_evt = f"t{ti_new}c{ci_new}"

                logical_forms, surface_form, mood, referents, dem_refs = data
                if logical_forms is None:
                    logical_forms = (None,) * 4

                new_record.append((logical_forms, surface_form, mood))

                # Dialogue state update
                self.dialogue.clause_info[ri_evt] = { "mood": mood }
                self.dialogue.referents["dis"].update(referents)

                # NL utterance to log/print (as long as not prefixed by a special
                # symbol '#')
                if not surface_form.startswith("#"):
                    return_val.append(("generate", (surface_form, dem_refs)))

            # Dialogue record update
            self.dialogue.record.append(("A", new_record))

            self.dialogue.to_generate = []

            return return_val
        else:
            return


def _map_and_format(data, referents, r2i):
    # Map MRS referents to ASP terms and format

    def fmt(rf):
        assert type(rf) == str
        if rf.startswith("x"):
            # Instance referents, may be variable or constant for now
            is_var = referents[rf].get('gen_quantified') or \
                referents[rf].get('wh_quantified')

            if is_var:
                return r2i[rf].capitalize()
            else:
                return r2i[rf]
        else:
            # Eventuality referents, always constant, referring to clauses that
            # already took place and are part of dialogue records
            assert rf.startswith("e")
            return r2i[rf]

    def process_conjuncts(conjuncts):
        return [
            (cnjt[0], cnjt[1], [fmt(arg) for arg in cnjt[2]]) for cnjt in conjuncts
        ]

    bvars, ante, cons = data

    return ({fmt(v) for v in bvars}, process_conjuncts(ante), process_conjuncts(cons))
