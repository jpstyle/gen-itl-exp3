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
            if "pred_mask" not in obj: continue

            mask = obj["pred_mask"]
            self.dialogue.referents["env"][-1][oi] = {
                "mask": mask,
                "area": mask.sum().item()
            }
            self.dialogue.referent_names[oi] = oi
        
        # Register these indices as names, for starters
        self.dialogue.referent_names.update({ i: i for i in self.dialogue.referents["env"][-1] })

    def understand(self, l_input):
        """
        Update dialogue state by interpreting the parsed NL input in the context
        of the ongoing dialogue.

        `pointing` (optional) is a dict summarizing the 'gesture' made along with
        the utterance, indicating the reference (represented as mask) made by the
        n'th occurrence of linguistic token. Mostly for programmed experiments.
        """
        ti = len(self.dialogue.record)      # Starting dialogue turn index

        # First segment language inputs into consecutive sequences of utterances
        # given by same speakers (Doesn't do much in dialogues with only two
        # participants, but for completeness' sake)
        segments = []; consecutive_parses = []; last_speaker = None
        for parse, dem_refs, spk in zip(*l_input):
            if spk == last_speaker:
                consecutive_parses.append((parse, dem_refs))
            else:
                # Speaker different from the last one, new speaker taking turn
                # (except None speaker, which doesn't stand for any real one)
                if last_speaker is not None:
                    segments.append((last_speaker, consecutive_parses))
                consecutive_parses = [(parse, dem_refs)]
                last_speaker = spk
        # Don't forget last consecutive segment
        segments.append((last_speaker, consecutive_parses))

        for spk, parse_seq in segments:
            # Record to be added for the turn
            new_record = []

            # For indexing clauses in dialogue turn
            se2ci = defaultdict(lambda: len(se2ci))

            for si, (parse, dem_refs) in enumerate(parse_seq):
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
                        dem_ref = dem_refs[rf_info["dem_ref"]]
                        if isinstance(dem_ref, str):
                            pointed = dem_ref
                        else:
                            dem_mask = dem_refs[rf_info["dem_ref"]].astype(bool)
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

                    clause_info = referents[ev_id]
                    match clause_info["mood"]:
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

                    new_record.append(((gq,)+formatted_data, source[ev_id]))
                    self.dialogue.clause_info[r2i[ev_id]] = clause_info

                    if len(presup) > 0:
                        raise NotImplementedError

            self.dialogue.record.append((spk, new_record))    # Add new record
            ti += 1         # New turn commences (if any further segments exist)

    def utter_simple(self, utterance):
        """
        Push a simple utterance (that doesn't need a specific logical form
        associated) to generation buffer
        """
        surface_form, clause_info = utterance
        self.dialogue.to_generate.append(
            (None, surface_form, { "e": clause_info }, {}, {})
        )

    def generate(self):
        """ Flush the buffer of utterances prepared """
        if len(self.dialogue.to_generate) == 0: return

        return_val = []

        ti_new = len(self.dialogue.record)

        new_record = []
        for ci_new, data in enumerate(self.dialogue.to_generate):
            ri_evt = f"t{ti_new}c{ci_new}"

            logical_forms, surface_form = data[:2]
            referents, predicates, dem_refs = data[2:]

            # Dialogue record update
            if logical_forms is not None:
                gq, bvars, ante, cons = logical_forms
                def rename_referents(refs):
                    """ Rename referents as appropriate """
                    renamed = []
                    for rf in refs:
                        if rf in referents: renamed.append(f"{ri_evt}{rf}")
                        else:
                            if isinstance(rf, tuple) and rf[0] == "e":
                                ci_offset = rf[1]
                                renamed.append(f"t{ti_new}c{ci_new+ci_offset}")
                            else: renamed.append(rf)
                    return renamed
                bvars = set(rename_referents(bvars))
                ante = [
                    (pos, name, rename_referents(args))
                    for pos, name, args in ante
                ]
                cons = [
                    (pos, name, rename_referents(args))
                    for pos, name, args in cons
                ]
                logical_forms = (gq, bvars, ante, cons)
            else:
                logical_forms = (None,) * 4
            new_record.append((logical_forms, surface_form))

            # Dialogue state update; attach appropriate index prefix to
            # referent & predicate identifiers
            for rf, data in referents.items():
                if rf == "e":
                    # Event referent
                    self.dialogue.clause_info[ri_evt] = data
                else:
                    # Instance referent
                    rf = f"{ri_evt}{rf}"
                    if data["entity"] is not None:
                        self.dialogue.value_assignment[rf] = data["entity"]
                    self.dialogue.referents["dis"][rf] = {
                        "source_evt": ri_evt, **data["rf_info"]
                    }
            for tok_ind, sense in predicates.items():
                tok_ind = (f"t{ti_new}", f"c{ci_new}", tok_ind)
                self.dialogue.word_senses[tok_ind] = sense

            # NL utterance to log/print (as long as not prefixed by a special
            # symbol '#')
            if not surface_form.startswith("#"):
                return_val.append(("generate", (surface_form, dem_refs)))

        # Dialogue record update
        self.dialogue.record.append(("Student", new_record))

        self.dialogue.to_generate = []

        return return_val


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
