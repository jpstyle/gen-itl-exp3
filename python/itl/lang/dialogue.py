import copy

import numpy as np

from ..vision.utils import mask_iou
from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import wrap_args


IOU_THRES = 0.8

class DialogueManager:
    """ Maintain dialogue state and handle NLU, NLG in context """

    def __init__(self):

        self.referents = {
            "env": [],          # Environment entities
            "dis": {}           # Discourse referents
        }

        # Store fixed assignment by demonstrative+pointing, names, etc.
        self.assignment_hard = {}
        # Store mapping from symbolic name to entity
        self.referent_names = {}
        # Store any important information re. individual clauses
        self.clause_info = {}

        # Store current best assignments from discourse referents to entities
        # in universe
        self.value_assignment = {}
        # Store current best assignments from predicate symbols used in
        # discourse to denoted concepts
        self.word_senses = {}

        # Dialogue record, each entry consisting of the speaker of the turn
        # and clauses in the turn
        self.record = []

        self.unanswered_Qs = set()
        self.unexecuted_commands = set()

        # Buffer of utterances to generate
        self.to_generate = []

    def refresh(self):
        """ Clear the current dialogue state to start fresh in a new situation """
        self.__init__()

    def dem_point(self, dem_mask):
        """
        Provided a segmentation mask specification, return reference (by object id)
        to an appropriate environment entity recognized, potentially creating one
        if not already explicitly aware of it as object.
        """
        env_entities = {
            k: v for k, v in self.referents["env"][-1].items()
            if not (k == "_self" or k == "_user")
        }

        if len(env_entities) > 0:
            # First check if there's any existing high-IoU segmentation mask; by
            # 'high' we refer to the threshold we set as global constant above
            env_ref_masks = np.stack([e["mask"] for e in env_entities.values()])
            ious = mask_iou(dem_mask[None], env_ref_masks)[0]
            best_match = ious.argmax()

            if ious[best_match] > IOU_THRES:
                # Presume the 'pointed' entity is actually this one
                pointed = list(env_entities.keys())[best_match]
            else:
                # Register the entity as a novel environment referent
                pointed = f"o{len(env_ref_masks)}"

                self.referents["env"][-1][pointed] = {
                    "mask": dem_mask,
                    "area": dem_mask.sum().item()
                }
                self.referent_names[pointed] = pointed
        else:
            # Register the entity as a novel environment referent
            pointed = "o0"

            self.referents["env"][-1][pointed] = {
                "mask": dem_mask,
                "area": dem_mask.sum().item()
            }
            self.referent_names[pointed] = pointed

        return pointed

    def resolve_symbol_semantics(self, lexicon):
        """
        Find a fully specified mapping from symbols in discourse record to corresponding
        entities and concepts; in other words, perform reference resolution and word
        sense disambiguation.

        Args:
            lexicon: Agent's lexicon, required for matching between environment entities
                vs. discourse referents for variable assignment
        """
        # Find the best estimate of referent value assignment
        sm_prog = Program()

        # Environmental referents
        occurring_atoms = set()
        for ent in self.referents["env"][-1]:
            if ent not in occurring_atoms:
                sm_prog.add_absolute_rule(Rule(head=Literal("env", wrap_args(ent))))
                occurring_atoms.add(ent)

        # Discourse referents
        for rf, v in self.referents["dis"].items():
            # Only process unresolved value assignments
            if rf in self.value_assignment: continue
            # No need to assign if not an entity referent
            if "x" not in rf: continue
            # No need to assign if universally quantified or wh-quantified
            if v.get("univ_quantified") or v.get("wh_quantified"): continue

            sm_prog.add_absolute_rule(Rule(head=Literal("dis", wrap_args(rf))))
            if v.get("referential"):
                sm_prog.add_absolute_rule(Rule(head=Literal("referential", wrap_args(rf))))

        # Hard assignments by pointing, etc.
        for rf, env in self.assignment_hard.items():
            # Only process unresolved value assignments
            if rf in self.value_assignment: continue

            sm_prog.add_absolute_rule(
                Rule(body=[Literal("assign", [(rf, False), (env, False)], naf=True)])
            )

        # Understood dialogue record contents
        occurring_preds = set()
        for ti, (speaker, turn_clauses) in enumerate(self.record):
            for ci, ((_, _, ante, cons), raw) in enumerate(turn_clauses):
                if speaker == "Student" and not raw.startswith("# Effect:"):
                    # Nothing particular to do with agent's own utterances generated
                    # by comp_actions; action effects (starting with "# Effect:")
                    # need processing though
                    continue

                ante_preds = [lit[:2] for lit in ante]
                cons_preds = [lit[:2] for lit in cons]

                # Symbol token occurrence locations
                for c, preds in [("c", cons_preds), ("a", ante_preds)]:
                    for pi, p in enumerate(preds):
                        # Skip reserved predicates of 'special' (denoted 'sp') type
                        if p[0] == "sp": continue

                        occurring_preds.add(p)

                        sym = f"{p[0]}_{p[1]}"
                        tok_loc = f"t{ti}_c{ci}_p{c}{pi}"
                        sm_prog.add_absolute_rule(
                            Rule(head=Literal("pred_token", wrap_args(tok_loc, sym)))
                        )

                ante_args = [lit[2] for lit in ante]
                cons_args = [lit[2] for lit in cons]
                occurring_ent_refs = sum(ante_args+cons_args, [])
                # Only include entity referents (including 'x'), not eventuality referents
                occurring_ent_refs = {ref for ref in occurring_ent_refs if "x" in ref}

                if all(arg in self.assignment_hard for arg in occurring_ent_refs):
                    # Below not required if all occurring args are hard-assigned to some entity
                    continue

        # Predicate info needed for word sense selection
        for p in occurring_preds:
            sym = f"{p[0]}_{p[1]}"

            # Consult lexicon to list denotation candidates
            if p in lexicon.s2d:
                for den in lexicon.s2d[p]:
                    pos_match = (p[0], den[0]) == ("n", "pcls") \
                        or (p[0], den[0]) == ("a", "pcls") \
                        or (p[0], den[0]) == ("p", "prel") \
                        or (p[0], den[0]) == ("vs", "prel") \
                        or (p[0], den[0]) == ("va", "arel")
                    if not pos_match: continue

                    sm_prog.add_absolute_rule(
                        Rule(head=Literal("may_denote", wrap_args(sym, f"{den[0]}_{den[1]}")))
                    )
            else:
                # Predicate symbol not found in lexicon: unresolved neologism
                sm_prog.add_absolute_rule(
                    Rule(head=Literal("may_denote", wrap_args(sym, "_neo")))
                )

        ## Assignment program rules

        # 1 { assign(X,E) : env(E) } 1 :- dis(X), referential(X).
        sm_prog.add_rule(Rule(
            head=Literal(
                "assign", wrap_args("X", "E"),
                conds=[Literal("env", wrap_args("E"))]
            ),
            body=[
                Literal("dis", wrap_args("X")),
                Literal("referential", wrap_args("X"))
            ],
            lb=1, ub=1
        ))

        # { assign(X,E) : env(E) } 1 :- dis(X), not referential(X).
        sm_prog.add_rule(Rule(
            head=Literal(
                "assign", wrap_args("X", "E"),
                conds=[Literal("env", wrap_args("E"))]
            ),
            body=[
                Literal("dis", wrap_args("X")),
                Literal("referential", wrap_args("X"), naf=True)
            ],
            ub=1
        ))

        # 1 { denote(T,D) : may_denote(S,D) } 1 :- pred_token(T,S).
        sm_prog.add_rule(Rule(
            head=Literal(
                "denote", wrap_args("T", "D"),
                conds=[Literal("may_denote", wrap_args("S", "D"))]
            ),
            body=[
                Literal("pred_token", wrap_args("T", "S"))
            ],
            lb=1, ub=1
        ))

        # By querying for the optimal assignment, essentially we are giving the user a 'benefit
        # of doubt', such that any statements made by the user are considered as True, and the
        # agent will try to find the 'best' assignment to make it so.
        # (Note: this is not a probabilistic inference, and the confidence scores provided as
        # arguments are better understood as properties of the env. entities & disc. referents.)
        opt_models = sm_prog.optimize([
            # Minimize arbitrary assignments
            ("minimize", [
                ([Literal("assign", wrap_args("X", "E"))], 1, ["X"])
            ])
        ])

        best_assignment = [atom.args for atom in opt_models[0] if atom.name == "assign"]
        best_assignment = { args[0][0]: args[1][0] for args in best_assignment }
        best_assignment.update({
            rf: None for rf in self.referents["dis"]
            if rf not in best_assignment and rf not in self.value_assignment
                # List all unlisted discourse referents so that they are not included in
                # the symbol grounding program again
        })

        tok2sym_map = [atom.args[:2] for atom in opt_models[0] if atom.name == "pred_token"]
        tok2sym_map = {
            tuple(token[0].split("_")): ((spl := symbol[0].split("_"))[0], "_".join(spl[1:]))
            for token, symbol in tok2sym_map
        }

        word_senses = [atom.args[:2] for atom in opt_models[0] if atom.name == "denote"]
        word_senses = {
            tuple(token[0].split("_")): denotation[0]
            for token, denotation in word_senses
        }
        word_senses = {
            token: (tok2sym_map[token], denotation)
                if denotation != "_neo" else (tok2sym_map[token], None)
            for token, denotation in word_senses.items()
        }

        self.value_assignment.update(best_assignment)
        self.word_senses.update(word_senses)

    def export_resolved_record(self):
        """
        Translate and export logical content of dialogue record based on current
        estimate of value assignment and word sense selection
        """
        a_map = lambda args: [
            self.value_assignment.get(arg, arg) or arg
            for arg in args
        ]

        # Recursive helper methods for encoding pre-translation tuples representing
        # literals into actual Literal objects
        encode_lits = lambda cnjt, ti, ci, c_or_a, li: Literal(
            self.word_senses.get(
                (f"t{ti}",f"c{ci}",f"p{c_or_a}{li}"),
                # If not found (likely reserved predicate), fall back to cnjt's pred
                (None, "_".join(cnjt[:2]))
            )[1] or "sp_neologism",
            args=wrap_args(*a_map(cnjt[2]))
        )

        record_translated = []
        for ti, (speaker, turn_clauses) in enumerate(self.record):
            turn_translated = []
            for ci, ((gq, bvars, ante, cons), raw) in enumerate(turn_clauses):
                # Translate consequent and antecedent by converting NL predicates to
                # denoted concepts
                if cons is not None and len(cons) > 0:
                    tr_cons = tuple(
                        encode_lits(lit, ti, ci, "c", li)
                        for li, lit in enumerate(cons)
                    )
                else:
                    tr_cons = None

                if ante is not None and len(ante) > 0:
                    tr_ante = tuple(
                        encode_lits(lit, ti, ci, "a", li)
                        for li, lit in enumerate(ante)
                    )
                else:
                    tr_ante = None

                clause_info = self.clause_info[f"t{ti}c{ci}"]
                turn_translated.append(((gq, bvars, tr_ante, tr_cons), raw, clause_info))

            record_translated.append((speaker, turn_translated))

        return record_translated
