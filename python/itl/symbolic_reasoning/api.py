"""
symbolic reasoning module API that exposes only the high-level functionalities
required by the ITL agent: make sense out of the current visual & language inputs
plus existing knowledge stored in knowledge base.

Implements 'sensemaking' process; that is, process of integrating and re-organizing
perceived information from various modalities to establish a set of judgements structured
in such a manner that they can later be exploited for other symbolic reasoning tasks --
in light of the existing general knowledge held by the perceiving agent.

(I borrow the term 'sensemaking' from the discipline of symbolic science & psychology.
According to Klein (2006), sensemaking is "the process of creating situational awareness
and understanding in situations of high complexity or uncertainty in order to make decisions".)

Here, we resort to declarative programming to encode individual sensemaking problems into
logic programs (written in the language of weighted ASP), which are solved with a belief
propagation method.
"""
from .query import query
from .attribute import attribute
from ..lpmln import Literal, Rule, Program
from ..lpmln.utils import wrap_args


TAB = "\t"              # For use in format strings

EPS = 1e-10             # Value used for numerical stabilization
U_IN_PR = 0.99          # How much the agent values information provided by the user

class SymbolicReasonerModule:

    def __init__(self):
        self.concl_vis = None
        self.Q_answers = {}

        self.value_assignment = {}    # Store best assignments (tentative) obtained by reasoning
        self.word_senses = {}         # Store current estimate of symbol denotations

        self.mismatches = []

    def refresh(self):
        self.__init__()

    def sensemake_vis(self, exported_kb, visual_evidence):
        """
        Combine raw visual perception outputs from the vision module (predictions with
        confidence) with existing knowledge to make final verdicts on the state of
        affairs, 'all things considered'.

        Args:
            exported_kb: Output from KnowledgeBase().export_reasoning_program()
            visual_evidence: Output from KnowledgeBase().visual_evidence_from_scene()
        """
        # Solve to find the best models of the program
        prog = exported_kb + visual_evidence
        reg_gr_v = prog.compile()

        # Store sensemaking result as module state
        self.concl_vis = reg_gr_v, (exported_kb, visual_evidence)
    
    def resolve_symbol_semantics(self, dialogue_state, lexicon):
        """
        Find a fully specified mapping from symbols in discourse record to corresponding
        entities and concepts; in other words, perform reference resolution and word
        sense disambiguation.

        Args:
            dialogue_state: Current dialogue information state exported from the dialogue
                manager
            lexicon: Agent's lexicon, required for matching between environment entities
                vs. discourse referents for variable assignment
        """
        # Find the best estimate of referent value assignment
        sm_prog = Program()

        # Environmental referents
        occurring_atoms = set()
        for ent in dialogue_state.referents["env"][-1]:
            if ent not in occurring_atoms:
                sm_prog.add_absolute_rule(Rule(head=Literal("env", wrap_args(ent))))
                occurring_atoms.add(ent)

        # Discourse referents
        for rf, v in dialogue_state.referents["dis"].items():
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
        for rf, env in dialogue_state.assignment_hard.items():
            # Only process unresolved value assignments
            if rf in self.value_assignment: continue

            sm_prog.add_absolute_rule(
                Rule(body=[Literal("assign", [(rf, False), (env, False)], naf=True)])
            )

        # Understood dialogue record contents
        occurring_preds = set()
        for ti, (speaker, turn_clauses) in enumerate(dialogue_state.record):
            # Nothing particular to do with agent's own utterances
            if speaker == "A": continue

            for ci, ((_, _, ante, cons), _, _) in enumerate(turn_clauses):
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
                occurring_ent_refs = sum(cons_args+ante_args, [])
                # Only include entity referents (including 'x'), not eventuality referents
                occurring_ent_refs = {ref for ref in occurring_ent_refs if "x" in ref}

                if all(arg in dialogue_state.assignment_hard for arg in occurring_ent_refs):
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
            rf: None for rf in dialogue_state.referents["dis"]
            if rf not in best_assignment
        })

        tok2sym_map = [atom.args[:2] for atom in opt_models[0] if atom.name == "pred_token"]
        tok2sym_map = {
            tuple(token[0].split("_")): tuple(symbol[0].split("_"))
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

    def translate_dialogue_content(self, dialogue_state):
        """
        Translate logical content of dialogue record (which should be already
        ASP-compatible) based on current estimate of value assignment and word
        sense selection. Dismiss (replace with None) any utterances containing
        unresolved neologisms.
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
        for ti, (speaker, turn_clauses) in enumerate(dialogue_state.record):
            turn_translated = []
            for ci, ((gq, bvars, ante, cons), raw, mood) in enumerate(turn_clauses):
                # Translate consequent and antecedent by converting NL predicates to
                # denoted concepts
                if len(cons) > 0:
                    tr_cons = tuple(
                        encode_lits(lit, ti, ci, "c", li)
                        for li, lit in enumerate(cons)
                    )
                else:
                    tr_cons = None

                if len(ante) > 0:
                    tr_ante = tuple(
                        encode_lits(lit, ti, ci, "a", li)
                        for li, lit in enumerate(ante)
                    )
                else:
                    tr_ante = None

                turn_translated.append(((gq, bvars, tr_ante, tr_cons), raw, mood))

            record_translated.append((speaker, turn_translated))

        return record_translated

    @staticmethod
    def query(reg_gr, q_vars, event, restrictors=None):
        return query(reg_gr, q_vars, event, restrictors or {})

    @staticmethod
    def attribute(reg_gr, target_event, evidence, competing_evts, vetos=None):
        return attribute(reg_gr, target_event, evidence, competing_evts, vetos)
