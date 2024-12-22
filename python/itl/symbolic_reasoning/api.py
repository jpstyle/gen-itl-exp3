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
from ..lpmln import Program


TAB = "\t"              # For use in format strings

EPS = 1e-10             # Value used for numerical stabilization
U_IN_PR = 0.99          # How much the agent values information provided by the user

class SymbolicReasonerModule:

    def __init__(self):
        self.concl_vis = None
        self.Q_answers = {}

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
        exported_kb = exported_kb or Program()

        # Solve to find the best models of the program
        prog = exported_kb + visual_evidence
        reg_gr_v = prog.compile()

        # Store sensemaking result as module state
        self.concl_vis = reg_gr_v, (exported_kb, visual_evidence)

    @staticmethod
    def query(reg_gr, q_vars, event, restrictors=None):
        return query(reg_gr, q_vars, event, restrictors or {})

    @staticmethod
    def attribute(reg_gr, target_event, evidence, competing_evts, vetos=None):
        return attribute(reg_gr, target_event, evidence, competing_evts, vetos)
