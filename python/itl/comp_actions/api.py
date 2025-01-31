"""
Agent actions API that implements and exposes composite actions. By 'composite',
it refers to actions requiring interplay between more than one agent modules.
By 'action', it may refer to internal (cognitive, epistemic) or external (physical,
environment-interactive) actions. Actions may be evoked by method name in plans
obtained from the action planning module.
"""
from .interact import *
from .learn import *

class CompositeActions:
    def __init__(self, agent):
        # Register actor agent
        self.agent = agent

    # Environmental interactions
    def attempt_Q(self, utt_pointer):
        return attempt_Q(self.agent, utt_pointer)
    def prepare_answer_Q(self, utt_pointer):
        return prepare_answer_Q(self.agent, utt_pointer)
    def attempt_command(self, utt_pointer):
        return attempt_command(self.agent, utt_pointer)
    def execute_command(self, action_spec):
        return execute_command(self.agent, action_spec)
    def handle_action_effect(self, effect, actor):
        return handle_action_effect(self.agent, effect, actor)

    # Cognitive actions
    def identify_mismatch(self, rule):
        return identify_mismatch(self.agent, rule)
    def identify_generics(self, rule, prev_Qs, provenance):
        return identify_generics(self.agent, rule, prev_Qs, provenance)
    def identify_acknowledgement(self, rule, prev_statements, prev_context):
        return identify_acknowledgement(self.agent, rule, prev_statements, prev_context)
    def handle_acknowledgement(self, acknowledgement_info):
        return handle_acknowledgement(self.agent, acknowledgement_info)
    def handle_neologism(self, novel_concepts, dialogue_state):
        return handle_neologism(self.agent, novel_concepts, dialogue_state)
    def report_neologism(self, neologism):
        return report_neologism(self.agent, neologism)
    def analyze_demonstration(self, demo):
        return analyze_demonstration(self.agent, demo)