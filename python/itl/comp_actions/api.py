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
    def report_planning_failure(self):
        return report_planning_failure(self.agent)
    def handle_action_effect(self, effect, actor):
        return handle_action_effect(self.agent, effect, actor)

    # Cognitive actions
    def identify_mismatch(self, statement):
        return identify_mismatch(self.agent, statement)
    def identify_generics(self, statement, provenance):
        return identify_generics(self.agent, statement, provenance)
    def handle_neologism(self, dialogue_state):
        return handle_neologism(self.agent, dialogue_state)
    def resolve_neologisms(self):
        return resolve_neologisms(self.agent)
    def report_neologism(self, neologism):
        return report_neologism(self.agent, neologism)
    def analyze_demonstration(self, demo):
        return analyze_demonstration(self.agent, demo)
    def posthoc_episode_analysis(self):
        return posthoc_episode_analysis(self.agent)
