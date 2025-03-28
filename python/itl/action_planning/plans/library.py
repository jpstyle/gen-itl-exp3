"""
Library of plans. Each plan consists of a sequence of action specifications.

Each action is specified by an 'action' method from some module of the agent and
a lambda function used to transform arguments as needed.
"""
class Val:
    """
    Designates value which can be either an object referrable within agent object, or
    pure data
    """
    def __init__(self, referrable=None, data=None):
        assert referrable is None or data is None, "Provide only one"

        self.referrable = referrable
        self.data = data
    
    def extract(self, agent_obj):
        """ Strip off the Val class wrapper to recover designated value """
        if self.referrable is not None:
            value = agent_obj
            for field in self.referrable:
                value = getattr(value, field)
            return value
        else:
            assert self.data is not None
            return self.data


library = {
    # Resolve neologism by requesting definitions or exemplars
    "address_neologism": [
        # Prepare answering utterance to generate
        {
            "action_method": Val(referrable=["comp_actions", "report_neologism"]),
            "action_args_getter": lambda x: (Val(data=x),)
        },
        # Generate whatever response queued
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ],

    # Handle unanswered question by first checking if it can be answered with agent's
    # current knowledge, and if so, adding to agenda to actually answer it
    "address_unanswered_Q": [
        # Prepare answering utterance to generate
        {
            "action_method": Val(referrable=["comp_actions", "attempt_Q"]),
            "action_args_getter": lambda x: (Val(data=x),)
        },
    ],

    # Answer a question by computing answer candidates, selecting an answer, translating
    # to natural language and then generating it
    "answer_Q": [
        # Prepare the answer to be uttered
        {
            "action_method": Val(referrable=["comp_actions", "prepare_answer_Q"]),
            "action_args_getter": lambda x: (Val(data=x),)
        },
        # Generate the prepared answer
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ],

    # Handle unexecuted command by first checking if it can be executed with agent's
    # current knowledge, and if so, adding to agenda to actually execute it
    "address_unexecuted_commands": [
        # Prepare answering utterance to generate
        {
            "action_method": Val(referrable=["comp_actions", "attempt_command"]),
            "action_args_getter": lambda x: (Val(data=x),)
        },
    ],

    # Execute a command by committing to an appropriate goal state based on perceived
    # environment state, translating goal state into ASP program fragment and running
    # ASP solver, then executing each primitive action in sequence in order
    "execute_command": [
        # Set goal, plan, execute
        {
            "action_method": Val(referrable=["comp_actions", "execute_command"]),
            "action_args_getter": lambda x: (Val(data=x),)
        }
    ],

    # After task execution is finished for an episode, run post-hoc analysis of
    # the final environment state to extract any learning signals
    "posthoc_episode_analysis": [
        # Called right before report of task completion in each episode
        {
            "action_method": Val(referrable=["comp_actions", "posthoc_episode_analysis"]),
            "action_args_getter": lambda x: ()
        }
    ],

    # Push a simple utterance (that doesn't need a specific logical form associated)
    # to generation buffer along with sentence mood
    "utter_simple": [
        # Push the speficied utterance with specified mood to buffer
        {
            "action_method": Val(referrable=["lang", "utter_simple"]),
            "action_args_getter": lambda x: (Val(data=x),)
        },
        # Generate the simple utterance
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ],

    # Generate whatever utterance is queued to be made in generation buffer
    "generate": [
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ],

    # Agent wasn't able to come up with a complete plan that ends up with
    # the desired end product due to uncertainty on types of parts used in
    # some subassembly. Report that it couldn't plan further and ask for help.
    # Expected to be called by langaugeless agents only
    "report_planning_failure": [
        # Called right before report of task completion in each episode
        {
            "action_method": Val(referrable=["comp_actions", "report_planning_failure"]),
            "action_args_getter": lambda x: ()
        },
        # Generate the prepared answer
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ],
}
