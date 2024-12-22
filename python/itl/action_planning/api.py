"""
Action planning module API that exposes only the high-level functionalities
required by the ITL agent: generate and follow plans as and when necessary,
in order to 1) fulfill goals externally assigned by users (generally more
grounded, physical, low-level) or to recover any of the agent's internal
maintenance goals violated by environmental changes (generally more cognitive,
social, high-level)
"""
from collections import deque
from .plans.library import library


class ActionPlannerModule:
    
    def __init__(self):
        # collection of to-dos implemented as double-ended queue, each entry
        # of which is a tuple of (type, parameters)
        self.agenda = deque()

        # For storing whichever state values that need to be tracked along execution
        # of some plan consisting of multiple to-dos
        self.execution_state = {}

    def obtain_plan(self, todo):
        """
        Obtain appropriate plan from plan library (maybe this could be extended later
        with automated planning...)
        """
        return library.get(todo)
