"""
Action planning module API that exposes only the high-level functionalities
required by the ITL agent: generate and follow plans as and when necessary,
in order to 1) fulfill goals externally assigned by users (generally more
grounded, physical, low-level) or to recover any of the agent's internal
maintenance goals violated by environmental changes (generally more cognitive,
social, high-level)
"""
from .plans.library import library


class ActionPlannerModule:
    
    def __init__(self):
        self.agenda = []

    def obtain_plan(self, todo):
        """
        Obtain appropriate plan from plan library (maybe this could be extended later
        with automated planning...)
        """
        return library.get(todo)
