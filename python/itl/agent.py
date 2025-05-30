"""
Outermost wrapper containing ITL agent API
"""
import os
import re
import copy
import pickle
import logging
from collections import deque

import torch

from .vision import VisionModule
from .lang import LanguageModule
from .memory import LongTermMemoryModule
from .symbolic_reasoning import SymbolicReasonerModule
from .action_planning import ActionPlannerModule
from .comp_actions import CompositeActions

logger = logging.getLogger(__name__)


class ITLAgent:

    def __init__(self, cfg):
        self.cfg = cfg

        # Initialize component modules
        self.vision = VisionModule(cfg)
        self.lang = LanguageModule()
        self.symbolic = SymbolicReasonerModule()
        self.planner = ActionPlannerModule()
        self.lt_mem = LongTermMemoryModule()

        # Provide access to methods in comp_actions
        self.comp_actions = CompositeActions(self)

        # Load agent model from specified path
        if "agent_model_path" in cfg.exp:
            self.load_model(cfg.exp.agent_model_path)

        # (Fields below would categorize as 'working memory' in conventional
        # cognitive architectures...)

        # Bookkeeping pairs of visual concepts that confused the agent, which
        # are resolved by asking 'concept-diff' questions to the user. Jusk ask
        # once to get answers as symbolic generic rules when the agent is aware
        # of the confusion for the first time, for each concept pair.
        # (In a sense, this kinda overlaps with the notion of 'common ground'?
        # May consider fulfilling this later...)
        self.confused_no_more = set()

        # Snapshot of KB, to be taken at the beginning of every training episode
        self.kb_snap = copy.deepcopy(self.lt_mem.kb)

        self.observed_demo = None
        self.execution_paused = False
        self.interrupted = False

    def loop(
            self, v_usr_in=None, l_usr_in=None,
            speaker=None, pointing=None, new_env=False
        ):
        """
        Single agent activity loop, called with inputs of visual scene, natural
        language utterances from user and affiliated gestures (demonstrative
        references by pointing in our case). new_env flag indicates whether
        the agent is situated in a new environment
        """
        self._vis_inp(usr_in=v_usr_in, new_env=new_env)
        self._lang_inp(usr_in=l_usr_in, pointing=pointing, speaker=speaker)
        self._update_belief(pointing=pointing, new_env=new_env)
        act_out = self._act()

        return act_out

    def save_model(self, ckpt_path):
        """
        Save current snapshot of the agent's long-term knowledge as torch checkpoint;
        in the current scope of the research, by 'long-term knowledge' we are referring
        to the following information:

        - Vision model; feature extractor backbone
        - Knowledge stored in long-term memory module:
            - Symbolic knowledge base, including generalized symbolic knowledge represented
                as logic programming rules
            - Visual exemplar base, including positive/negative exemplars of visual concepts
                represented as internal feature vectors along with original image patches from
                which the vectors are obtained
            - Lexicon, including associations between words (linguistic form) and their
                denotations (linguistic meaning; here, visual concepts)
        """
        ckpt = {
            "vision": {
                "inventories": self.vision.inventories
            },
            "lt_mem": {
                "exemplars": vars(self.lt_mem.exemplars),
                "kb": vars(self.lt_mem.kb),
                "lexicon": vars(self.lt_mem.lexicon)
            }
        }

        torch.save(ckpt, ckpt_path)
        logger.info(f"Saved current agent model at {ckpt_path}")

    def load_model(self, ckpt_path):
        """
        Load from a torch checkpoint to initialize the agent; the checkpoint may contain
        a snapshot of agent knowledge obtained as an output of self.save_model() evoked
        previously, or just pre-trained weights of the vision module only (likely generated
        as output of the vision module's training API)
        """
        # Resolve path to checkpoint
        assert os.path.exists(ckpt_path)
        local_ckpt_path = ckpt_path

        # Load agent model checkpoint file
        try:
            ckpt = torch.load(local_ckpt_path, weights_only=False)
        except RuntimeError:
            with open(local_ckpt_path, "rb") as f:
                ckpt = pickle.load(f)

        # Fill in module components with loaded data
        for module_name, module_data in ckpt.items():
            for module_component, component_data in module_data.items():
                if isinstance(component_data, dict):
                    for component_prop, prop_data in component_data.items():
                        component = getattr(getattr(self, module_name), module_component)
                        setattr(component, component_prop, prop_data)
                else:
                    module = getattr(self, module_name)
                    setattr(module, module_component, component_data)

    def _vis_inp(self, usr_in, new_env):
        """ Handle provided visual input """
        if new_env:
            self.vision.latest_inputs = []
            self.vision.scene = None
        self.vision.latest_inputs.append(usr_in)

    def _lang_inp(self, usr_in, pointing, speaker):
        """ Handle provided language input (could be from user or self) """
        if usr_in is None:
            usr_in = []
        elif not isinstance(usr_in, list):
            assert isinstance(usr_in, str)
            usr_in = [usr_in]

        pointing = pointing or {}
        speaker = speaker or []

        assert len(usr_in) == len(pointing) == len(speaker)

        usr_in_flt = []; pointing_flt = []; speaker_flt = []
        for utt, dem_refs, spk in zip(usr_in, pointing, speaker):
            # Process any non-mask references
            dem_refs = {
                crange: next(
                    oi for oi, obj in self.vision.scene.items()
                    if ref == obj["env_handle"]
                ) if isinstance(ref, str) else ref
                for crange, ref in dem_refs.items()
            }

            if spk == "Teacher":
                # Teacher somehow reacted by providing corrective feedback
                # or signaling silent observation. Clear any null item on
                # the top of the agenda so that agent can proceed with the
                # pending action plan.
                if len(self.planner.agenda) == 0:
                    pass
                elif self.planner.agenda[0] == ('execute_command', (None, None)):
                    self.planner.agenda.popleft()

                if utt == "Stop." and not self.execution_paused:
                    # Enter pause mode, where any execution related agenda
                    # items will be ignored. Interruption also implies
                    # currently remaining plan is invalid and replanning
                    # would be needed.
                    self.execution_paused = True
                    self.interrupted = True     # Also log interruption
                    # Replanning to agenda
                    goal_action, goal_target = self.planner.execution_state["plan_goal"]
                    goal_action = self.lt_mem.lexicon.s2d[("va", goal_action)][0][1]
                    self.planner.execution_state["last_scrapped_plan"] = \
                        self.planner.agenda
                    self.planner.agenda = deque([
                        ("execute_command", (goal_action, (goal_target, True)))
                    ])
                if utt == "Continue." and self.execution_paused:
                    # Exit pause mode
                    self.execution_paused = False

            if spk == "Student" and utt == "# Idle":
                # Frontend agent has been idle for long enough time, interpret
                # silence as signal that agent should continue whatever it was
                # doing
                if len(self.planner.agenda) == 0:
                    pass
                elif self.planner.agenda[0] == ('execute_command', (None, None)):
                    self.planner.agenda.popleft()

            if not (utt == "# Observing" or utt == "# Idle"):
                # Process any general case NL input
                usr_in_flt.append(utt)
                pointing_flt.append(dem_refs)
                speaker_flt.append(spk)

        if len(usr_in_flt) > 0:
            parsed_input = self.lang.semantic.nl_parse(usr_in_flt, pointing_flt)
            self.lang.latest_input = (parsed_input, pointing_flt, speaker_flt)
        else:
            self.lang.latest_input = None

    def _update_belief(self, pointing, new_env):
        """ Form beliefs based on visual and/or language input """
        # Lasting storage of pointing info
        if pointing is None: pointing = {}

        # Some cleaning steps needed whenever visual context changes
        if new_env:
            # Refresh agent states to prepare for new episode
            self.lang.dialogue.refresh()
            self.symbolic.refresh()
            self.planner.refresh()
            self.observed_demo = None
            self.execution_paused = False
            self.interrupted = False

            # Update KB snapshot on episode-basis
            self.kb_snap = copy.deepcopy(self.lt_mem.kb)

        # Index of latest dialogue turn
        ti_last = len(self.lang.dialogue.record)

        exemplars = self.lt_mem.exemplars       # Shortcut var

        # Keep updating beliefs until there's no more immediately exploitable learning
        # opportunities
        xb_updated = False      # Whether learning happened at 'neural'-level (in exemplar base)
        kb_updated = False      # Whether learning happened at 'symbolic'-level (in knowledge base)
        while True:
            ###################################################################
            ##                 Processing perceived inputs                   ##
            ###################################################################

            # Run visual prediction (only) when not in demo observation mode
            if self.observed_demo is None:
                if xb_updated:
                    # Concept exemplar base updated, need reclassification while keeping
                    # the discovered objects and embeddings intact
                    self.vision.predict(None, exemplars, reclassify=True)
                else:
                    if new_env:
                        if self.vision.latest_inputs[-1] is not None:
                            # Ground raw visual perception by running ensemble prediction;
                            # latest input is None only if GT masks were provided as cheat
                            # sheet when new_env=True
                            self.vision.predict(self.vision.latest_inputs[-1], exemplars)
                    else:
                        # Just process whole image with visual feature extractor, avoiding
                        # ensemble prediction
                        self.vision.predict(self.vision.latest_inputs[-1], exemplars, masks={})

            if self.lang.latest_input is not None:
                # Revert to pre-update dialogue state at the start of each loop iteration
                self.lang.dialogue.record = self.lang.dialogue.record[:ti_last]

                # New environment entities info for current timeframe
                self.lang.dialogue.referents["env"].append(
                    { "_self": None, "_user": None }         # Always include self and user
                )

                # Inform the language module of the current visual context (only)
                # when not in demo observation mode
                if self.observed_demo is None: self.lang.situate(self.vision.scene)

                # Understand the user input in the context of the dialogue
                self.lang.understand(self.lang.latest_input)

            ents_updated = False
            if self.vision.scene is not None and self.observed_demo is None:
                # If a new entity is registered as a result of understanding the latest
                # input, re-run vision module to update with new predictions for it
                new_ents = set(self.lang.dialogue.referents["env"][-1]) - set(self.vision.scene)
                new_ents.remove("_self"); new_ents.remove("_user")
                if len(new_ents) > 0:
                    masks = {
                        ent: self.lang.dialogue.referents["env"][-1][ent]["mask"]
                        for ent in new_ents
                    }

                    # Incrementally predict on the designated bbox
                    self.vision.predict(None, self.lt_mem.exemplars, masks=masks)

                    ents_updated = True     # Set flag for another round of sensemaking

            ###################################################################
            ##       Sensemaking via synthesis of perception+knowledge       ##
            ###################################################################

            # if self.observed_demo is None and (ents_updated or xb_updated or kb_updated):
            #     # Sensemaking from vision input only
            #     exported_kb = self.lt_mem.kb.export_reasoning_program()
            #     visual_evidence = self.lt_mem.kb.visual_evidence_from_scene(self.vision.scene)
            #     self.symbolic.sensemake_vis(exported_kb, visual_evidence)

            if self.lang.latest_input is not None:
                # Reference & word sense resolution to connect vision & discourse
                self.lang.dialogue.resolve_symbol_semantics(self.lt_mem.lexicon)

            # Resetting flags
            xb_updated = False
            kb_updated = False

            # Handle any neologisms
            xb_updated |= self.comp_actions.handle_neologism(self.lang.dialogue)

            # Translate dialogue record into processable format based on the result
            # of lang.resolve_symbol_semantics
            resolved_record = self.lang.dialogue.export_resolved_record()

            # Check here whether a user demonstration is pending and agent should
            # switch to 'observation mode', where agent hands over the control to
            # user and begins to record subsequent events. Postpone any type of
            # learning until the observation mode is switched off.
            demonstrated_event = None
            if len(resolved_record) > 0:
                # First recognize any future indicative 'demonstrate how' by user
                speaker, turn_clauses = resolved_record[-1]; ti = len(resolved_record)-1
                for ci, ((_, _, _, cons), _, clause_info) in enumerate(turn_clauses):
                    if len(cons) == 0: break

                    demo_lit = [lit for lit in cons if lit.name=="sp_demonstrate"]
                    if len(demo_lit) == 0: break

                    _, (demonstrator, _), (demo_target, _) = demo_lit[0].args

                    if not (clause_info["mood"] == "." and clause_info["tense"] == "future"):
                        break
                    if not (demonstrator == "_user" and speaker == "Teacher"): break

                    how_lit = [
                        lit for lit in cons
                        if lit.name=="sp_manner" and lit.args[0][0]==demo_target
                    ]
                    if len(how_lit) > 0:
                        demonstrated_event = how_lit[0].args[1][0]
                        break

            if demonstrated_event is not None:
                # Demonstration notice recognized, record corresponding event pointer
                # and switch to observation mode
                demonstrated_event = re.findall(r"t(\d+)c(\d+)", demonstrated_event)[0]
                demo_ti = int(demonstrated_event[0])
                demo_ci = int(demonstrated_event[1])
                demonstrated_event = resolved_record[demo_ti][1][demo_ci][0][-1]
                self.observed_demo = (demonstrated_event, (demo_ti, demo_ci))
                # Resolve any unexecuted command with the same event signature
                resolved_commands = []
                demo_evt_sgn = {lit.name for lit in demonstrated_event}
                for ti, ci in self.lang.dialogue.unexecuted_commands:
                    command = resolved_record[ti][1][ci][0][-1]
                    command_evt_sgn = {lit.name for lit in command}
                    if demo_evt_sgn <= command_evt_sgn:
                        resolved_commands.append((ti, ci))
                for utt_pointer in resolved_commands:
                    del self.lang.dialogue.unexecuted_commands[utt_pointer]
                break           # Break here without proceeding

            if self.observed_demo is not None:
                ###################################################################
                ##          Recording ongoing task demonstration by user         ##
                ###################################################################

                # Set of statements made by user at this turn
                _, user_statements = resolved_record[-1]

                # If a statement provides labeled instance of the 'target concept'
                # of a "Build X" task, consider demonstration is finished
                build_pred = self.lt_mem.lexicon.s2d[("va", "build")][0]
                build_pred = f"{build_pred[0]}_{build_pred[1]}"
                task_preds = {lit.name for lit in self.observed_demo[0]}

                if any(
                    ({lit.name for lit in user_last_statement} | {build_pred}) <= task_preds \
                        and clause_info["mood"] == "."
                    for (_, _, _, user_last_statement), _, clause_info in user_statements
                ):
                    # End of demonstration recognized; collect & learn from demo data
                    user_annotations = [
                        (ti, contents) for ti, (spk, contents) in enumerate(resolved_record)
                        if spk=="Teacher"
                    ]
                    demo_vis = [img for img in self.vision.latest_inputs if img is not None]
                    demo_ti = self.observed_demo[1][0]
                    aligned_demo_data = [
                        (img, contents)
                        for (ti, contents), img in zip(user_annotations, demo_vis)
                        if ti >= demo_ti
                    ]

                    # All parts of learning from demonstration take place here
                    self.comp_actions.analyze_demonstration(aligned_demo_data)

                    # Back to normal execution mode
                    self.observed_demo = None

                break

            else:
                ###################################################################
                ##                Handle environment action effect               ##
                ###################################################################

                # Process feedback from Unity environment generated due to executing
                # a primitive action (in particular, 'pick_up' and 'assemble') so
                # that agent can determine how to carry on with the remainder of the
                # ongoing action plan. Process only the last one(s), if any.
                last_action_effect = None
                for ti, (speaker, turn_clauses) in enumerate(resolved_record):
                    if ti < ti_last: continue           # Already examined

                    for ci, ((_, _, _, cons), raw, _) in enumerate(turn_clauses):
                        if not raw.startswith("# Effect: "): continue
                        last_action_effect = cons
                        last_actor = speaker

                if last_action_effect is not None:
                    self.comp_actions.handle_action_effect(
                        last_action_effect, last_actor
                    )

                ###################################################################
                ##           Identify & exploit learning opportunities           ##
                ###################################################################

                # Process translated dialogue record to do the following:
                #   - Identify recognition mismatch btw. user provided vs. agent
                #   - Identify new generic rules to be integrated into KB
                for ti, (speaker, turn_clauses) in enumerate(resolved_record):
                    if speaker != "Teacher": continue   # Only process user input
                    if ti < ti_last: continue           # Already examined

                    for ci, clause in enumerate(turn_clauses):
                        statement, raw, clause_info = clause

                        # Skip any non-indicative statements (or presuppositions)
                        if clause_info["mood"] != ".": continue
                        # Skip any nonlinguistic annotations
                        if raw.startswith("# "): continue

                        # Identify learning opportunities; i.e., any deviations from the
                        # agent's estimated states of affairs or generic rules delivered
                        # via NL generic statements
                        xb_updated |= self.comp_actions.identify_mismatch(statement)
                        kb_updated |= self.comp_actions.identify_generics(statement, raw)

                if not (xb_updated or kb_updated):
                    # Terminate the loop when 'equilibrium' is reached
                    break

        # Examine and update the current set of unresolved neologisms based
        # on agent's current knowledge state
        self.comp_actions.resolve_neologisms()

    def _act(self):
        """
        Just eagerly try to resolve each item in agenda as much as possible, generating
        and performing actions until no more agenda items can be resolved for now. I
        wonder if we'll ever need a more sophisticated mechanism than this simple, greedy
        method for a good while?
        """
        ## Generate agenda items from maintenance goals
        # Currently, the maintenance goals are not to leave:
        #   - any unaddressed neologism which is unresolvable
        #   - any unaddressed recognition inconsistency btw. agent and user
        #   - any unanswered question that is unaddressed
        #   - any unexecuted command that is unattempted
        #
        # Ideally, this is to be accomplished declaratively by properly setting up formal
        # maintenance goals and then performing automated planning or something to come
        # up with right sequence of actions to be added to agenda. However, the ad-hoc
        # code below (+ plan library in action_planner/plans/library.py) will do
        # for our purpose right now; we will see later if we'll ever need to generalize
        # and implement the said procedure.)

        if self.observed_demo is not None:
            # Agent currently in observation mode; just signal that agent is properly
            # recording user demonstration
            return [("generate", ("# Observing", {}))]

        # If didn't return above, Agent is in full control, identify and resolve any
        # pending items in agenda
        for n in self.lang.unresolved_neologisms:
            self.planner.agenda.appendleft(("address_neologism", n))
        for utt_pointer in self.lang.dialogue.unanswered_Qs:
            self.planner.agenda.appendleft(("address_unanswered_Q", utt_pointer))
        for utt_pointer in self.lang.dialogue.unexecuted_commands:
            self.planner.agenda.appendleft(("address_unexecuted_commands", utt_pointer))

        return_val = []
        act_on_env = False

        while True:
            # Loop through agenda items from the top; new agenda items may be
            # appended on the fly to the left or right of the deque
            num_resolved_items = 0; unresolved_items = []
            while len(self.planner.agenda) > 0:
                # Pop the top agenda item from the top
                todo_type, todo_args = self.planner.agenda.popleft()

                # In pause mode, ignore any `execute_command` items, as well as
                # any success/failure report and post-hoc analysis (since task won't
                # ever progress)
                if self.execution_paused:
                    if todo_type == "execute_command":
                        unresolved_items.append((todo_type, todo_args))
                        continue
                    if todo_type == "posthoc_episode_analysis":
                        unresolved_items.append((todo_type, todo_args))
                        continue
                    if todo_type == "report_planning_failure":
                        unresolved_items.append((todo_type, todo_args))
                        continue
                    if todo_type == "utter_simple" and todo_args[0] == "Done.":
                        unresolved_items.append((todo_type, todo_args))
                        continue

                # Check if this item can be resolved at this stage and if so, obtain
                # appropriate plan (sequence of actions) for resolving the item
                plan = self.planner.obtain_plan(todo_type)

                if plan is not None:
                    # Perform plan actions
                    for action in plan:
                        act_method = action["action_method"].extract(self)
                        act_args = action["action_args_getter"](todo_args)
                        if type(act_args) == tuple:
                            act_args = tuple(arg.extract(self) for arg in act_args)
                        else:
                            act_args = (act_args.extract(self),)

                        act_out = act_method(*act_args)
                        return_val += act_out or []
                    else:
                        num_resolved_items += 1
                else:
                    # Plan not found, agenda item unresolved
                    unresolved_items.append((todo_type, todo_args))

                if todo_type == "execute_command" and len(return_val) > 0:
                    # If the processed agenda item is to execute some command,
                    # and the agent has decided to act upon the environment
                    # to fulfill the command, need to break here before proceeding
                    # to the remainining items in the agenda
                    act_on_env = True
                    unresolved_items += self.planner.agenda
                    break

            # Any unresolved items back to agenda
            self.planner.agenda = deque(unresolved_items)

            if act_on_env:
                # Must break here to act upon the environment
                break

            if num_resolved_items == 0:
                # No resolvable agenda item any more

                # Has user done/said anything this turn?
                user_inputs = [
                    (raw, spk)
                    for parse, _, spk in zip(*self.lang.latest_input)
                    for raw in parse["source"].values()
                ] if self.lang.latest_input is not None else []
                user_nl_inputs = [
                    (raw, spk) for raw, spk in user_inputs
                    if spk == "Teacher" and not raw.startswith("#")
                ]

                if len(user_inputs) > 0 and len(return_val) == 0:
                    if self.execution_paused:
                        # Entered (only) when agent, most likely to be language-
                        # less player type, had to pause execution due to its
                        # imperfect knowledge and thus needs to wait for user's
                        # partial demo. Signal that it is observing user's actions.
                        return_val.append(("generate", ("# Observing", {})))

                if len(user_nl_inputs) > 0 and len(return_val) == 0:
                    # Break loop with cool acknowledgement to whichever NL
                    # input from user so that user doesn't feel awkward. This
                    # will terminate the episode, so don't say this if there's
                    # still a pending, unfulfilled command.
                    return_val.append(("generate", ("OK.", {})))

                # Break loop with return vals
                break

        return return_val
