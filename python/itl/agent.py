"""
Outermost wrapper containing ITL agent API
"""
import os
import re
import copy
import pickle
import logging

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
        self.lt_mem = LongTermMemoryModule(cfg)

        # Provide access to methods in comp_actions
        self.comp_actions = CompositeActions(self)

        # Load agent model from specified path
        if "model_path" in cfg.agent:
            self.load_model(cfg.agent.model_path)

        # Agent learning strategy params
        self.strat_generic = cfg.agent.strat_generic
        self.strat_assent = cfg.agent.strat_assent

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

    def loop(self, v_usr_in=None, l_usr_in=None, pointing=None, new_env=False):
        """
        Single agent activity loop, called with inputs of visual scene, natural
        language utterances from user and affiliated gestures (demonstrative
        references by pointing in our case). new_vis flag indicates whether
        the visual scene input is new.
        """
        self._vis_inp(usr_in=v_usr_in, new_env=new_env)
        self._lang_inp(usr_in=l_usr_in, pointing=pointing)
        self._update_belief(pointing=pointing)
        act_out = self._act()

        return act_out

    def save_model(self, ckpt_path):
        """
        Save current snapshot of the agent's long-term knowledge as torch checkpoint;
        in the current scope of the research, by 'long-term knowledge' we are referring
        to the following information:

        - Vision model; feature extractor backbone most importantly, and concept-specific
            vectors
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
            ckpt = torch.load(local_ckpt_path)
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
        self.vision.latest_inputs.append(usr_in)

    def _lang_inp(self, usr_in, pointing):
        """ Handle provided language input (from user) """
        if usr_in is None:
            usr_in = []
        elif not isinstance(usr_in, list):
            assert isinstance(usr_in, str)
            usr_in = [usr_in]

        assert len(usr_in) == len(pointing)

        if len(usr_in) > 0:
            parsed_input = self.lang.semantic.nl_parse(usr_in, pointing)
            self.lang.latest_input = parsed_input
        else:
            self.lang.latest_input = None

    def _update_belief(self, pointing):
        """ Form beliefs based on visual and/or language input """
        # Lasting storage of pointing info
        if pointing is None:
            pointing = {}

        # Flag whether environment has 'changed'
        new_env = len(self.vision.latest_inputs) == 1

        # Translated dialogue record and visual context from currently stored values
        # (scene may or may not change)
        prev_translated = self.symbolic.translate_dialogue_content(self.lang.dialogue)
        prev_vis_scene = self.vision.scene
        prev_pr_prog = self.symbolic.concl_vis[1][1] if self.symbolic.concl_vis else None
        prev_kb = self.kb_snap
        prev_context = (prev_vis_scene, prev_pr_prog, prev_kb)

        # Some cleaning steps needed whenever visual context changes
        if len(self.vision.latest_inputs) == 1:
            # Refresh dialogue manager & symbolic reasoning module states
            self.lang.dialogue.refresh()
            self.symbolic.refresh()

            # Update KB snapshot on episode-basis
            self.kb_snap = copy.deepcopy(self.lt_mem.kb)

        # Index of latest dialogue turn
        ti_last = len(self.lang.dialogue.record)

        # Set of new visual concepts (equivalently, neologisms) newly registered
        # during the loop
        novel_concepts = set()

        # Keep updating beliefs until there's no more immediately exploitable learning
        # opportunities
        xb_updated = False      # Whether learning happened at 'neural'-level (in exemplar base)
        kb_updated = False      # Whether learning happened at 'symbolic'-level (in knowledge base)
        while True:
            ###################################################################
            ##                  Processing perceived inputs                  ##
            ###################################################################

            # Run visual prediction (only) when not in observation mode
            if self.observed_demo is None:
                if xb_updated:
                    # Concept exemplar base updated, need reclassification while keeping
                    # the discovered objects and embeddings intact
                    self.vision.predict(
                        None, self.lt_mem.exemplars, reclassify=True, visualize=False
                    )
                else:
                    # Ground raw visual perception with scene graph generation module
                    self.vision.predict(
                        self.vision.latest_inputs[-1], self.lt_mem.exemplars,
                        lexicon=self.lt_mem.lexicon
                    )

            if self.lang.latest_input is not None:
                # Revert to pre-update dialogue state at the start of each loop iteration
                self.lang.dialogue.record = self.lang.dialogue.record[:ti_last]

                # Understand the user input in the context of the dialogue
                self.lang.understand(self.lang.latest_input, pointing=pointing)

            # Inform the language module of the current visual context
            self.lang.situate(self.vision.scene)

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
                    self.vision.predict(
                        None, self.lt_mem.exemplars, masks=masks, visualize=False
                    )

                    ents_updated = True     # Set flag for another round of sensemaking

            ###################################################################
            ##       Sensemaking via synthesis of perception+knowledge       ##
            ###################################################################

            if self.observed_demo is None and (ents_updated or xb_updated or kb_updated):
                # Sensemaking from vision input only
                exported_kb = self.lt_mem.kb.export_reasoning_program()
                visual_evidence = self.lt_mem.kb.visual_evidence_from_scene(self.vision.scene)
                self.symbolic.sensemake_vis(exported_kb, visual_evidence)
                self.lang.dialogue.sensemaking_v_snaps[ti_last] = self.symbolic.concl_vis

            if self.lang.latest_input is not None:
                # Reference & word sense resolution to connect vision & discourse
                self.symbolic.resolve_symbol_semantics(
                    self.lang.dialogue, self.lt_mem.lexicon
                )

            # Translate dialogue record into processable format based on the result
            # of symbolic.resolve_symbol_semantics
            translated = self.symbolic.translate_dialogue_content(self.lang.dialogue)

            # Check here whether a user demonstration is pending and agent should
            # switch to 'observation mode', where agent hands over the control to
            # user and begins to record subsequent events. Postpone any type of
            # learning until the observation mode is switched off.

            # First recognize any future indicative 'demonstrate how' by user
            demonstrated_event = None
            speaker, turn_clauses = translated[-1]; ti = len(translated)-1
            for ci, ((_, _, _, cons), _, mood) in enumerate(turn_clauses):
                demo_lit = [lit for lit in cons if lit.name=="sp_demonstrate"]

                if len(demo_lit) == 0: break

                (demo_evt, _), (demonstrator, _), (demo_target, _) = demo_lit[0].args
                clause_info = self.lang.dialogue.clause_info[demo_evt]

                if not (clause_info["mood"] == "." and clause_info["tense"] == "future"):
                    break
                if not (demonstrator == "_user" and speaker == "U"): break

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
                demonstrated_event = translated[demo_ti][1][demo_ci][0][-1]
                self.observed_demo = (demonstrated_event, (demo_ti, demo_ci))
                break           # Break here without proceeding

            if self.observed_demo is not None:
                ###################################################################
                ##          Recording ongoing task demonstration by user         ##
                ###################################################################

                # Set of statements made by user at this time step
                user_statements = set.union(*[set(lf[-1]) for lf, _, _ in translated[-1][1]])

                # If a statement provides labeled instance of the 'target concept'
                # of a "Build X" task, consider demonstration is finished
                build_pred = self.lt_mem.lexicon.s2d[("va", "build")][0]
                build_pred = f"{build_pred[0]}_{build_pred[1]}"
                task_preds = {lit.name for lit in self.observed_demo[0]}

                if ({lit.name for lit in user_statements} | {build_pred}) <= task_preds:
                    # End of demonstration recognized; collect & learn from demo data
                    user_annotations = [
                        (ti, contents) for ti, (spk, contents) in enumerate(translated)
                        if spk=="U"
                    ]
                    demo_ti = self.observed_demo[1][0]
                    aligned_demo_data = [
                        (img, contents, env_refs)
                        for (ti, contents), img, env_refs in zip(
                            user_annotations,
                            self.vision.latest_inputs,
                            self.lang.dialogue.referents["env"],
                        )
                        if ti >= demo_ti
                    ]

                    # All parts of learning from demonstration take place here
                    self.comp_actions.analyze_demonstration(aligned_demo_data)

                    # Back to normal execution mode
                    self.observed_demo = None

                break

            else:
                ###################################################################
                ##           Identify & exploit learning opportunities           ##
                ###################################################################

                # Disable learning when agent is in test mode
                if self.cfg.agent.test_mode: break

                # Resetting flags
                xb_updated = False
                kb_updated = False

                # Generic statements to be added to KB
                generics = []

                # Collect previous factual statements and questions made during this
                # dialogue
                prev_statements = []; prev_Qs = []
                for ti, (speaker, turn_clauses) in enumerate(prev_translated):
                    for ci, ((rule, ques), raw) in enumerate(turn_clauses):
                        # Factual statement
                        if rule is not None and len(rule[0])==1 and rule[1] is None:
                            prev_statements.append(((ti, ci), (speaker, rule)))

                        # Question
                        if ques is not None:
                            # Here, `rule` represents presuppositions included in `ques`
                            prev_Qs.append(((ti, ci), (speaker, ques, rule, raw)))

                # Process translated dialogue record to do the following:
                #   - Identify recognition mismatch btw. user provided vs. agent
                #   - Identify visual concept confusion
                #   - Identify new generic rules to be integrated into KB
                for ti, (speaker, turn_clauses) in enumerate(translated):
                    if speaker != "U": continue

                    for ci, ((gq, bvars, ante, cons), raw, mood) in enumerate(turn_clauses):
                        # Skip any non-indicative statements (or presuppositions)
                        if mood != ".": continue

                        # Disregard clause if it does not discuss the task domain
                        clause_info = self.lang.dialogue.clause_info[f"t{ti}c{ci}"]
                        if not clause_info.get("domain_describing"):
                            continue

                        # Identify learning opportunities; i.e., any deviations from the
                        # agent's estimated states of affairs, generic rules delivered
                        # via NL generic statements, or acknowledgements (positive or lack
                        # of negative)
                        self.comp_actions.identify_mismatch(rule)
                        self.comp_actions.identify_confusion(
                            rule, prev_statements, novel_concepts
                        )
                        self.comp_actions.identify_acknowledgement(
                            rule, prev_statements, prev_context
                        )
                        self.comp_actions.identify_generics(
                            rule, raw, prev_Qs, generics
                        )

                # By default, treat lack of any negative acknowledgements to an agent's statement
                # as positive acknowledgement
                prev_or_curr = "prev" if new_env else "curr"
                for (ti, ci), (speaker, (statement, _)) in prev_statements:
                    if speaker != "A": continue         # Not interested

                    stm_ind = (prev_or_curr, ti, ci)
                    if stm_ind not in self.lang.dialogue.acknowledged_stms:
                        acknowledgement_data = (statement, True, prev_context)
                        self.lang.dialogue.acknowledged_stms[stm_ind] = acknowledgement_data

                # Update knowledge base with obtained generic statements
                for rule, w_pr, knowledge_source, knowledge_type in generics:
                    kb_updated |= self.lt_mem.kb.add(
                        rule, w_pr, knowledge_source, knowledge_type
                    )

                # Handle neologisms
                xb_updated |= self.comp_actions.handle_neologism(
                    novel_concepts, self.lang.dialogue
                )

                # Terminate the loop when 'equilibrium' is reached
                if not (xb_updated or kb_updated):
                    break

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
            return_val = [("generate", ("# Observing", {}))]

        else:
            # Agent is in full control, identify and resolve any pending agenda items

            # This ordering ensures any knowledge updates (that doesn't require interaction)
            # happen first, addressing & answering questions happen next, finally asking
            # any questions afterwards
            for m in self.symbolic.mismatches:
                self.planner.agenda.append(("address_mismatch", m))
            for a in self.lang.dialogue.acknowledged_stms.items():
                self.planner.agenda.append(("address_acknowledgement", a))
            for ti, ci in self.lang.dialogue.unanswered_Qs:
                self.planner.agenda.append(("address_unanswered_Q", (ti, ci)))
            for ti, ci in self.lang.dialogue.unexecuted_commands:
                self.planner.agenda.append(("address_unexecuted_commands", (ti, ci)))
            for n in self.lang.unresolved_neologisms:
                self.planner.agenda.append(("address_neologism", n))
            for c in self.vision.confusions:
                self.planner.agenda.append(("address_confusion", c))

            return_val = []

            num_resolved_items = 0; unresolved_items = []
            while True:
                # Loop through agenda stack items from the top; new agenda items may
                # be pushed to the top by certain plans
                while len(self.planner.agenda) > 0:
                    # Pop the top agenda item from the stack
                    todo_state, todo_args = self.planner.agenda.pop(0)

                    # Check if this item can be resolved at this stage and if so, obtain
                    # appropriate plan (sequence of actions) for resolving the item
                    plan = self.planner.obtain_plan(todo_state)

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
                            if act_out is not None:
                                return_val += act_out

                                # Note) We don't need to consider any sort of plan failures
                                # right now, but when that happens (should be identifiable
                                # from act_out value), in principle, will need to break
                                # from plan execution and add to unresolved item list
                                if False:       # Plan failure check not implemented
                                    unresolved_items.append((todo_state, todo_args))
                                    break
                        else:
                            num_resolved_items += 1
                    else:
                        # Plan not found, agenda item unresolved
                        unresolved_items.append((todo_state, todo_args))

                # Any unresolved items back to agenda stack
                self.planner.agenda = unresolved_items

                if num_resolved_items == 0 or len(self.planner.agenda) == 0:
                    # No resolvable agenda item any more, or stack clear
                    if len(return_val) == 0 and self.lang.latest_input is not None:
                        # No specific reaction to utter, acknowledge any user input
                        self.planner.agenda.append(("acknowledge", None))
                    else:
                        # Break loop with return vals
                        break

        return return_val
