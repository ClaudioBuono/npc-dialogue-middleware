from typing import Any, Dict
from api.schemas import ComposedDialogue
from core.configuration.settings import Settings
from core.helpers.formatters import format_composed_dialogue, format_dialogue_history, format_game_context, format_intent_data, format_judge_issues, format_judge_questions, format_npc_context
from core.types.dataclasses import Contract, JudgeIssue, JudgeQuestion
from core.types.contexts import *
from core.llm.prompts import *
from core.types.contexts import GameContext

class ContractBuilder:
    """
    Builds the Contract that will be sent to the LLM,
    combining game_context and npc_context.
    """
    def build_dialogue_contract(self, game_context: GameContext, npc_context: NPCContext, dialogue_history: List[Dict[str,str]]) -> Contract:
        """
        Orchestrates the construction of the generation Contract for the LLM.

        This method acts as the main entry point, combining global game world configurations 
        with specific NPC context data. It evaluates the user's intent type to dynamically 
        assemble the tailored system prompt, user prompt, and JSON validation schema.

        Args:
            game_context (GameContext): The global state, epoch, and environment data of the game.
            npc_context (NPCContext): The profile, dialogue constraints, and intent of the target NPC.

        Returns:
            Contract: A structured container holding the system prompt, user prompt, and output schema.

        Raises:
            ValueError: If the `npc_context.intent` does not match any supported types (e.g., Quest, Dialogue).
        """

        # Builds System prompt
        system_prompt = self._build_dialogue_system_prompt(npc_context)
        user_prompt = self._build_dialogue_user_prompt(game_context, npc_context, dialogue_history)
        output_schema = self._build_output_schema(npc_context)
        return Contract(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema
        )


    def build_judge_contract(
        self,
        composed_dialogue: ComposedDialogue,
        game_context: GameContext,
        npc_context: NPCContext,
        questions: List[JudgeQuestion],
        bad_reasons: str | None = None,
    ) -> Contract:
        """Build the execution contract for the judge LLM evaluation task.

        Formats the dialogue, game, and NPC contexts into system and user prompts,
        and defines the expected JSON schema output for validating dialogue compliance
        against fixed evaluation rules.

        Args:
            composed_dialogue (ComposedDialogue): The composed dialogue instance to be evaluated.
            game_context (GameContext): The global game state and environment context.
            npc_context (NPCContext): The NPC profile, personality, and dialogue parameters.
            questions (List[JudgeQuestion]): The yes/no questions the judge must answer.
            feedback (str | None): Optional correction message describing problems found
                in the judge's previous answer (e.g. invalid reasons). When provided, it is
                appended to the user prompt so the judge can regenerate its answer.

        Returns:
            Contract: A Contract instance containing the assembled system prompt, user prompt,
            and JSON output schema.
        """
        system_prompt = self._build_judge_system_prompt()
        user_prompt = self._build_judge_user_prompt(composed_dialogue, game_context, npc_context, questions)
        output_schema = self._build_judge_output_schema([q.id for q in questions])

        if bad_reasons:
            judger_prompt = JUDGE_FEEDBACK_PROMPT_TEMPLATE.format(details=bad_reasons)
            user_prompt = f"{user_prompt}\n\n{judger_prompt}"

        return Contract(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )

    def build_healer_contract(self, composed_dialogue: ComposedDialogue, game_context: GameContext, npc_context: NPCContext, issues: List[JudgeIssue]) -> Contract:
        """Build the execution contract for the healer LLM repair task.

        Formats the dialogue, game and NPC contexts, and the issues flagged by
        the judge into system and user prompts, and defines the expected JSON
        schema output for the repaired dialogue.

        Args:
            composed_dialogue (ComposedDialogue): The dialogue instance to be repaired.
            game_context (GameContext): The global game state and environment context.
            npc_context (NPCContext): The NPC profile, personality, and dialogue parameters.
            issues (List[JudgeIssue]): The problems flagged by the judge that the healer must resolve.

        Returns:
            Contract: A Contract instance containing the assembled system prompt, user prompt,
            and JSON output schema.
        """
        system_prompt = self._build_healer_system_prompt()
        user_prompt = self._build_healer_user_prompt(composed_dialogue, game_context, npc_context, issues)
        output_schema = self._build_output_schema(npc_context)

        return Contract(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=output_schema,
        )
    

    # Helper methods for Dialogue prompt -------------------------------------------------------

    def _build_dialogue_system_prompt(self, npc_context: NPCContext) -> str:
        """Builds the system prompt shared across all dialogue intents."""
        settings = Settings()
        intent = npc_context.intent

        is_quest = intent.type != "Dialogue"
        has_options = intent.number_of_options
        has_choice = is_quest and intent.has_choice  # accept/refuse exist only for quests

        blocks = [
            ROLE_PROMPT,
            INPUT_FORMAT_PROMPT,
            SECURITY_RULES_PROMPT,
            INTENT_FIELDS_PROMPT,
            LANGUAGE_RULE_TEMPLATE.format(language=settings.language.name),
            CONTENT_RULES_PROMPT,
        ]

        if is_quest:
            blocks.append(QUEST_CONTENT_RULES_PROMPT)

        blocks += [GROUNDING_RULES_PROMPT, STYLE_RULES_PROMPT]

        if settings.fairness_filter:
            blocks.append(FAIRNESS_BASE_RULES_PROMPT)

        blocks += [NPC_FIELD_GUIDANCE_PROMPT, TALKATIVENESS_GUIDE_PROMPT]

        player_options = self._build_player_options_section(
            has_options=has_options,
            has_choice=has_choice,
            number_of_options=npc_context.intent.number_of_options,
        )
        if player_options:
            blocks.append(player_options)

        blocks.append(OUTPUT_FORMAT_PROMPT)

        return "\n\n".join(blocks)


    @staticmethod
    def _build_player_options_section(has_options: bool, has_choice: bool, number_of_options: int) -> str | None:
        """Single '# PLAYER OPTIONS' section, composed from independent parts."""
        if not (has_options or has_choice):
            return None

        lines = [PLAYER_OPTIONS_HEADER_PROMPT]

        if has_options:
            lines.append(DIALOGUE_OPTIONS_TEMPLATE.format(number_of_options=number_of_options))
        if has_choice:
            lines.append(QUEST_CHOICE_PROMPT)
        if has_options and has_choice:
            lines.append(DIALOGUE_OPTIONS_NO_DECISION_PROMPT)

        return "\n".join(lines)  # single newline: it's one section


    def _build_dialogue_user_prompt(self, game_context: GameContext, npc_context: NPCContext, history: List[Dict[str,str]]) -> str:
        settings = Settings()
        intent = npc_context.intent
        is_quest = intent.type != "Dialogue"

        history_text = format_dialogue_history(history, npc_context.name)
        intent_text = format_intent_data(intent)

        blocks = [
            WORLD_CONTEXT_TEMPLATE.format(game_context=format_game_context(game_context)),
            NPC_TEMPLATE.format(npc_context=format_npc_context(npc_context)),
        ]
        if history_text:
            blocks.append(DIALOGUE_HISTORY_TEMPLATE.format(dialogue_history=history_text))
        if intent_text:
            blocks.append(INTENT_TEMPLATE.format(intent_data=intent_text))

        has_choice = is_quest and intent.has_choice
        blocks.append(self._build_task_section(
            is_quest, intent.number_of_options, has_choice, npc_context.intent.number_of_options
        ))

        return "\n\n".join(blocks)
    
    @staticmethod
    def _build_task_section(is_quest: bool, has_options: bool, has_choice: bool, number_of_options: int) -> str:
        lines = [TASK_QUEST_PROMPT if is_quest else TASK_DIALOGUE_PROMPT]

        deliverables = [TASK_DIALOGUE_FIELD_PROMPT]
        if has_options:
            deliverables.append(TASK_OPTIONS_FIELD_TEMPLATE.format(number_of_options=number_of_options))
        if has_choice:
            deliverables.append(TASK_CHOICE_FIELD_PROMPT)

        # Only mention the output list when there is more than the dialogue itself
        if len(deliverables) > 1:
            lines += ["", TASK_OUTPUT_HEADER_PROMPT, *deliverables]

        return "\n".join(lines)

    def _build_output_schema(self, npc_context: NPCContext) -> dict[str, Any]:
        """
        Builds the JSON output schema dynamically. Every intent is at minimum
        a Dialogue, so the base 'player_options' schema is always built first;
        if the intent is also a Quest, its extra fields (accept/refuse) are
        merged on top.
        """
        properties: dict[str, Any] = {
            "dialogue": {
                "type": "string",
                "description": "The dialogue line(s) spoken by the NPC."
            }
        }
        required = ["dialogue"]

        intent = npc_context.intent

        # Every intent is a Dialogue, so the base schema always applies.
        player_options_schema = self._build_player_options_schema_dialogue(intent)

        # Quest adds extra fields (accept/refuse) on top of the base.
        if isinstance(intent, Quest):
            player_options_schema = self._extend_player_options_schema_quest(
                intent, player_options_schema
            )

        if player_options_schema is not None:
            properties["player_options"] = player_options_schema
            required.append("player_options")

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
        return schema

    def _build_player_options_schema_dialogue(self, dialogue: Dialogue) -> dict[str, Any] | None:
        """
        Builds the base 'player_options' schema shared by every Dialogue intent
        (Quest included), based on dialogue.number_of_options.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        # More dialogue options
        if dialogue.number_of_options > 0:
            properties["dialogue_options"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": f"Must include {dialogue.number_of_options} options the player can choose from in response (e.g. asking for more details)."
                            " Must NEVER include explicit Accept and Refuse options."
            }
            required.append("dialogue_options")

        if not properties:
            return None

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }

    def _extend_player_options_schema_quest(self, quest: Quest, base_schema: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Extends a base player_options schema with Quest-specific fields
        (accept/refuse), added on top of whatever the base Dialogue already built.
        """
        properties: dict[str, Any] = dict(base_schema["properties"]) if base_schema else {}
        required: list[str] = list(base_schema.get("required", [])) if base_schema else []

        if quest.has_choice:
            properties["accept"] = {
                "type": "string",
                "description": "Player dialogue option to accept the quest."
            }
            properties["refuse"] = {
                "type": "string",
                "description": "Player dialogue option to refuse the quest."
            }
            required.extend(["accept", "refuse"])

        if not properties:
            return None

        schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False
        }
        if required:
            schema["required"] = required
            
        return schema

    # Helper methods for Judge prompt -------------------------------------------------------
    def _build_judge_system_prompt(self) -> str:
        """
        Builds the system prompt for the Judge, which includes the base prompt and rules.
        """
        system_prompt_lines = [
            JUDGE_BASE_PROMPT,
            JUDGE_RULES_PROMPT,
        ]

        return "\n\n".join(system_prompt_lines)

    def _build_judge_user_prompt(self, composed_dialogue: ComposedDialogue, game_context: GameContext, npc_context: NPCContext, questions: List[JudgeQuestion]) -> str:
        """
        Builds the user prompt for the Judge, which includes the formatted dialogue, NPC context, game context, and judge questions.
        """
        formatted_dialogue = format_composed_dialogue(composed_dialogue, include_intent=True)
        formatted_npc_context = format_npc_context(npc_context)
        formatted_game_context = format_game_context(game_context)

        judge_body_prompt = JUDGE_BODY_TEMPLATE.format(
            npc_context=formatted_npc_context,
            game_context=formatted_game_context,
            dialogue=formatted_dialogue,
        )

        user_prompt_lines = [
            JUDGE_TASK_TEMPLATE.format(questions=format_judge_questions(questions)),
            judge_body_prompt,
        ]

        return "\n\n".join(user_prompt_lines)


    def _build_judge_output_schema(self, question_ids: list[str], include_reason: bool = True) -> dict[str, Any]:
        """
        Builds the JSON output schema for the judge dynamically.
        The judge must return exactly one boolean answer per question, so the
        'id' field is restricted to the ids actually asked. A 'reason' field is
        added to each answer unless include_reason is False.
        """
        item_properties: dict[str, Any] = {
            "id": {
                "type": "string",
                "enum": list(question_ids),
                "description": "The id of the question being answered."
            }
        }
        item_required = ["id"]

        if include_reason:
            item_properties["reason"] = {
                "type": "string",
                "description": (
                    "If answer is false, explain concretely what is wrong so a "
                    "second editor can fix it without re-reading the "
                    "full context: quote or point to the specific problematic part "
                    "of the dialogue, and state what it should say or do instead. "
                    "If answer is true, a short confirmation is enough."
                )
            }
            item_required.append("reason")

        item_properties["answer"] = {
            "type": "boolean",
            "description": "True if the condition holds, False otherwise."
        }
        item_required.append("answer")

        item_schema = {
            "type": "object",
            "properties": item_properties,
            "required": item_required,
            "additionalProperties": False
        }

        schema = {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": item_schema,
                }
            },
            "required": ["answers"],
            "additionalProperties": False
        }
        return schema

    def _build_healer_system_prompt(self) -> str:
        """
        Builds the system prompt for the Healer, which includes the base prompt and rules.
        """
        system_prompt_lines = [
            HEALER_BASE_PROMPT,
            HEALER_RULES_PROMPT,
        ]

        return "\n\n".join(system_prompt_lines)
    
    def _build_healer_user_prompt(self, composed_dialogue: ComposedDialogue, game_context: GameContext, npc_context: NPCContext, issues: list[JudgeIssue]) -> str:
        """
        Builds the user prompt for the Healer, which includes the formatted dialogue, game context, NPC context, judge found issues.
        """
        formatted_dialogue = format_composed_dialogue(composed_dialogue, include_intent=True)
        formatted_npc_context = format_npc_context(npc_context)
        formatted_game_context = format_game_context(game_context)

        healer_body_prompt = HEALER_BODY_TEMPLATE.format(
            npc_context=formatted_npc_context,
            game_context=formatted_game_context,
            dialogue=formatted_dialogue,
        )

        user_prompt_lines = [
            HEALER_TASK_TEMPLATE.format(issues=format_judge_issues(issues)),
            healer_body_prompt,
        ]

        return "\n\n".join(user_prompt_lines)