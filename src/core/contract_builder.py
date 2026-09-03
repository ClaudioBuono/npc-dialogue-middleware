from typing import Any, Dict
from core.config.settings import Settings
from core.types.dataclasses import Contract
from core.types.contexts import *
from core.llm.prompts import *
from core.types.contexts import GameContext


class ContractBuilder:
    """
    Builds the Contract that will be sent to the LLM,
    combining game_context and npc_context.
    """

    def build(self, game_context: GameContext, npc_context: NPCContext, dialogue_history: List[Dict[str,str]]) -> Contract:
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
        system_prompt = self._build_system_prompt(game_context)

        # Quest Intent
        if isinstance(npc_context.intent, Quest):
            user_prompt = self._build_user_prompt_quest(npc_context, dialogue_history)
            output_schema = self._build_output_schema(npc_context)

            return Contract(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema
            )
        
        # Dialogue Intent
        if isinstance(npc_context.intent, Dialogue):
            user_prompt = self._build_user_prompt_dialogue(npc_context, dialogue_history)
            output_schema = self._build_output_schema(npc_context)

            return Contract(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema
            )

        raise ValueError(f"Unsupported intent type: {type(npc_context.intent)}")

    def _build_system_prompt(self, game_context: GameContext) -> str:
        """Builds the common part of the system prompt, shared across all dialogue intents.

        Establishes the model's role, the game world context, and general output
        rules.
        """
        world_context = WORLD_CONTEXT_PROMPT.format(
            environment=game_context.environment,
            epoch=game_context.epoch,
            world_state=game_context.world_state,
        )

        prompts = [
            ROLE_PROMPT,
            world_context,
        ]

        if game_context.main_character_description:
            prompts.append(
                MAIN_CHARACTER_PROMPT.format(
                    main_character_description=game_context.main_character_description
                )
            )

        prompts.append(GENERAL_RULES_PROMPT)

        if Settings().prompt_fairness_filter:
            prompts.append(FAIRNESS_BASE_RULES_PROMPT)

        return "\n\n".join(prompts)

    
    def _build_prompt_npc_context(self, npc_context: NPCContext, dialogue_history: List[Dict[str,str]]) -> str:
        """
        Builds the formatted NPC prompt context from the provided NPC data.
        Iterates through required fields (name, age, personality, etc.) and 
        appends optional background or behavioral fields if they are available.
        """
        
        lines = [
            NPC_CONTEXT_BASE_QUEST_PROMPT if isinstance(npc_context.intent, Quest) else NPC_CONTEXT_BASE_DIALOGUE_PROMPT,
            NPC_FIELDS_PROMPT.format(
                name = npc_context.name,
                age = npc_context.age,
                personality = npc_context.personality,
                context = npc_context.context,
                talkativeness = npc_context.talkativeness.value,
                main_character_relation = npc_context.main_character_relation
            )
        ]

        if npc_context.recent_plot:
            lines.append(f"- Recent Events: {npc_context.recent_plot}")
        if npc_context.visual_description:
            lines.append(f"- Visual Description: {npc_context.visual_description}")
        if npc_context.backstory:
            lines.append(f"- Backstory: {npc_context.backstory}")
        if npc_context.language:
            lines.append(f"- Language: {npc_context.language}")

        if npc_context.intent.more_info:
            lines.append(f"- Additional informations: {npc_context.intent.more_info}")


        result = "\n".join(lines)

        if dialogue_history:
            dialogue_history_prompt = DIALOGUE_HISTORY_PROMPT.format(dialogue_history = dialogue_history)

            result = "\n".join(
                [
                    result,
                    dialogue_history_prompt
                ]
            )

        return result


    def _build_prompt_quest(self, quest: Quest) -> str:
        """
        Builds the prompt section specific to quest details when the intent is a Quest.
        Maps out core quest parameters like objective, description, and rewards,
        and appends specific prompts to handle quest acceptance/refusal branches.
        """
        lines = [
            QUEST_BASE_PROMPT,
            f"- Objective: {quest.objective}",
        ]

        if quest.name:
            lines.append(f"- Name: {quest.name}")
        if quest.description:
            lines.append(f"- Description: {quest.description}")
        if quest.location:
            lines.append(f"- Location: {quest.location}")
        if quest.reward:
            lines.append(f"- Reward: {quest.reward}")

        lines.append("\n")


        result = "\n".join(lines)
        return result

    # Merges the NPC context prompt, base Dialogue prompt and Quest prompt
    # for building the user prompt when the intent is a Quest.
    def _build_user_prompt_quest(self, npc_context: NPCContext, dialogue_history: List[Dict[str,str]]) -> str:
        npc_section = self._build_prompt_npc_context(npc_context, dialogue_history)
        quest_section = self._build_prompt_quest(npc_context.intent)

        result = "\n".join([npc_section, DIALOGUE_BASE_PROMPT, quest_section])

        if npc_context.intent.has_choice or npc_context.intent.has_options or npc_context.intent.must_use_expression:
            rules_section = self._build_rules(npc_context.intent)
            result = "\n".join([result, rules_section])

        return result

    # Merges the NPC context prompt and base Dialogue prompt for building
    # the user prompt when the intent is a plain Dialogue (no quest).
    def _build_user_prompt_dialogue(self, npc_context: NPCContext, dialogue_history: List[Dict[str,str]]) -> str:
        npc_section = self._build_prompt_npc_context(npc_context, dialogue_history)
        rules_section = self._build_rules(npc_context.intent)

        result = "\n".join([npc_section, DIALOGUE_BASE_PROMPT])

        if npc_context.intent.has_options or npc_context.intent.must_use_expression:
            rules_section = self._build_rules(npc_context.intent)
            result = "\n".join([result, rules_section])

        return result

    def _build_rules(self, intent: Quest | Dialogue) -> str:

        lines = [DIALOGUE_RULES_PROMPT]
        
        if isinstance(intent, Dialogue):
            if intent.must_use_expression:
                lines.append(f"- You MUST USE the following expression in the dialogue: {intent.must_use_expression}")

            # TODO: Number of max options configurable
            if intent.has_options:
                lines.append(DIALOGUE_OPTIONS_PROMPT.format(number_of_options = Settings().number_of_options))


        if isinstance(intent, Quest):

            if intent.has_choice:
                lines.append(QUEST_CHOICE_PROMPT)


        result = "\n".join(lines)
        return result

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
        (Quest included), based on has_options.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        # More dialogue options
        if dialogue.has_options:
            properties["dialogue_options"] = {
                "type": "array",
                "items": {"type": "string"},
                "description": "Must include the player can choose "
                            "from in response (e.g. asking for more details)."
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