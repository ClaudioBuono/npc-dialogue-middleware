# core/contract_builder.py
from typing import Any
from core.types.contract import Contract
from core.types.contexts import GameContext, NPCContext
from core.llm.system_prompts import RULES_SYSTEM_PROMPT


class ContractBuilder:
    """
    Builds the Contract that will be sent to the LLM,
    combining game_context and npc_context.
    """

    # Ouput schema 
    # TODO: To improve making it dynamic, options depend on the intent
    _OUTPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dialogue": {
                "type": "string"
            },
            "player_options": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["dialogue", "player_options"],
        "additionalProperties": False
    }

    def build(self, game_context: GameContext, npc_context: NPCContext) -> Contract:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(game_context, npc_context)
        return Contract(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=self._OUTPUT_SCHEMA
        )

    def _build_system_prompt(self) -> str:
        return (
            f"{RULES_SYSTEM_PROMPT}\n"
            f"Expected response schema: {self._OUTPUT_SCHEMA}. No text outside the JSON."
        )

    def _build_user_prompt(self, game_context: GameContext, npc_context: NPCContext) -> str:
        return (
            f"Setting: {game_context.environment}\n"
            f"Epoch: {game_context.epoch}\n"
            f"Plot: {game_context.plot}\n\n"
            f"NPC: {npc_context.name}\n"
            f"Dialogue intent: {npc_context.intent}\n"
            f"NPC personality: {npc_context.personality}\n\n"
            "Generate a dialogue for this NPC consistent with the context above."
        )