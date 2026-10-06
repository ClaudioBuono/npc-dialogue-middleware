from __future__ import annotations
import logging
import time
from pydantic import ValidationError
from api.schemas import ComposedDialogue
from core.configuration.settings import Settings
from core.generation.contract_builder import ContractBuilder
from core.generation.output_composer import DialogueOutputComposer
from core.helpers.formatters import format_composed_dialogue
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import PreProcessingError, ValidationErrorCode
from core.tools.errors import PreProcessingError
from core.types.contexts import GameContext, NPCContext
from core.types.dataclasses import JudgeIssue

logger = logging.getLogger(__name__)

class Healer:
    """
    Repairs an NPC dialogue by rewriting it to resolve the issues flagged
    during judging, while preserving the NPC's persona and the dialogue's
    original intent as much as possible.
    """
    def __init__(self, contract_builder: ContractBuilder, dialogue_composer: DialogueOutputComposer) -> None:
        """Initialize the healer with the contract builder and dialogue composer.

        Args:
            contract_builder: Builds the healer's system/user prompt and
                output schema from the dialogue, context, and issues.
            dialogue_composer: Parses the raw LLM output back into a
                validated ComposedDialogue.
        """
        self.contract_builder = contract_builder
        self.dialogue_composer = dialogue_composer


    def set_client(self, client: OpenAICompatibleClient) -> None:
        """Inject the LLM client to use for generation.

        Kept separate from __init__ so the client can be swapped or
        configured after construction (e.g. once settings are loaded).
        """
        self._client = client


    def heal_dialogue(self, composed_dialogue: ComposedDialogue, game_context: GameContext, npc_context: NPCContext, issues: list[JudgeIssue]) -> ComposedDialogue:
        logger.info(
            "Healing dialogue: %d issue(s) (%s)",
            len(issues), ", ".join(i.category for i in issues),
        )

        healer_contract = self.contract_builder.build_healer_contract(composed_dialogue, game_context, npc_context, issues)

        try:
            healer_output_raw = self._client.generate(healer_contract, temperature=Settings().llm.healer_temperature)  # invariato
        except Exception:
            logger.exception("Healer LLM call failed")
            raise

        try:
            composed_dialogue = self.dialogue_composer.compose_dialogue(npc_context, healer_output_raw)
        except ValidationError as e:
            logger.error(
                "Healer output does not match schema (%d validation errors). Raw output: %s",
                e.error_count(), healer_output_raw,
            )
            raise PreProcessingError(
                code=ValidationErrorCode.INVALID_VALUE,
                errors=[f"Healer output does not match expected schema: {e}", f"Raw output: {healer_output_raw}"],
            )

        logger.info("Dialogue healed successfully")
        return composed_dialogue