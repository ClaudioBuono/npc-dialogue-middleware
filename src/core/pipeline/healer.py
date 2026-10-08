from __future__ import annotations
import logging
import time
from pydantic import ValidationError
from api.schemas import ComposedDialogue
from core.configuration.settings import Settings
from core.generation.contract_builder import ContractBuilder
from core.generation.output_composer import DialogueOutputComposer
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
        """Produce a corrected version of a dialogue based on Judger feedback.

        Builds a healer contract from the dialogue, the contexts and the
        reported issues, sends it to the LLM, and composes the raw output
        back into a ComposedDialogue.

        Args:
            composed_dialogue: The dialogue to be corrected.
            game_context: Contextual information about the game/world state.
            npc_context: Contextual information about the NPC speaking the dialogue.
            issues: The issues reported by the Judger that the healer must fix.

        Returns:
            A new ComposedDialogue with the issues addressed.

        Raises:
            PreProcessingError: If the healer output does not match the
                expected schema (ValidationError during composition).
            Exception: Any exception raised by the LLM client during
                generation is logged and re-raised unchanged.
        """
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