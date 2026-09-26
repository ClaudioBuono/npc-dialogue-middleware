from __future__ import annotations
from pydantic import ValidationError
from api.schemas import ComposedDialogue
from core.config.settings import Settings
from core.contract_builder import ContractBuilder
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import PreProcessingError, ValidationErrorCode
from core.tools.output_composer import DialogueOutputComposer
from core.types.contexts import GameContext, NPCContext
from core.types.dataclasses import JudgeIssue


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
        """Rewrite the dialogue to resolve the given issues.

        Args:
            composed_dialogue: The dialogue to repair, as originally composed.
            game_context: Global world/setting context, used to keep the
                rewrite grounded.
            npc_context: The NPC's profile (persona, tone, relation to the
                player), used to keep the rewrite in character.
            issues: The problems flagged by the judge that the healer must fix.

        Returns:
            A new ComposedDialogue with the flagged issues resolved.

        Raises:
            PreProcessingError: If the healer's raw output does not match
                the expected schema once parsed.
        """
        healer_contract = self.contract_builder.build_healer_contract(composed_dialogue, game_context, npc_context, issues)

        # Slightly higher temperature than the judge: the healer needs to
        # rewrite naturally, not just reproduce a deterministic pattern.
        healer_output_raw = self._client.generate(healer_contract, temperature = Settings().llm.temperature) # TODO: Test lower temperatures

        try:
            # Re-validates the healed dialogue against the same schema as
            # the original composition step, so a malformed healer output
            # never silently reaches the rest of the pipeline.
            composed_dialogue = self.dialogue_composer.compose_dialogue(npc_context, healer_output_raw)

        except ValidationError as e:
            raise PreProcessingError(code=ValidationErrorCode.INVALID_VALUE, errors=[f"Healer output does not match expected schema: {e}", f"Raw output: {healer_output_raw}"])

        return composed_dialogue