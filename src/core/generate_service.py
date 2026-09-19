from typing import Iterator, Optional
import logging
from api.schemas import ComposedDialogue
from core.orchestrator import Orchestrator
from core.guardrail import Guardrail
from core.tools.history import DialogueHistory
from core.state_manager import StateManager
from core.config.settings import Settings
from core.tools import pre_processing
from core.types.contexts import GameContext, NPCContext
from core.types.enums import MiddlewareState, ProfanityMode
from core.tools.errors import MiddlewareError, MiddlewareErrorCode

logger = logging.getLogger(__name__)

class GenerateService:
    """
    Use-case layer for request preconditions.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        guardrail: Guardrail,
        dialogue_history: DialogueHistory,
    ):
        self._orchestrator = orchestrator
        self._guardrail = guardrail
        self._dialogue_history = dialogue_history

    # -- Precondition checks -------------------------------------------------

    def _ensure_context_set(self) -> None:
        """
        Ensure the game context is set.
        """
        if self._orchestrator.game_context is None:
            raise MiddlewareError(code=MiddlewareErrorCode.CONTEXT_NOT_SET, errors=["Game context is not set."])

    def _ensure_idle(self) -> None:
        """
        Ensure the middleware is idle.
        """
        if not StateManager().is_in(MiddlewareState.IDLE):
            raise MiddlewareError(code=MiddlewareErrorCode.GENERATING, errors=["The middleware is busy generating."])

    def _ensure_history_not_empty(self) -> None:
        """
        Ensure the dialogue history is not empty.
        """
        if self._dialogue_history.is_empty():
            raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["Dialogue history is empty."])

    def _validate_and_normalize_npc_context(self, npc_context: NPCContext) -> NPCContext:
        """
        Validate and normalize the NPC context.
        """
        if not self._guardrail.validate_npc_context(npc_context):
            raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["The middleware refused the npc context."])
        return pre_processing.normalize_and_validate_npc_context(npc_context)

    def _profanity_warning_headers(self) -> dict[str, str]:
        """
        Return profanity warning headers based on the current settings.
        """
        if Settings().profanity_mode == ProfanityMode.STOP:
            return {
                "X-Profanity-Mode-Warning": (
                    "STOP mode profanity filter cannot be used in streaming, continuing dialog in CENSOR mode."
                )
            }
        return {}

    # -- Public use cases ------------------------------------------------------
    def set_game_context(self, game_context: GameContext) -> None:
        """
        Normalize and validate the game context, then sets it in the orchestrator. 
        """
        pre_processing.normalize_and_validate_game_context(game_context)
        self._orchestrator.set_game_context(game_context)

    def generate(self, npc_context: NPCContext) -> ComposedDialogue:
        """
        Generate a dialogue using the provided NPC context.
        """
        self._ensure_context_set()
        self._ensure_idle()
        npc_context = self._validate_and_normalize_npc_context(npc_context)

        dialogue = self._orchestrator.generate_dialogue(npc_context, None)

        if dialogue is None:
            raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["Dialogue generation was refused."])

        return dialogue

    def start_stream(self, npc_context: NPCContext) -> tuple[Iterator[str], dict[str, str]]:
        """
        Start a dialogue stream using the provided NPC context.
        """
        self._ensure_context_set()
        self._ensure_idle()

        headers = self._profanity_warning_headers()
        npc_context = self._validate_and_normalize_npc_context(npc_context)

        self._dialogue_history.clear_dialogue_history()

        stream = self._safe_stream(npc_context, last_player_choice=None)
        return stream, headers

    def continue_stream(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> tuple[Iterator[str], dict[str, str]]:
        """
        Continue a dialogue stream using the provided NPC context and last player choice.
        """
        self._ensure_context_set()
        self._ensure_history_not_empty()
        self._ensure_idle()

        headers = self._profanity_warning_headers()
        npc_context = self._validate_and_normalize_npc_context(npc_context)

        stream = self._safe_stream(npc_context, last_player_choice=last_player_choice)
        return stream, headers

    def _safe_stream(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> Iterator[str]:
        """
        Safely generate a dialogue stream, handling any exceptions.
        """
        try:
            yield from self._orchestrator.generate_dialogue_stream(npc_context, last_player_choice)
        except Exception as e:
            logger.error(f"Unhandled error during dialogue stream: {e}")
            yield "\n[STREAM_ERROR]\n"