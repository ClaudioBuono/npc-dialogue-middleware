from typing import Any, Optional, Iterator
import logging
from api.schemas import ComposedDialogue
from core.pipeline.refiner import Refiner
from core.infrastructure.state_manager import StateManager
from core.configuration.settings import Settings
from core.generation.contract_builder import ContractBuilder
from core.generation.dialogue_generator import DialogueGenerator
from core.pipeline.guardrail import Guardrail
from core.generation.history import DialogueHistory
from core.llm.openai_client import OpenAICompatibleClient
from core.generation.output_composer import DialogueOutputComposer
from core.routing.router import LLMRouter
from core.helpers.formatters import to_json_format
from core.types.contexts import Dialogue, GameContext, NPCContext
from core.types.enums import MiddlewareState, ProfanityMode

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Coordinates all core components of the middleware, providing a single entry point for generating NPC dialogue and player options.
    Implemented as a Singleton: there is a single instance for the entire process.
    Use `Orchestrator.get_instance()` to retrieve it from any other module/method.
    Args:
        pre_processor:    Instance of PreProcessor.
        llm_handler:      Instance of LLMHandler.
    """
    
    _instance: "Orchestrator | None" = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Orchestrator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        guardrail: Guardrail,
        dialogue_history: DialogueHistory,
        contract_builder: ContractBuilder,
        llm_router: LLMRouter,
        dialogue_generator: DialogueGenerator,
        dialogue_composer: DialogueOutputComposer,
        refiner: Refiner,
    ) -> None:
        if self._initialized:
            return

        self.contract_builder = contract_builder
        self.llm_router = llm_router
        self.dialogue_generator = dialogue_generator
        self.dialogue_composer = dialogue_composer
        self.dialogue_history = dialogue_history
        self.guardrail = guardrail
        self.refiner = refiner

        self.game_context: GameContext | None = None

        self._iterations = 0
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "Orchestrator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Test utility: reset the singleton instance."""
        cls._instance = None

    # Orchestrator Methods ----------------------------------------------------------------------------


    """Holds validated game context for middleware operations."""
    def set_game_context(self, game_context: GameContext) -> None:
        """Set the game context by validating environment, epoch, and lore."""

        StateManager().transition_to(MiddlewareState.SETTING_CONTEXT)

        self.game_context = game_context

        StateManager().transition_to(MiddlewareState.IDLE)
        

    def generate_dialogue(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> ComposedDialogue | None:
        
        """Generate NPC dialogue using the NPC and game context."""

        logger.info(f"Generating dialogue for NPC '{npc_context.name}'")
            
        StateManager().transition_to(MiddlewareState.GENERATING)


        if last_player_choice:
            self.dialogue_history.add_player_dialogue_to_history(last_player_choice)
            logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")


        contract = self.contract_builder.build_dialogue_contract(self.game_context, npc_context, self.dialogue_history.get_dialogue_history())

        client: OpenAICompatibleClient = self.llm_router.select_model(game_context = self.game_context, npc_context = npc_context)
        logger.debug(f"Selected LLM client: {type(client).__name__}")

        self.dialogue_generator.set_client(client)
        raw_dialogue: str = self.dialogue_generator.generate(contract)

        composed_dialogue = self.dialogue_composer.compose_dialogue(npc_context, raw_dialogue)

        self.refiner.set_client(client)

        composed_dialogue = self.refiner.refine_dialogue(composed_dialogue, npc_context, self.game_context)

        self.dialogue_history.add_npc_dialogue_to_history(composed_dialogue)

        StateManager().transition_to(MiddlewareState.IDLE)

        logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")

        return composed_dialogue

    
    def generate_dialogue_stream(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> Iterator[str]:
        """Generate NPC dialogue using the NPC and game context via streaming."""

        StateManager().transition_to(MiddlewareState.GENERATING)
        logger.info(f"Generating dialogue stream for NPC '{npc_context.name}'")

        try:
            raw_stream = self._start_dialogue_stream(npc_context, last_player_choice)

            released: list[str] = []
            for text in self._guarded_stream(raw_stream):
                released.append(text)
                yield text

            self._save_streamed_dialogue(npc_context, "".join(released))
        finally:
            StateManager().transition_to(MiddlewareState.IDLE)

    def _start_dialogue_stream(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> Iterator[str]:
        """Update history, build the contract, select the client and start the raw stream."""
        if last_player_choice:
            self.dialogue_history.add_player_dialogue_to_history(last_player_choice)
            logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")

        contract = self.contract_builder.build_dialogue_contract(
            self.game_context, npc_context, self.dialogue_history.get_dialogue_history()
        )

        client: OpenAICompatibleClient = self.llm_router.select_model(
            game_context=self.game_context, npc_context=npc_context
        )
        logger.debug(f"Selected LLM client: {type(client).__name__}")

        self.dialogue_generator.set_client(client)
        return self.dialogue_generator.generate_stream(contract)

    def _guarded_stream(self, stream: Iterator[str]) -> Iterator[str]:
        """
        Filter the stream through the guardrail scanner using a sliding window.

        The last `max_len` characters are held back, because a forbidden term may be
        split across two chunks. Without a scanner the window is 0 and every chunk
        is released immediately.
        """
        scanner = self.guardrail.get_streaming_scanner()
        max_len = getattr(scanner, "_max_len", 0) if scanner else 0
        buffer = ""

        for chunk in filter(None, stream):
            matches = scanner.feed(chunk) if scanner else None
            buffer += chunk

            if matches:
                logger.info(f"Chunk redacted for fairness violation. Matches: {matches}")
                buffer = self.guardrail.redact_terms(buffer, matches)

            safe_len = len(buffer) - max_len
            if safe_len > 0:
                yield buffer[:safe_len]
                buffer = buffer[safe_len:]

        pending = scanner.flush() if scanner else None
        if pending:
            logger.info(f"Stream redacted for fairness violation at end of stream. Matches: {pending}")
            buffer = self.guardrail.redact_terms(buffer, pending)

        if buffer:
            yield buffer

    def _save_streamed_dialogue(self, npc_context: NPCContext, full_dialogue: str) -> None:
        """Compose the full dialogue and store it in history. Never raises."""
        try:
            composed_dialogue = self.dialogue_composer.compose_dialogue(npc_context, full_dialogue)
            self.dialogue_history.add_npc_dialogue_to_history(composed_dialogue)
            logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")
        except Exception:
            logger.exception("Failed to compose dialogue after stream")