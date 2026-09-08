import re
from typing import Any, Optional, Iterator
import logging
from api.schemas import ComposedDialogue
from core.state_manager import StateManager
from core.config.settings import Settings
from core.contract_builder import ContractBuilder
from core.dialogue_generator import DialogueGenerator
from core.guardrail import Guardrail
from core.tools.history import DialogueHistory
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.output_composer import DialogueOutputComposer
from core.routing.router import LLMRouter
from core.helpers.logger import to_json_format
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness
from core.types.enums import MiddlewareState
from core.tools import pre_processing

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

    """Holds validated game context for middleware operations."""
    game_context: GameContext | None = None

    _instance: "Orchestrator | None" = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Orchestrator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.contract_builder: ContractBuilder = ContractBuilder()
        self.llm_router: LLMRouter = LLMRouter()
        self.dialogue_generator: DialogueGenerator = DialogueGenerator()
        self.dialogue_history: DialogueHistory = DialogueHistory()
        self.dialogue_composer: DialogueOutputComposer = DialogueOutputComposer()
        self.guardrail: Guardrail = Guardrail()

        self._iterations = 0
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "Orchestrator":
        """
        Return the already-created singleton instance.
        If it has never been created (no one has called Orchestrator()
        yet, with or without arguments), create it with default values.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Test utility: reset the singleton instance."""
        cls._instance = None

    # Orchestrator Methods ----------------------------------------------------------------------------

    def set_game_context(self, game_context: GameContext) -> None:
        """Set the game context by validating environment, epoch, and lore."""

        StateManager().transition_to(MiddlewareState.SETTING_CONTEXT)

        self.game_context = game_context

        StateManager().transition_to(MiddlewareState.IDLE)
        

    def generate_dialogue(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> ComposedDialogue | None:
        
        """Generate NPC dialogue using the NPC and game context."""

        logger.info(f"Generating dialogue for NPC '{npc_context.name}'")
            
        StateManager().transition_to(MiddlewareState.GENERATING)

        self._iterations += 1

        if last_player_choice:
            self.dialogue_history.add_player_dialogue_to_history(last_player_choice)
            logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")


        contract = self.contract_builder.build(self.game_context, npc_context, self.dialogue_history.get_dialogue_history())

        client: OpenAICompatibleClient = self.llm_router.select_model(game_context = self.game_context, npc_context = npc_context)
        logger.debug(f"Selected LLM client: {type(client).__name__}")

        self.dialogue_generator.set_client(client)
        raw_dialogue: str = self.dialogue_generator.generate(contract)

        composed_dialogue = self.dialogue_composer.compose_dialogue(npc_context, raw_dialogue)

        if Settings().profanity_filter:
            valid_output: bool = self.guardrail.validate_composed_output(composed_dialogue)

            if not valid_output:
                logger.info(f"Dialogue refused for fairness violation.")

                StateManager().transition_to(MiddlewareState.IDLE)
                
                return None

        self.dialogue_history.add_npc_dialogue_to_history(composed_dialogue)

        StateManager().transition_to(MiddlewareState.IDLE)

        logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")

        return composed_dialogue

    @staticmethod
    def _redact_terms(text: str, terms: list[str]) -> str:
        """Changes banned words with [REDACTED]."""
        #TODO: add custom changed token
        for term in terms:
            pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
            text = pattern.sub("[REDACTED]", text)
        return text

    def generate_dialogue_stream(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> Iterator[str]:
        """Generate NPC dialogue using the NPC and game context via streaming."""

        StateManager().transition_to(MiddlewareState.GENERATING)
        logger.info(f"Generating dialogue stream for NPC '{npc_context.name}'")

        self._iterations += 1

        if last_player_choice:
            self.dialogue_history.add_player_dialogue_to_history(last_player_choice)
            logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")

        contract = self.contract_builder.build(self.game_context, npc_context, self.dialogue_history.get_dialogue_history())

        client: OpenAICompatibleClient = self.llm_router.select_model(game_context=self.game_context, npc_context=npc_context)
        logger.debug(f"Selected LLM client: {type(client).__name__}")

        self.dialogue_generator.set_client(client)
        stream = self.dialogue_generator.generate_stream(contract)

        scanner = None
        max_len = 0
        if Settings().profanity_filter:
            scanner = self.guardrail.get_streaming_scanner()
            max_len = getattr(scanner, "_max_len", 0)

        full_dialogue = ""
        refused = False
        output_buffer = ""

        for chunk in stream:
            if not chunk:
                continue

            if scanner:
                matches = scanner.feed(chunk)
                output_buffer += chunk

                if matches:
                    logger.info(f"Chunk redacted for fairness violation. Matches: {matches}")
                    output_buffer = self._redact_terms(output_buffer, matches)
                    refused = True

                safe_len = len(output_buffer) - max_len
                if safe_len > 0:
                    to_release = output_buffer[:safe_len]
                    full_dialogue += to_release
                    yield to_release
                    output_buffer = output_buffer[safe_len:]
            else:
                full_dialogue += chunk
                yield chunk

        
        if scanner:
            pending_matches = scanner.flush()
            if pending_matches:

                logger.info(f"Dialogue stream redacted for fairness violation at end of stream. Matches: {pending_matches}")
                output_buffer = self._redact_terms(output_buffer, pending_matches)
                refused = True

            if output_buffer:
                full_dialogue += output_buffer
                yield output_buffer

        if not refused:
            try:
                composed_dialogue = self.dialogue_composer.compose_dialogue(npc_context, full_dialogue)
                self.dialogue_history.add_npc_dialogue_to_history(composed_dialogue)
                StateManager().transition_to(MiddlewareState.IDLE)
                logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")
            except Exception as e:
                logger.error(f"Failed to compose dialogue after stream: {e}")
        else:
            try:
                composed_dialogue = self.dialogue_composer.compose_dialogue(npc_context, full_dialogue)
                self.dialogue_history.add_npc_dialogue_to_history(composed_dialogue)
            except Exception as e:
                StateManager().transition_to(MiddlewareState.IDLE)
                logger.error(f"Failed to compose dialogue after refused stream: {e}")