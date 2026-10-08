import time
from typing import Any, Optional, Iterator
import logging
from api.schemas import ComposedDialogue
from core.configuration.settings import Settings
from core.pipeline.refiner import Refiner
from core.infrastructure.state_manager import StateManager
from core.generation.contract_builder import ContractBuilder
from core.generation.dialogue_generator import DialogueGenerator
from core.pipeline.guardrail import Guardrail
from core.generation.history import DialogueHistory
from core.llm.openai_client import OpenAICompatibleClient
from core.generation.output_composer import DialogueOutputComposer
from core.routing.router import LLMRouter
from core.helpers.formatters import format_composed_dialogue, format_stream, to_json_format
from core.types.contexts import GameContext, NPCContext
from core.types.enums import MiddlewareState

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

        logger.info("Game context set")

        StateManager().transition_to(MiddlewareState.IDLE)

        self.game_context = game_context
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Game context: %s", to_json_format(game_context))


    def generate_dialogue(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> ComposedDialogue | None:
        """Generate NPC dialogue using the NPC and game context.

        Args:
            npc_context: Information about the NPC that is speaking.
            last_player_choice: The option the player picked in the previous turn,
                or None if this is the first line of the conversation.

        Returns:
            The final structured dialogue (text, intent and player options).
        """
        StateManager().transition_to(MiddlewareState.GENERATING)
        start = time.perf_counter()
        logger.info(
            "Generating dialogue: npc=%r, intent=%s, history_turns=%d, player_choice=%s",
            npc_context.name, type(npc_context.intent).__name__,
            len(self.dialogue_history.get_dialogue_history()), bool(last_player_choice),
        )

        try:
            composed_dialogue = self._generate_dialogue(npc_context, last_player_choice)
        except Exception as e:
            logger.error(
                "Dialogue generation failed after %.2fs (%s): %s",
                time.perf_counter() - start, type(e).__name__, e,
            )
            raise
        finally:
            StateManager().transition_to(MiddlewareState.IDLE)

        logger.info("Dialogue generated for NPC %r in %.2fs", npc_context.name, time.perf_counter() - start)
        logger.info("Generated dialogue:\n%s", format_composed_dialogue(composed_dialogue, include_intent=False))
        return composed_dialogue

    
    def generate_dialogue_stream(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> Iterator[str]:
        """Generate NPC dialogue using the NPC and game context via streaming."""

        StateManager().transition_to(MiddlewareState.GENERATING)
        start = time.perf_counter()
        logger.info(
            "Generating dialogue stream: npc=%r, intent=%s, history_turns=%d, player_choice=%s",
            npc_context.name, type(npc_context.intent).__name__,
            len(self.dialogue_history.get_dialogue_history()), bool(last_player_choice),
        )

        try:
            raw_stream = self._start_dialogue_stream(npc_context, last_player_choice)

            released: list[str] = []
            for text in self._guarded_stream(raw_stream):
                released.append(text)
                yield text

            full_text = "".join(released)
            self._save_streamed_dialogue(npc_context, full_text)
        finally:
            StateManager().transition_to(MiddlewareState.IDLE)

        logger.info("Dialogue stream completed for NPC %r in %.2fs", npc_context.name, time.perf_counter() - start)
        logger.info("Generated stream:\n%s", format_stream(full_text))

    # Pipeline Methods ----------------------------------------------------------------------------
    def _generate_dialogue(self, npc_context: NPCContext, last_player_choice: Optional[str]) -> ComposedDialogue | None:
        """
        Pipeline: build the prompt contract, generate the raw dialogue with the
        selected LLM, compose it into a structured ComposedDialogue, then apply
        either the refiner or the profanity censor, and finally store the result
        in the dialogue history.
        """
        # Record the player's reply first, so the contract includes it as context.
        if last_player_choice:
            self.dialogue_history.add_player_dialogue_to_history(last_player_choice)
            
        # Build the prompt contract from game state, NPC info and conversation so far.
        contract = self.contract_builder.build_dialogue_contract(
            self.game_context, npc_context, self.dialogue_history.get_dialogue_history()
        )

        # The router picks the LLM best suited to this NPC and game context.
        client: OpenAICompatibleClient = self.llm_router.select_model(
            game_context=self.game_context, npc_context=npc_context
        )
        logger.info("Selected LLM: %s", client.model_name)

        # Generate the raw LLM output, then parse it into a structured dialogue.
        self.dialogue_generator.set_client(client)
        raw_dialogue: str = self.dialogue_generator.generate(contract)
        composed_dialogue = self.dialogue_composer.compose_dialogue(npc_context, raw_dialogue)

        # Post-processing: the refiner iteratively corrects the dialogue using the
        # judger questions (which already cover profanity when profanity_filter is on).
        # Without the refiner, banned words are censored directly with censor_word.
        # If neither of these are enabled, then the dialogue is returned as-is.
        if Settings().refine_dialogue:
            self.refiner.set_client(client)
            composed_dialogue = self.refiner.refine_dialogue(composed_dialogue, npc_context, self.game_context)

        elif Settings().profanity_filter and not Settings().refine_dialogue:
            composed_dialogue = self.guardrail.censor_composed_dialogue(composed_dialogue)

        # Store the final (post-processed) dialogue so later turns see what the player saw.
        self.dialogue_history.add_npc_dialogue_to_history(composed_dialogue)
        self._log_history()

        return composed_dialogue

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

    def _log_history(self) -> None:
        """Logs the current dialogue history at DEBUG level."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Dialogue history updated:\n%s",
                to_json_format(self.dialogue_history.get_dialogue_history()),
            )