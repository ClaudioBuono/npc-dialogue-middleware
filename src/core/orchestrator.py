from typing import Any, Optional, Iterator
import logging
from api.schemas import ComposedDialogue
from core.config.settings import Settings
from core.contract_builder import ContractBuilder
from core.dialogue_generator import DialogueGenerator
from core.guardrail import Guardrail
from core.history import DialogueHistory
from core.llm.openai_client import OpenAICompatibleClient
from core.output_composer import DialogueOutputComposer
from core.routing.models import load_config_from_file
from core.routing.registry import ModelRegistry
from core.routing.router import LLMRouter
from core.logger import to_json_format
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness
from tools import pre_processing

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

        # Load models 
        configs = load_config_from_file(Settings()._config_dir / "modelconfigs.json")
        ModelRegistry().set_models(configs, profiler=False)

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

    def set_game_context(self, environment: str, epoch: str, world_state: str, main_character_description: str | None) -> None:
        """Set the game context by validating environment, epoch, and lore."""
        game_context = GameContext(
            epoch=epoch,
            environment=environment,
            world_state=world_state,
            main_character_description=main_character_description
        )
        self.game_context = pre_processing.validate_game_context(game_context)

    def generate_dialogue(
        self,
        name: str,
        age: int,
        personality: str,
        context: str,
        talkativeness: Talkativeness,
        main_character_relation: str,
        intent: Quest | Dialogue,
        last_player_choice: Optional[str]
    ) -> ComposedDialogue | None:
        
        """Generate NPC dialogue using the NPC and game context."""

        logger.info(f"Generating dialogue for NPC '{name}'")

        if self.game_context:

            self._iterations += 1

            if last_player_choice:
                self.dialogue_history.add_player_dialogue_to_history(last_player_choice)
                logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")

            
            npc_context = NPCContext(
                name=name,
                age=age,
                personality=personality,
                context=context,
                talkativeness=talkativeness,
                main_character_relation=main_character_relation,
                intent=intent,
            )
            validated_npc_context = pre_processing.validate_npc_context(npc_context)
            contract = self.contract_builder.build(self.game_context, validated_npc_context, self.dialogue_history.get_dialogue_history())

            client: OpenAICompatibleClient = self.llm_router.select_model(game_context = self.game_context, npc_context = validated_npc_context)
            logger.debug(f"Selected LLM client: {type(client).__name__}")

            self.dialogue_generator.set_client(client)
            raw_dialogue: str = self.dialogue_generator.generate(contract)

            composed_dialogue = self.dialogue_composer.compose_dialogue(validated_npc_context, raw_dialogue)

            if Settings().profanity_filter:
                valid_output: bool = self.guardrail.validate(composed_dialogue)

                if not valid_output:
                    logger.info(f"Dialogue refused for fairness violation.")
                    return None

            self.dialogue_history.add_npc_dialogue_to_history(composed_dialogue)

            logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")


            return composed_dialogue

        else:
            # TODO: Give a separate exception
            logger.info("Game context was not set.")
            return None

    def generate_dialogue_stream(
        self,
        name: str,
        age: int,
        personality: str,
        context: str,
        talkativeness: Talkativeness,
        main_character_relation: str,
        intent: Quest | Dialogue,
        last_player_choice: Optional[str]
    ) -> Iterator[str]:
        
        """Generate NPC dialogue using the NPC and game context via streaming."""

        logger.info(f"Generating dialogue stream for NPC '{name}'")

        if self.game_context:

            self._iterations += 1

            if last_player_choice:
                self.dialogue_history.add_player_dialogue_to_history(last_player_choice)
                logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")
            
            npc_context = NPCContext(
                name=name,
                age=age,
                personality=personality,
                context=context,
                talkativeness=talkativeness,
                main_character_relation=main_character_relation,
                intent=intent,
            )
            validated_npc_context = pre_processing.validate_npc_context(npc_context)
            contract = self.contract_builder.build(self.game_context, validated_npc_context, self.dialogue_history.get_dialogue_history())

            client: OpenAICompatibleClient = self.llm_router.select_model(game_context = self.game_context, npc_context = validated_npc_context)
            logger.debug(f"Selected LLM client: {type(client).__name__}")

            self.dialogue_generator.set_client(client)
            stream = self.dialogue_generator.generate_stream(contract)
            
            scanner = None
            if Settings().profanity_filter:
                scanner = self.guardrail.get_streaming_scanner()

            full_dialogue = ""
            refused = False

            for chunk in stream:
                if scanner:
                    matches = scanner.feed(chunk)
                    if matches:
                        logger.info(f"Dialogue stream refused for fairness violation during generation. Matches: {matches}")
                        refused = True
                        break
                
                full_dialogue += chunk
                yield chunk

            if scanner and not refused:
                matches = scanner.flush()
                if matches:
                    logger.info(f"Dialogue stream refused for fairness violation at end of stream. Matches: {matches}")
                    refused = True

            if not refused:
                try:
                    composed_dialogue = self.dialogue_composer.compose_dialogue(validated_npc_context, full_dialogue)
                    self.dialogue_history.add_npc_dialogue_to_history(composed_dialogue)
                    logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")
                except Exception as e:
                    logger.error(f"Failed to compose dialogue after stream: {e}")

        else:
            logger.info("Game context was not set.")
            return