from typing import Any, Optional
import logging
from core.contract_builder import ContractBuilder
from core.dialogue_generator import DialogueGenerator
from core.history import DialogueHistory
from core.llm.openai_client import OpenAICompatibleClient
from core.output_composer import DialogueOutputComposer
from core.routing.models import load_config_from_file
from core.routing.registry import ModelRegistry
from core.routing.router import LLMRouter
from core.logger import to_json_format
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness
from core.types.dataclasses import ComposedDialogue
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

    def set_game_context(self, environment: str, epoch: str, world_state: str) -> None:
        """Set the game context by validating environment, epoch, and lore."""
        game_context = GameContext(
            epoch=epoch,
            environment=environment,
            world_state=world_state,
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
    ) -> ComposedDialogue:
        
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

            configs = load_config_from_file("src/modelconfigs_test.json")

            ModelRegistry().set_models(configs,profiler=False)

            client: OpenAICompatibleClient = self.llm_router.select_model(game_context = self.game_context, npc_context = validated_npc_context)
            logger.debug(f"Selected LLM client: {type(client).__name__}")

            self.dialogue_generator.set_client(client)
            raw_dialogue: str = self.dialogue_generator.generate(contract)

            composed_dialogue = self.dialogue_composer.compose_dialogue(validated_npc_context, raw_dialogue)
            self.dialogue_history.add_npc_dialogue_to_history(composed_dialogue)

            logger.debug(f"Dialogue history updated:\n{to_json_format(self.dialogue_history.get_dialogue_history())}")


            return composed_dialogue