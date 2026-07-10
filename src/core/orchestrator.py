from typing import Any
from core.contract_builder import ContractBuilder
from core.dialogue_generator import DialogueGenerator
from core.llm.openai_client import OpenAICompatibleClient
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness
from tools import pre_processing
import dataclasses, json


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
        self.contract_builder = ContractBuilder()
        self.dialogue_generator = DialogueGenerator(
            client=OpenAICompatibleClient(
                endpoint="http://localhost:11434/v1",
                api_key=None,
                model_identifier="llama3.2",
            )
        )
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
        intent: Quest | Dialogue
    ) -> str:
        
        """Generate NPC dialogue using the NPC and game context."""


        if self.game_context:
            
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
            contract = self.contract_builder.build(self.game_context, validated_npc_context)
            
            print(json.dumps(dataclasses.asdict(contract), indent=2))


            dialogue: str = self.dialogue_generator.generate(contract)

            return dialogue