import logging
from typing import Any
from core.contract_builder import ContractBuilder
from tools.pre_processing import PreProcessor
from tools.llm_handler import LLMHandler

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
    game_context: dict[str, Any] | None = None

    _instance: "Orchestrator | None" = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Orchestrator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, pre_processor: PreProcessor | None = None, llm_handler: LLMHandler | None = None) -> None:
        if self._initialized:
            return
        self.pre_processor = pre_processor or PreProcessor()
        self.contract_builder = ContractBuilder()
        self.llm_handler = llm_handler or LLMHandler()
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

    def set_game_context(self, environment: str, epoch: str, lore: str) -> None:
        """Set the game context by validating environment, epoch, and lore."""
        self.game_context = self.pre_processor.validate_game_context(environment, epoch, lore)

    def generate_dialogue(self, name: str, intent: str, description: str) -> dict[str, Any]:
        """Generate NPC dialogue using the NPC and game context."""
        npc_context: dict[str, Any] = self.pre_processor.validate_NPC_context(name, intent, description)
        contract = self.contract_builder.build(self.game_context, npc_context)
        llm_handler_result: dict[str, Any] = self.llm_handler.call(contract)
        return llm_handler_result