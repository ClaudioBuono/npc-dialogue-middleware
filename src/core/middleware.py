import logging
from typing import Any

from tools.pre_processing import PreProcessor
from tools.llm_handler import LLMHandler


logger = logging.getLogger(__name__)


class MiddlewareOrchestrator:
    """
    Coordinates all core components of the middleware, providing a single entry point for generating NPC dialogue and player options.

    Args:
        pre_processor:    Instance of PreProcessor.
        llm_handler:      Instance of LLMHandler.
    """

    """Holds validated game context for middleware operations."""
    game_context: dict[str, Any] | None = None

    def __init__(self, pre_processor: PreProcessor | None = None, llm_handler: LLMHandler | None = None) -> None:

        self.pre_processor = pre_processor or PreProcessor()
        self.llm_handler = llm_handler or LLMHandler()


    def set_game_context(self, environment: str, epoch: str, lore: str) -> None:
        """Set the game context by validating environment, epoch, and lore."""

        self.game_context = self.pre_processor.validate_game_context(environment, epoch, lore)
    
    def generate_dialogue(self, name: str, intent: str, description: str) -> dict[str, Any]:
        """Generate NPC dialogue using the NPC and game context."""

        npc_context: dict[str, Any] = self.pre_processor.validate_NPC_context(name, intent, description)
        llm_handler_result: dict[str, Any] = self.llm_handler.call(npc_context, self.game_context)
        return llm_handler_result
        
        

