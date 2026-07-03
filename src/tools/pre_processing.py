import logging
from typing import Any

logger = logging.getLogger(__name__)


class PreProcessingError(Exception):
    """Exception raised when pre-processing fails on unrecoverable data."""
    pass


class PreProcessor:
    """
    Cleans, validates and normalizes the raw dictionary coming
    from the Game Engine adapter before it reaches the ContractBuilder.

    Responsibilities:
        - Check presence of required fields.
        - Sanitize strings (strip, remove non-UTF8 characters).
        - Apply default values for missing optional fields.
        - Convert types where needed (e.g. str → int).

    Args:
        max_dialogue_hint_length: Maximum allowed length for dialogue_hint.
        max_npc_name_length:      Maximum allowed length for npc_name.
    """

    def __init__(
        self,
        max_dialogue_hint_length: int = 300,
        max_npc_name_length: int = 80,
    ) -> None:
        self.max_dialogue_hint_length = max_dialogue_hint_length
        self.max_npc_name_length = max_npc_name_length

    def validate_NPC_context(self, name: str, intent: str, description: str) -> dict[str, Any]:
        """
        Validate and normalize NPC context.

        Args:
            name:        NPC name (required).
            intent:      NPC intent (required).
            description: NPC description (optional).

        Returns:
            Normalized dictionary with validated fields.

        Raises:
            PreProcessingError: If data is invalid or required fields are missing.
        """
        if not name or not isinstance(name, str):
            raise PreProcessingError("The 'name' field is required and must be a string.")
        if len(name) > self.max_npc_name_length:
            raise PreProcessingError(f"The 'name' field exceeds the maximum length of {self.max_npc_name_length} characters.")

        if not intent or not isinstance(intent, str):
            raise PreProcessingError("The 'intent' field is required and must be a string.")

        if description is not None and not isinstance(description, str):
            raise PreProcessingError("The 'description' field, if present, must be a string.")

        return {
            "npc_name": name.strip(),
            "npc_intent": intent.strip(),
            "npc_description": description.strip() if description else "",
        }
    
    def validate_game_context(self, environment: str, epoch: str, lore: str) -> dict[str, Any]:
        """
        Validate and normalize game context.

        Args:
            environment: Game environment (required).
            epoch:       Historical period or epoch (required).
            lore:        Lore or narrative background (optional).

        Returns:
            Normalized dictionary with validated fields.

        Raises:
            PreProcessingError: If data is invalid or required fields are missing.
        """
        if not environment or not isinstance(environment, str):
            raise PreProcessingError("The 'environment' field is required and must be a string.")
        if not epoch or not isinstance(epoch, str):
            raise PreProcessingError("The 'epoch' field is required and must be a string.")
        if lore is not None and not isinstance(lore, str):
            raise PreProcessingError("The 'lore' field, if present, must be a string.")

        return {
            "game_environment": environment.strip(),
            "game_epoch": epoch.strip(),
            "game_lore": lore.strip() if lore else "",
        }


