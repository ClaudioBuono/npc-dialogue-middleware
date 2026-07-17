from typing import Dict, List

from core.types.dataclasses import ComposedDialogue


class DialogueHistory:
    """
    Tracks the dialogue exchanged between an NPC and the Player during
    a conversation, storing each turn as an entry..
    """

    history: List[Dict[str, str]] = []

    def add_npc_dialogue_to_history(self, dialogue: ComposedDialogue) -> None:
        """
        Append an NPC dialogue turn to the history.
        """

        dialogue_item = {
            "NPC": dialogue.dialogue
        }

        self.history.append(dialogue_item)

    def add_player_dialogue_to_history(self, player_choice: str) -> None:
        """
        Append a player dialogue turn to the history.
        """

        dialogue_item = {
            "Player": player_choice
        }

        self.history.append(dialogue_item)

    def get_dialogue_history(self) -> List[Dict[str, str]]:
        """
        Return the full dialogue history recorded so far.
        """
        return self.history

    def clear_dialogue_history(self) -> None:
        """Remove all entries from the dialogue history."""
        self.history.clear()