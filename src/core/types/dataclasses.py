from dataclasses import dataclass, field
from typing import Any, List, Union, Optional
from pydantic import BaseModel, Field

from core.types.contexts import Dialogue, NPCContext, Quest


@dataclass
class Contract:
    """
    Structured payload ready to be sent to the LLM.
    """
    system_prompt: str
    user_prompt: str
    output_schema: dict[str, Any] = field(default_factory=dict)


class DialogueOptionsSchema(BaseModel):
    """
    Player options for a standard dialogue.
    """
    dialogue_options: List[str] = Field(
        ...,
        description="Must include the options the player can choose from in response (e.g. asking for more details).",
        examples=[["Tell me more about the ring", "I have to go"]]
    )


class QuestChoiceSchema(BaseModel):
    """
    Player options for a quest choice.
    """
    accept: str = Field(
        ...,
        description="Player dialogue option to accept the quest.",
        examples=["I will find the sword for you."]
    )
    refuse: str = Field(
        ...,
        description="Player dialogue option to refuse the quest.",
        examples=["I don't have time for this."]
    )
    dialogue_options: Optional[List[str]] = Field(
        None,
        description="Additional dialogue options to ask for details before accepting or refusing.",
        examples=[["Where was it lost?", "Is it guarded?"]]
    )


class ComposedDialogue(BaseModel):
    """
    Structured dialogue to be sent as an output.
    """
    intent: Union[Quest, Dialogue] = Field(
        ...,
        description="The driving purpose of the generated dialogue.",
        discriminator="type"
    )
    dialogue: str = Field(
        ...,
        description="The dialogue line(s) spoken by the NPC.",
        examples=["Hello there, traveler. I need your help."]
    )
    player_options: Optional[Union[QuestChoiceSchema, DialogueOptionsSchema]] = Field(
        None,
        description="Available choices for the player to respond."
    )

class DialogueStreamRequest(BaseModel):
    """
    Payload for initiating or continuing a streaming dialogue request with an NPC.
    """
    npc_context: NPCContext = Field(
        ...,
        description="The contextual profile and narrative state of the target NPC."
    )
    last_player_choice: Optional[str] = Field(
        None,
        description="The previous dialogue or quest option selected by the player, if any.",
        examples=["Accept the quest"]
    )