from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class Quest(BaseModel):
    """Quest block.

    Contains the information needed to describe a quest that the NPC
    can assign to the player.
    """

    objective: str = Field(..., description="Objective the player must accomplish.")
    name: Optional[str] = Field(None, description="Name of the quest, if already defined.")
    description: Optional[str] = Field(None, description="Narrative description of the quest, if already defined.")
    location: Optional[str] = Field(None, description="Location where the quest takes place or must be completed, if already defined.")
    reward: Optional[str] = Field(None, description="Reward given upon completion of the quest, if any.")
    generate_accept_refuse: Optional[bool] = Field(
        False,
        description="Whether the dialogue options should include the possibility for the "
                    "player to accept or refuse the quest."
    )
    generate_more_opt: Optional[bool] = Field(
        False,
        description="Whether additional dialogue options beyond accept/refuse should be "
                    "generated for this quest (e.g. asking for more details, negotiating terms)."
    )


class Dialogue(BaseModel):
    """Dialogue block, covering both key and simple dialogue cases.

    Used for any NPC interaction that is not a quest. If `must_use` is
    set, the NPC must necessarily convey that specific piece
    of information during the conversation.
    """

    must_use: Optional[str] = Field(None, description="Piece of information the NPC must necessarily communicate or use in the dialogue.")
    more_info: Optional[str] = Field(
        None,
        description="Additional information the dialogue options should be based on when "
                    "generating them (e.g. context or details to inform the player's choices)."
    )


class Talkativeness(Enum):
    """How talkative the NPC is, used to calibrate the length and frequency of its lines."""

    VERYLOW = 1
    LOW = 2
    AVERAGE = 3
    HIGH = 4
    VERYHIGH = 5


class GameContext(BaseModel):
    """Overall context of the game.

    Defines the setting and background information shared
    by all characters in the game world.
    """

    # Required
    epoch: str = Field(..., description="Epoch in which the story takes place (e.g. medieval/futuristic/cyberpunk).")
    environment: str = Field(..., description="Description of the environment in which the story takes place.")

    # Optional
    plot: Optional[str] = Field(None, description="Relevant plot known to everyone in the game world.")
    main_character_description: Optional[str] = Field(
        None,
        description="Description of the main character. Should be purely physical; "
                    "can include personality traits only if the character is already famous in the game world."
    )


class NPCContext(BaseModel):
    """Full description of an NPC (Non-Player Character).

    Contains both biographical/personality information and the current
    narrative state of the NPC, including its dialogue intent.
    """

    # Required
    name: str = Field(..., description="Name of the character. May also include a surname or any other identifier suitable for the game world.")
    age: int = Field(..., description="Age of the character, expressed in whatever format is suitable for the game world.")
    personality: str = Field(..., description="Personality of the NPC.")
    context: str = Field(..., description="Current context of the NPC: location, time of day, weather, and other useful details.")
    intent: Quest | Dialogue = Field(
        ...,
        description="Dialogue intent of the NPC. Use a Quest when the NPC is assigning or "
                    "discussing a mission. Use a Dialogue for anything else, including plain "
                    "conversation with no particular narrative purpose."
    )
    talkativeness: Talkativeness = Field(..., description="How much the NPC tends to talk.")
    main_character_relation: str = Field(..., description="Relationship of the NPC with the main character.")

    # Optional
    recent_plot: Optional[str] = Field(None, description="Recent events in the game world relevant to the NPC.")
    visual_description: Optional[str] = Field(None, description="Physical description of the NPC.")
    backstory: Optional[str] = Field(None, description="Backstory of the NPC.")
    language: Optional[List[str]] = Field(
        None,
        description="Languages spoken by the NPC. Defaults to the configuration's default language if not specified." # TODO: Set default language in game context from CONFIG
    )