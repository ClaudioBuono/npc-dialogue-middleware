from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class QuestDialogue(BaseModel):
    """Quest dialogue block.

    Contains all the information needed to describe a quest
    that the NPC can assign to the player.
    """

    name: str = Field(..., description="Name of the quest.")
    objective: str = Field(..., description="Objective the player must accomplish.")
    description: str = Field(..., description="Narrative description of the quest.")
    location: str = Field(..., description="Location where the quest takes place or must be completed.")
    reward: Optional[str] = Field(None, description="Reward given upon completion of the quest, if any.")


class KeyDialogue(BaseModel):
    """Key dialogue block.

    Used when the NPC must necessarily convey or receive a piece of
    information that is crucial to the plot, without it being tied to a quest.
    """

    must_use: str = Field(..., description="Piece of information the NPC must necessarily communicate or use in the dialogue.")


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


class NPContext(BaseModel):
    """Full description of an NPC (Non-Player Character).

    Contains both biographical/personality information and the current
    narrative state of the NPC, including its dialogue intent, if any.
    """

    # Required
    name: str = Field(..., description="Name of the character. May also include a surname or any other identifier suitable for the game world.")
    age: int = Field(..., description="Age of the character, expressed in whatever format is suitable for the game world.")
    personality: str = Field(..., description="Personality of the NPC.")
    context: str = Field(..., description="Current context of the NPC: location, time of day, weather, and other useful details.")
    intent: Optional[QuestDialogue | KeyDialogue] = Field(
        None,
        description="Dialogue intent of the NPC: can be a Quest, a KeyDialogue, "
                    "or None in the case of a simple dialogue with no particular purpose."
    )
    talkativeness: Talkativeness = Field(..., description="How much the NPC tends to talk.")
    main_character_relation: str = Field(..., description="Relationship of the NPC with the main character.")

    # Optional
    recent_plot: Optional[str] = Field(None, description="Recent events in the game world relevant to the NPC.")
    visual_description: Optional[str] = Field(None, description="Physical description of the NPC.")
    backstory: Optional[str] = Field(None, description="Backstory of the NPC.")
    language: Optional[List[str]] = Field(
        None,
        description="Languages spoken by the NPC. Defaults to the configuration's default language if not specified."
    )