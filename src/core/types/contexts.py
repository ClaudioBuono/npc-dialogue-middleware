from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class Dialogue(BaseModel):
    """
    Base dialogue block describing any standard NPC interaction or conversation.
    """

    must_use_expression: Optional[str] = Field(
        None, 
        description="Core piece of information or key message the NPC must communicate during the dialogue."
    )
    more_info: Optional[str] = Field(
        None,
        description="Contextual details or background information to guide the generation and eventual player's dialogue choices."
    )
    has_options: Optional[bool] = Field(
        False,
        description="If True, additional dialogue options should be generated (e.g. asking for details)."
    )

class Quest(Dialogue):
    """
    Quest block.

    Extends `Dialogue` with the information needed to describe a quest that the NPC can assign to the player.
    """

    objective: str = Field(..., description="The main task or goal the player must accomplish to complete the quest.")
    name: Optional[str] = Field(None, description="The title of the quest, if predetermined.")
    description: Optional[str] = Field(None, description="Narrative introduction and lore summary of the quest.")
    location: Optional[str] = Field(None, description="The specific zone, world location, or target area where the quest takes place.")
    reward: Optional[str] = Field(None, description="The loot, experience, currency, or favor awarded to the player upon completion.")
    has_choice: Optional[bool] = Field(
        False,
        description="If True, the player can actively choose whether to accept or decline the quest."
    )


class Talkativeness(Enum):
    """
    The verbosity level of the NPC, controlling the length and frequency of their dialogue lines.
    """

    VERY_LOW = "Very terse"
    LOW = "Reserved"
    AVERAGE = "Balanced"
    HIGH = "Talkative"
    VERY_HIGH = "Very talkative"


class GameContext(BaseModel):
    """
    Global game context defining the shared world setting, lore, and background.
    """

    epoch: str = Field(
        ..., 
        description="The historical or thematic era of the story (e.g., Medieval, Hard Sci-Fi, Cyberpunk, High Fantasy)."
    )
    environment: str = Field(
        ..., 
        description="General description of the world's nature, atmosphere, climate, or architectural theme."
    )
    world_state: str = Field(
        ..., 
        description="The current global situation or major ongoing events in the world that everyone is aware of."
    )
    main_character_description: Optional[str] = Field(
        None,
        description="Physical appearance of the protagonist. Include personality traits only if they are widely famous in the game world."
    )


class NPCContext(BaseModel):
    """
    Comprehensive profile of an NPC, combining biographical lore, personality, and current narrative state.
    """

    name: str = Field(..., description="The full name, title, or alias of the NPC suitable for the game setting.")
    age: int = Field(..., description="The age of the character.")
    personality: str = Field(..., description="Psychological profile and behavioral traits of the NPC.")
    context: str = Field(..., description="The immediate state of the NPC: their current location, time of day, weather, and current activity.")
    intent: Quest | Dialogue = Field(
        ...,
        description="The current driving purpose of the conversation."
    )
    talkativeness: Talkativeness = Field(..., description="How much the NPC tends to talk.") 
    main_character_relation: str = Field(..., description="The NPC's current attitude, stance, and relationship status toward the protagonist.")

    recent_plot: Optional[str] = Field(
        None, 
        description="Recent world events or narrative shifts that directly affect or interest this specific NPC."
    )
    visual_description: Optional[str] = Field(
        None, 
        description="Detailed physical appearance, clothing, equipment, and visible expressions of the NPC."
    )
    backstory: Optional[str] = Field(
        None, 
        description="The history, past experiences, and core motivations of the NPC."
    )
    language: Optional[List[str]] = Field(
        None,
        description="List of languages the NPC can speak. If null, falls back to the system's default language configuration." # TODO: Set default language in game context from CONFIG
    )