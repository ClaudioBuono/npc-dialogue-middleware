from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

class Dialogue(BaseModel):
    """
    Base dialogue block describing any standard NPC interaction or conversation.
    """
    type: Literal["Dialogue"] = Field(
        "Dialogue",
        description="Discriminator field that identifies this block as a plain Dialogue.",
        examples=["Dialogue"]
    )
    must_use_expression: Optional[str] = Field(
        None, 
        description="Core piece of information or key message the NPC must communicate during the dialogue.",
        examples=["I used to be an adventurer like you."]
    )
    more_info: Optional[str] = Field(
        None,
        description="Contextual details or background information to guide the generation and eventual player's dialogue choices.",
        examples=["The NPC is secretly hiding a stolen ring."]
    )
    has_options: Optional[bool] = Field(
        False,
        description="If True, additional dialogue options should be generated (e.g. asking for details).",
        examples=[True, False]
    )

class Quest(Dialogue):
    """
    Quest block.

    Extends `Dialogue` with the information needed to describe a quest that the NPC can assign to the player.
    """
    type: Literal["Quest"] = Field(
        "Quest",
        description="Discriminator field that identifies this block as a Quest.",
        examples=["Quest"]
    )
    objective: str = Field(..., description="The main task or goal the player must accomplish to complete the quest.", examples=["Find the lost sword of Elendil"])
    name: Optional[str] = Field(None, description="The title of the quest, if predetermined.", examples=["The Lost Heirloom"])
    description: Optional[str] = Field(None, description="Narrative introduction and lore summary of the quest.", examples=["An ancient sword was lost in the ruins of the old castle centuries ago."])
    location: Optional[str] = Field(None, description="The specific zone, world location, or target area where the quest takes place.", examples=["Ruins of old Castle Dour"])
    reward: Optional[str] = Field(None, description="The loot, experience, currency, or favor awarded to the player upon completion.", examples=["100 Gold Coins", "Sword of the Ancients"])
    has_choice: Optional[bool] = Field(
        False,
        description="If True, the player can actively choose whether to accept or decline the quest.",
        examples=[True, False]
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
        description="The historical or thematic era of the story (e.g., Medieval, Hard Sci-Fi, Cyberpunk, High Fantasy).",
        examples=["Medieval Fantasy"]
    )
    environment: str = Field(
        ..., 
        description="General description of the world's nature, atmosphere, climate, or architectural theme.",
        examples=["Dark and gritty world with dangerous wilderness and imposing castles."]
    )
    world_state: str = Field(
        ..., 
        description="The current global situation or major ongoing events in the world that everyone is aware of.",
        examples=["A recent war has left the kingdom in ruins and poverty is rampant."]
    )
    main_character_description: Optional[str] = Field(
        None,
        description="Physical appearance of the protagonist. Include personality traits only if they are widely famous in the game world.",
        examples=["A tall warrior with a scar across his face, clad in heavy armor."]
    )


class NPCContext(BaseModel):
    """
    Comprehensive profile of an NPC, combining biographical lore, personality, and current narrative state.
    """
    intent: Union[Quest, Dialogue] = Field(
        ...,
        discriminator="type",
        description="The current driving purpose of the conversation."
    )
    name: str = Field(..., description="The full name, title, or alias of the NPC suitable for the game setting.", examples=["Garrick the Blacksmith"])
    age: int = Field(..., description="The age of the character.", examples=[45])
    personality: str = Field(..., description="Psychological profile and behavioral traits of the NPC.", examples=["Gruff but kind-hearted, easily annoyed by time-wasters."])
    context: str = Field(..., description="The immediate state of the NPC: their current location, time of day, weather, and current activity.", examples=["Working at the forge, late evening, raining outside."])
    talkativeness: Talkativeness = Field(..., description="How much the NPC tends to talk.", examples=["Balanced"]) 
    main_character_relation: str = Field(..., description="The NPC's current attitude, stance, and relationship status toward the protagonist.", examples=["Friendly and respectful"])

    recent_plot: Optional[str] = Field(
        None, 
        description="Recent world events or narrative shifts that directly affect or interest this specific NPC.",
        examples=["Iron shipments have been delayed due to bandit attacks on the roads."]
    )
    visual_description: Optional[str] = Field(
        None, 
        description="Detailed physical appearance, clothing, equipment, and visible expressions of the NPC.",
        examples=["Covered in soot, muscular, wearing a thick leather apron and holding a heavy hammer."]
    )
    backstory: Optional[str] = Field(
        None, 
        description="The history, past experiences, and core motivations of the NPC.",
        examples=["Learned the trade from his father, deeply values hard work and honesty."]
    )
    language: Optional[List[str]] = Field(
        None,
        description="List of languages the NPC can speak. If null, falls back to the system's default language configuration.", # TODO: Set default language in game context from CONFIG
        examples=[["Common", "Dwarven"]]
    )