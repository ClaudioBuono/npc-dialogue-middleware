from pydantic import ValidationError
import pytest
from core.contract_builder import ContractBuilder
from core.types.contexts import GameContext, NPCContext, Dialogue, Quest, Talkativeness
from core.types.dataclasses import Contract

@pytest.fixture
def builder():
    return ContractBuilder()

@pytest.fixture
def game_context():
    return GameContext(
        environment="Forest",
        epoch="Medieval",
        world_state="Peaceful",
        main_character_description="A brave knight"
    )

def test_build_dialogue_intent(builder, game_context):
    dialogue = Dialogue(type="Dialogue", has_options=True)
    npc_context = NPCContext(
        name="Eldrin",
        age=83,
        personality="Wise",
        context="Sitting by the fire",
        talkativeness=Talkativeness.HIGH,
        main_character_relation="Neutral",
        intent=dialogue
    )
    
    contract = builder.build(game_context, npc_context, dialogue_history=[])
    
    assert isinstance(contract, Contract)
    assert "Eldrin" in contract.user_prompt
    assert "A brave knight" in contract.system_prompt
    assert contract.output_schema["type"] == "object"
    assert "dialogue" in contract.output_schema["properties"]
    assert "player_options" in contract.output_schema["properties"]
    assert "dialogue_options" in contract.output_schema["properties"]["player_options"]["properties"]

def test_build_quest_intent(builder, game_context):
    quest = Quest(
        type="Quest",
        objective="Find the lost ring",
        has_choice=True,
        has_options=True,
        name="The Lost Ring",
        description="A golden ring lost in the woods."
    )
    npc_context = NPCContext(
        name="Elara",
        age=18,
        personality="Anxious",
        context="Looking under a tree",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Friendly",
        intent=quest
    )
    
    contract = builder.build(game_context, npc_context, dialogue_history=[])
    
    assert isinstance(contract, Contract)
    assert "Elara" in contract.user_prompt
    assert "Find the lost ring" in contract.user_prompt
    assert contract.output_schema["type"] == "object"
    
    player_options_schema = contract.output_schema["properties"]["player_options"]["properties"]
    assert "accept" in player_options_schema
    assert "refuse" in player_options_schema
    assert "dialogue_options" in player_options_schema

def test_build_invalid_intent(builder, game_context):
    with pytest.raises(ValidationError, match="Input should be a valid dictionary or object"):
        npc_context = NPCContext(
                name="Gorg",
                age=15, # TODO: Maybe enable "Unknown" age
                personality="Angry",
                context="Cave",
                talkativeness=Talkativeness.LOW,
                main_character_relation="Hostile",
                intent="Invalid Intent Type" # type: ignore
            )
        builder.build(game_context, npc_context, dialogue_history=[])

def test_build_with_history(builder, game_context):
    dialogue = Dialogue(type="Dialogue", has_options=False)
    npc_context = NPCContext(
        name="Eldrin",
        age=83,
        personality="Wise",
        context="Sitting by the fire",
        talkativeness=Talkativeness.HIGH,
        main_character_relation="Neutral",
        intent=dialogue
    )
    
    history = [{"player": "Hello", "npc": "Greetings"}]
    contract = builder.build(game_context, npc_context, dialogue_history=history)
    
    assert "Greetings" in contract.user_prompt
