import pytest
from pydantic import ValidationError
from api.schemas import (
    DialogueOptionsSchema,
    QuestChoiceSchema,
    ComposedDialogue,
    DialogueStreamRequest,
    MiddlewareStatusResponse,
    LanguageRequest,
    ToggleRequest,
    NumberOfOptionsRequest,
    SettingsUpdatedResponse
)
from core.types.contexts import Dialogue, Quest, NPCContext, Talkativeness
from core.types.enums import Language

def test_dialogue_options_schema_valid():
    schema = DialogueOptionsSchema(dialogue_options=["Option 1", "Option 2"])
    assert schema.dialogue_options == ["Option 1", "Option 2"]

def test_dialogue_options_schema_invalid():
    with pytest.raises(ValidationError):
        DialogueOptionsSchema(dialogue_options="Not a list")

def test_quest_choice_schema_valid():
    schema = QuestChoiceSchema(accept="Yes", refuse="No", dialogue_options=["More info?"])
    assert schema.accept == "Yes"
    assert schema.refuse == "No"
    assert schema.dialogue_options == ["More info?"]

    # Without optional dialogue_options
    schema2 = QuestChoiceSchema(accept="Yes", refuse="No")
    assert schema2.dialogue_options is None

def test_quest_choice_schema_invalid():
    with pytest.raises(ValidationError):
        QuestChoiceSchema(accept="Yes")  # Missing refuse

def test_composed_dialogue_valid():
    dialogue_intent = Dialogue()
    schema = ComposedDialogue(
        intent=dialogue_intent,
        dialogue="Hello traveler!",
        player_options=DialogueOptionsSchema(dialogue_options=["Hi", "Bye"])
    )
    assert schema.dialogue == "Hello traveler!"
    assert schema.intent.type == "Dialogue"

def test_composed_dialogue_invalid_intent():
    with pytest.raises(ValidationError):
        ComposedDialogue(
            intent={"invalid": "intent"},
            dialogue="Hello traveler!"
        )

def test_dialogue_stream_request_valid():
    npc_context = NPCContext(
        name="Bob",
        age=45,
        personality="Calm and welcoming farmer",
        context="Working in the fields",
        talkativeness=Talkativeness.VERY_HIGH,
        main_character_relation="Stranger",
        intent=Dialogue()
    )
    schema = DialogueStreamRequest(npc_context=npc_context, last_player_choice="Hello")
    assert schema.npc_context.name == "Bob"
    assert schema.npc_context.age == 45
    assert schema.npc_context.personality == "Calm and welcoming farmer"
    assert schema.npc_context.context == "Working in the fields"
    assert schema.npc_context.talkativeness == Talkativeness.VERY_HIGH
    assert schema.npc_context.main_character_relation == "Stranger"
    assert schema.npc_context.intent.type == "Dialogue"
    assert schema.last_player_choice == "Hello"

def test_dialogue_stream_request_invalid():
    with pytest.raises(ValidationError):
        DialogueStreamRequest(last_player_choice="Hello")  # Missing npc_context

def test_middleware_status_response():
    schema = MiddlewareStatusResponse(state="running")
    assert schema.state == "running"
    assert schema.error_code is None
    
    schema2 = MiddlewareStatusResponse(state="error", error_code="500", message="Internal Error")
    assert schema2.error_code == "500"

def test_language_request():
    schema = LanguageRequest(language=Language.ENGLISH)
    assert schema.language == Language.ENGLISH

def test_toggle_request():
    schema = ToggleRequest(enabled=True)
    assert schema.enabled is True

def test_number_of_options_request_valid():
    schema = NumberOfOptionsRequest(value=3)
    assert schema.value == 3

def test_number_of_options_request_invalid():
    with pytest.raises(ValidationError):
        NumberOfOptionsRequest(value=0)  # ge=1 constraint

def test_settings_updated_response():
    schema = SettingsUpdatedResponse()
    assert schema.status == "ok"
