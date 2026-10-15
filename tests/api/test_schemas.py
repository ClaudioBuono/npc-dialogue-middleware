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
    CensorWordRequest,
    IntegerRequest,
    SettingsUpdatedResponse,
    GenerateDialogueRequest,
)
from core.types.contexts import Dialogue, Quest, NPCContext, Talkativeness
from core.types.enums import Language


# --------------------------------------------------------------------------- #
# DialogueOptionsSchema
# --------------------------------------------------------------------------- #
def test_dialogue_options_schema_valid():
    schema = DialogueOptionsSchema(dialogue_options=["Option 1", "Option 2"])
    assert schema.dialogue_options == ["Option 1", "Option 2"]


def test_dialogue_options_schema_invalid():
    with pytest.raises(ValidationError):
        DialogueOptionsSchema(dialogue_options="Not a list")

    with pytest.raises(ValidationError):
        DialogueOptionsSchema()  # missing required dialogue_options


# --------------------------------------------------------------------------- #
# QuestChoiceSchema
# --------------------------------------------------------------------------- #
def test_quest_choice_schema_valid():
    schema = QuestChoiceSchema(accept="Yes", refuse="No", dialogue_options=["More info?"])
    assert schema.accept == "Yes"
    assert schema.refuse == "No"
    assert schema.dialogue_options == ["More info?"]

    # Without optional dialogue_options (defaults to None)
    schema2 = QuestChoiceSchema(accept="Yes", refuse="No")
    assert schema2.dialogue_options is None


def test_quest_choice_schema_invalid():
    with pytest.raises(ValidationError):
        QuestChoiceSchema(accept="Yes")  # Missing refuse

    with pytest.raises(ValidationError):
        QuestChoiceSchema(refuse="No")  # Missing accept


# --------------------------------------------------------------------------- #
# ComposedDialogue
# --------------------------------------------------------------------------- #
def test_composed_dialogue_with_dialogue_intent():
    dialogue_intent = Dialogue()
    schema = ComposedDialogue(
        intent=dialogue_intent,
        dialogue="Hello traveler!",
        player_options=DialogueOptionsSchema(dialogue_options=["Hi", "Bye"]),
    )
    assert schema.dialogue == "Hello traveler!"
    assert schema.intent.type == "Dialogue"
    assert isinstance(schema.player_options, DialogueOptionsSchema)


def test_composed_dialogue_with_quest_intent():
    quest_intent = Quest(objective="Defeat the dragon")
    schema = ComposedDialogue(
        intent=quest_intent,
        dialogue="Will you slay the dragon?",
        player_options=QuestChoiceSchema(accept="I will", refuse="Too dangerous"),
    )
    assert schema.dialogue == "Will you slay the dragon?"
    assert schema.intent.type == "Quest"
    assert isinstance(schema.player_options, QuestChoiceSchema)


def test_composed_dialogue_without_player_options():
    schema = ComposedDialogue(
        intent=Dialogue(),
        dialogue="Just some idle gossip.",
    )
    assert schema.player_options is None


def test_composed_dialogue_invalid_intent():
    with pytest.raises(ValidationError):
        ComposedDialogue(
            intent={"invalid": "intent"},
            dialogue="Hello traveler!",
        )


def test_composed_dialogue_missing_dialogue():
    with pytest.raises(ValidationError):
        ComposedDialogue(intent=Dialogue())


# --------------------------------------------------------------------------- #
# DialogueStreamRequest
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_npc_context():
    return NPCContext(
        name="Bob",
        age=45,
        personality="Calm and welcoming farmer",
        context="Working in the fields",
        talkativeness=Talkativeness.VERY_HIGH,
        main_character_relation="Stranger",
        intent=Dialogue(),
    )


def test_dialogue_stream_request_valid(sample_npc_context):
    schema = DialogueStreamRequest(npc_context=sample_npc_context, last_player_choice="Hello")
    assert schema.npc_context.name == "Bob"
    assert schema.npc_context.age == 45
    assert schema.last_player_choice == "Hello"


def test_dialogue_stream_request_optional_last_player_choice(sample_npc_context):
    schema = DialogueStreamRequest(npc_context=sample_npc_context)
    assert schema.last_player_choice is None


def test_dialogue_stream_request_invalid():
    with pytest.raises(ValidationError):
        DialogueStreamRequest(last_player_choice="Hello")  # Missing npc_context


# --------------------------------------------------------------------------- #
# MiddlewareStatusResponse
# --------------------------------------------------------------------------- #
def test_middleware_status_response_state_only():
    schema = MiddlewareStatusResponse(state="Idle")
    assert schema.state == "Idle"
    assert schema.error_code is None
    assert schema.message is None


def test_middleware_status_response_with_errors():
    schema = MiddlewareStatusResponse(
        state="Starting", error_code="starting", message="Middleware is starting"
    )
    assert schema.state == "Starting"
    assert schema.error_code == "starting"
    assert schema.message == "Middleware is starting"


def test_middleware_status_response_invalid():
    with pytest.raises(ValidationError):
        MiddlewareStatusResponse()  # Missing state


# --------------------------------------------------------------------------- #
# LanguageRequest
# --------------------------------------------------------------------------- #
def test_language_request_valid():
    schema = LanguageRequest(language=Language.ENGLISH)
    assert schema.language == Language.ENGLISH

    schema_italian = LanguageRequest(language=Language.ITALIAN)
    assert schema_italian.language == Language.ITALIAN


def test_language_request_invalid():
    with pytest.raises(ValidationError):
        LanguageRequest(language="Klingon")


# --------------------------------------------------------------------------- #
# ToggleRequest
# --------------------------------------------------------------------------- #
def test_toggle_request_valid():
    schema_true = ToggleRequest(enabled=True)
    assert schema_true.enabled is True

    schema_false = ToggleRequest(enabled=False)
    assert schema_false.enabled is False


def test_toggle_request_invalid():
    with pytest.raises(ValidationError):
        ToggleRequest()  # Missing enabled


# --------------------------------------------------------------------------- #
# CensorWordRequest
# --------------------------------------------------------------------------- #
def test_censor_word_request_valid():
    schema = CensorWordRequest(word="[CENSORED]")
    assert schema.word == "[CENSORED]"


def test_censor_word_request_boundary():
    # Min length boundary: 1 char
    schema_min = CensorWordRequest(word="*")
    assert schema_min.word == "*"

    # Max length boundary: 32 chars
    max_word = "X" * 32
    schema_max = CensorWordRequest(word=max_word)
    assert schema_max.word == max_word


def test_censor_word_request_invalid():
    # Empty string (fails min_length=1)
    with pytest.raises(ValidationError):
        CensorWordRequest(word="")

    # 33 characters (fails max_length=32)
    with pytest.raises(ValidationError):
        CensorWordRequest(word="X" * 33)

    # Missing word
    with pytest.raises(ValidationError):
        CensorWordRequest()


# --------------------------------------------------------------------------- #
# IntegerRequest
# --------------------------------------------------------------------------- #
def test_integer_request_valid():
    schema = IntegerRequest(value=3)
    assert schema.value == 3

    # Boundary: ge=1
    schema_boundary = IntegerRequest(value=1)
    assert schema_boundary.value == 1


def test_integer_request_invalid():
    with pytest.raises(ValidationError):
        IntegerRequest(value=0)  # violates ge=1

    with pytest.raises(ValidationError):
        IntegerRequest(value=-5)  # negative value

    with pytest.raises(ValidationError):
        IntegerRequest()  # missing value


# --------------------------------------------------------------------------- #
# SettingsUpdatedResponse
# --------------------------------------------------------------------------- #
def test_settings_updated_response_default():
    schema = SettingsUpdatedResponse()
    assert schema.status == "ok"


def test_settings_updated_response_custom():
    schema = SettingsUpdatedResponse(status="updated")
    assert schema.status == "updated"


# --------------------------------------------------------------------------- #
# GenerateDialogueRequest
# --------------------------------------------------------------------------- #
def test_generate_dialogue_request_defaults(sample_npc_context):
    schema = GenerateDialogueRequest(npc_context=sample_npc_context)
    assert schema.npc_context.name == "Bob"
    assert schema.last_player_choice is None
    assert schema.stream is False


def test_generate_dialogue_request_custom(sample_npc_context):
    schema = GenerateDialogueRequest(
        npc_context=sample_npc_context,
        last_player_choice="I accept",
        stream=True,
    )
    assert schema.npc_context.name == "Bob"
    assert schema.last_player_choice == "I accept"
    assert schema.stream is True


def test_generate_dialogue_request_invalid():
    with pytest.raises(ValidationError):
        GenerateDialogueRequest()  # Missing npc_context
