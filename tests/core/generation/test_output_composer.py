import json
import pytest
from pydantic import ValidationError
from api.schemas import ComposedDialogue, DialogueOptionsSchema, QuestChoiceSchema
from core.generation.output_composer import DialogueOutputComposer
from core.types.contexts import Dialogue, NPCContext, Quest, Talkativeness


@pytest.fixture
def npc_dialogue_context():
    dialogue = Dialogue(type="Dialogue", number_of_options=2)
    return NPCContext(
        name="Theron",
        age=35,
        personality="Helpful",
        context="At the market",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Neutral",
        intent=dialogue,
    )


@pytest.fixture
def npc_quest_context():
    quest = Quest(
        type="Quest",
        objective="Deliver the letter",
        has_choice=True,
        number_of_options=2,
    )
    return NPCContext(
        name="Courier",
        age=22,
        personality="Hurried",
        context="Roadside",
        talkativeness=Talkativeness.LOW,
        main_character_relation="Neutral",
        intent=quest,
    )


# --- Happy Path Tests ---

def test_compose_dialogue_with_dialogue_options(npc_dialogue_context):
    raw_output = json.dumps({
        "dialogue": "Greetings, traveler! What brings you to our fair town?",
        "player_options": {
            "dialogue_options": ["I seek shelter for the night.", "Just passing through."]
        }
    })

    result = DialogueOutputComposer.compose_dialogue(npc_dialogue_context, raw_output)

    assert isinstance(result, ComposedDialogue)
    assert result.dialogue == "Greetings, traveler! What brings you to our fair town?"
    assert result.intent == npc_dialogue_context.intent
    assert isinstance(result.player_options, DialogueOptionsSchema)
    assert result.player_options.dialogue_options == [
        "I seek shelter for the night.",
        "Just passing through.",
    ]


def test_compose_dialogue_without_options(npc_dialogue_context):
    raw_output = json.dumps({
        "dialogue": "I have nothing more to say to you.",
    })

    result = DialogueOutputComposer.compose_dialogue(npc_dialogue_context, raw_output)

    assert isinstance(result, ComposedDialogue)
    assert result.dialogue == "I have nothing more to say to you."
    assert result.player_options is None


def test_compose_dialogue_with_null_player_options(npc_dialogue_context):
    raw_output = json.dumps({
        "dialogue": "Move along, citizen.",
        "player_options": None
    })

    result = DialogueOutputComposer.compose_dialogue(npc_dialogue_context, raw_output)

    assert result.player_options is None


def test_compose_quest_with_choice_and_dialogue_options(npc_quest_context):
    raw_output = json.dumps({
        "dialogue": "Please take this letter to the castle posthaste!",
        "player_options": {
            "accept": "I shall deliver it with haste.",
            "refuse": "Find someone else, I am busy.",
            "dialogue_options": ["Who is the recipient?", "What is the pay?"]
        }
    })

    result = DialogueOutputComposer.compose_dialogue(npc_quest_context, raw_output)

    assert isinstance(result, ComposedDialogue)
    assert result.dialogue == "Please take this letter to the castle posthaste!"
    assert result.intent == npc_quest_context.intent
    assert isinstance(result.player_options, QuestChoiceSchema)
    assert result.player_options.accept == "I shall deliver it with haste."
    assert result.player_options.refuse == "Find someone else, I am busy."
    assert result.player_options.dialogue_options == ["Who is the recipient?", "What is the pay?"]


def test_compose_quest_with_choice_only(npc_quest_context):
    raw_output = json.dumps({
        "dialogue": "Will you help me slay the wolves?",
        "player_options": {
            "accept": "Consider them dealt with.",
            "refuse": "Too dangerous for me.",
            "dialogue_options": None
        }
    })

    result = DialogueOutputComposer.compose_dialogue(npc_quest_context, raw_output)

    assert isinstance(result.player_options, QuestChoiceSchema)
    assert result.player_options.accept == "Consider them dealt with."
    assert result.player_options.refuse == "Too dangerous for me."
    assert result.player_options.dialogue_options is None


def test_compose_quest_without_choice_fields_returns_none_options(npc_quest_context):
    raw_output = json.dumps({
        "dialogue": "Just an update on the situation.",
        "player_options": {
            "dialogue_options": ["Understood."]
        }
    })

    result = DialogueOutputComposer.compose_dialogue(npc_quest_context, raw_output)

    assert result.player_options is None


def test_compose_dialogue_with_null_dialogue_options_returns_none_options(npc_dialogue_context):
    raw_output = json.dumps({
        "dialogue": "Good day.",
        "player_options": {
            "dialogue_options": None
        }
    })

    result = DialogueOutputComposer.compose_dialogue(npc_dialogue_context, raw_output)

    assert result.player_options is None


# --- Error Handling & Validation Tests ---

def test_compose_dialogue_invalid_json(npc_dialogue_context):
    with pytest.raises(ValueError, match="LLM output is not valid JSON"):
        DialogueOutputComposer.compose_dialogue(npc_dialogue_context, "Not valid json at all")


def test_compose_dialogue_empty_string(npc_dialogue_context):
    with pytest.raises(ValueError, match="LLM output is not valid JSON"):
        DialogueOutputComposer.compose_dialogue(npc_dialogue_context, "")


@pytest.mark.parametrize("non_dict_payload", [
    json.dumps([1, 2, 3]),
    json.dumps("string payload"),
    json.dumps(12345),
    json.dumps(True),
])
def test_compose_dialogue_non_dict_root(npc_dialogue_context, non_dict_payload):
    with pytest.raises(ValueError, match="Expected JSON output to be an object"):
        DialogueOutputComposer.compose_dialogue(npc_dialogue_context, non_dict_payload)


def test_compose_dialogue_missing_dialogue_field(npc_dialogue_context):
    raw_output = json.dumps({"player_options": {}})
    with pytest.raises(ValueError, match="Missing or invalid 'dialogue' field in output"):
        DialogueOutputComposer.compose_dialogue(npc_dialogue_context, raw_output)


@pytest.mark.parametrize("invalid_dialogue_val", [123, None, ["line 1"], {"text": "hi"}])
def test_compose_dialogue_invalid_dialogue_type(npc_dialogue_context, invalid_dialogue_val):
    raw_output = json.dumps({"dialogue": invalid_dialogue_val})
    with pytest.raises(ValueError, match="Missing or invalid 'dialogue' field in output"):
        DialogueOutputComposer.compose_dialogue(npc_dialogue_context, raw_output)


@pytest.mark.parametrize("invalid_options_val", ["not a dict", [1, 2, 3], 42])
def test_compose_dialogue_invalid_player_options_type(npc_dialogue_context, invalid_options_val):
    raw_output = json.dumps({
        "dialogue": "Hello",
        "player_options": invalid_options_val,
    })
    with pytest.raises(ValueError, match="Invalid 'player_options' field in output"):
        DialogueOutputComposer.compose_dialogue(npc_dialogue_context, raw_output)
