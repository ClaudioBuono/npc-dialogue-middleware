import pytest
from api.schemas import ComposedDialogue, DialogueOptionsSchema
from core.generation.history import DialogueHistory
from core.types.contexts import Dialogue


@pytest.fixture(autouse=True)
def clean_history():
    """Ensure history is cleared before and after every test."""
    history = DialogueHistory()
    history.clear_dialogue_history()
    yield history
    history.clear_dialogue_history()


def test_initially_empty(clean_history):
    assert clean_history.is_empty() is True
    assert clean_history.get_dialogue_history() == []


def test_add_player_dialogue_to_history(clean_history):
    clean_history.add_player_dialogue_to_history("Hello, good merchant.")

    assert clean_history.is_empty() is False
    recorded = clean_history.get_dialogue_history()
    assert len(recorded) == 1
    assert recorded[0] == {"Player": "Hello, good merchant."}


def test_add_npc_dialogue_to_history(clean_history):
    composed = ComposedDialogue(
        intent=Dialogue(type="Dialogue", number_of_options=0),
        dialogue="Welcome to my store! Take a look around.",
    )
    clean_history.add_npc_dialogue_to_history(composed)

    assert clean_history.is_empty() is False
    recorded = clean_history.get_dialogue_history()
    assert len(recorded) == 1
    assert recorded[0] == {"NPC": "Welcome to my store! Take a look around."}


def test_conversation_turns_chronological_order(clean_history):
    dialogue_intent = Dialogue(type="Dialogue", number_of_options=1)

    clean_history.add_player_dialogue_to_history("Do you have healing potions?")
    clean_history.add_npc_dialogue_to_history(
        ComposedDialogue(
            intent=dialogue_intent,
            dialogue="Yes, I have two left.",
            player_options=DialogueOptionsSchema(dialogue_options=["I'll buy one"]),
        )
    )
    clean_history.add_player_dialogue_to_history("I'll buy one")
    clean_history.add_npc_dialogue_to_history(
        ComposedDialogue(
            intent=dialogue_intent,
            dialogue="Here you go. Be safe out there.",
        )
    )

    recorded = clean_history.get_dialogue_history()
    assert len(recorded) == 4
    assert recorded == [
        {"Player": "Do you have healing potions?"},
        {"NPC": "Yes, I have two left."},
        {"Player": "I'll buy one"},
        {"NPC": "Here you go. Be safe out there."},
    ]


def test_clear_dialogue_history(clean_history):
    clean_history.add_player_dialogue_to_history("Turn 1")
    assert clean_history.is_empty() is False

    clean_history.clear_dialogue_history()

    assert clean_history.is_empty() is True
    assert clean_history.get_dialogue_history() == []


def test_multiple_instances_reference_same_history():
    h1 = DialogueHistory()
    h2 = DialogueHistory()

    h1.add_player_dialogue_to_history("Shared message")
    assert h2.get_dialogue_history() == [{"Player": "Shared message"}]
    assert h2.is_empty() is False


def test_empty_string_player_choice(clean_history):
    clean_history.add_player_dialogue_to_history("")
    assert clean_history.get_dialogue_history() == [{"Player": ""}]
    assert clean_history.is_empty() is False
