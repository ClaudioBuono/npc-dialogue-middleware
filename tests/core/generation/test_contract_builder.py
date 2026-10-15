from unittest.mock import patch
from pydantic import ValidationError
import pytest
from api.schemas import ComposedDialogue, DialogueOptionsSchema, QuestChoiceSchema
from core.configuration.settings import Settings
from core.generation.contract_builder import ContractBuilder
from core.generation.history import DialogueHistory
from core.llm.prompts import (
    FAIRNESS_BASE_RULES_PROMPT,
    JUDGE_BASE_PROMPT,
    JUDGE_RULES_PROMPT,
    HEALER_BASE_PROMPT,
    HEALER_RULES_PROMPT,
    QUEST_CONTENT_RULES_PROMPT,
    ROLE_PROMPT,
    TASK_CHOICE_FIELD_PROMPT,
    TASK_DIALOGUE_PROMPT,
    TASK_OPTIONS_FIELD_TEMPLATE,
    TASK_QUEST_PROMPT,
)
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness
from core.types.dataclasses import Contract, JudgeIssue, JudgeQuestion


@pytest.fixture
def builder():
    return ContractBuilder()


@pytest.fixture
def game_context():
    return GameContext(
        environment="Forest",
        epoch="Medieval",
        world_state="Peaceful",
        main_character_description="A brave knight",
    )


@pytest.fixture
def npc_dialogue_context():
    dialogue = Dialogue(type="Dialogue", number_of_options=2)
    return NPCContext(
        name="Eldrin",
        age=83,
        personality="Wise",
        context="Sitting by the fire",
        talkativeness=Talkativeness.HIGH,
        main_character_relation="Neutral",
        intent=dialogue,
    )


@pytest.fixture
def npc_quest_context():
    quest = Quest(
        type="Quest",
        objective="Find the lost ring",
        has_choice=True,
        number_of_options=2,
        name="The Lost Ring",
        description="A golden ring lost in the woods.",
    )
    return NPCContext(
        name="Elara",
        age=18,
        personality="Anxious",
        context="Looking under a tree",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Friendly",
        intent=quest,
    )


# --- build_dialogue_contract tests ---

def test_build_dialogue_intent_with_options(builder, game_context, npc_dialogue_context):
    contract = builder.build_dialogue_contract(game_context, npc_dialogue_context, dialogue_history=[])

    assert isinstance(contract, Contract)
    assert ROLE_PROMPT in contract.system_prompt
    assert "Eldrin" in contract.user_prompt
    assert "A brave knight" in contract.user_prompt
    assert contract.output_schema["type"] == "object"
    assert "dialogue" in contract.output_schema["properties"]
    assert "player_options" in contract.output_schema["properties"]

    player_options_schema = contract.output_schema["properties"]["player_options"]["properties"]
    assert "dialogue_options" in player_options_schema
    assert "accept" not in player_options_schema
    assert "refuse" not in player_options_schema


def test_build_dialogue_intent_without_options(builder, game_context):
    dialogue = Dialogue(type="Dialogue", number_of_options=0)
    npc_context = NPCContext(
        name="Silent Monk",
        age=40,
        personality="Stoic",
        context="Meditating",
        talkativeness=Talkativeness.LOW,
        main_character_relation="Neutral",
        intent=dialogue,
    )

    contract = builder.build_dialogue_contract(game_context, npc_context, dialogue_history=[])

    assert isinstance(contract, Contract)
    assert "player_options" not in contract.output_schema["properties"]
    assert contract.output_schema["required"] == ["dialogue"]
    assert "# PLAYER OPTIONS" not in contract.system_prompt


def test_build_quest_intent_with_choice_and_options(builder, game_context, npc_quest_context):
    contract = builder.build_dialogue_contract(game_context, npc_quest_context, dialogue_history=[])

    assert isinstance(contract, Contract)
    assert QUEST_CONTENT_RULES_PROMPT in contract.system_prompt
    assert "Elara" in contract.user_prompt
    assert "Find the lost ring" in contract.user_prompt
    assert contract.output_schema["type"] == "object"

    player_options_schema = contract.output_schema["properties"]["player_options"]["properties"]
    assert "accept" in player_options_schema
    assert "refuse" in player_options_schema
    assert "dialogue_options" in player_options_schema
    assert "player_options" in contract.output_schema["required"]


def test_build_quest_intent_with_choice_only(builder, game_context):
    quest = Quest(
        type="Quest",
        objective="Defend the gate",
        has_choice=True,
        number_of_options=0,
    )
    npc_context = NPCContext(
        name="Captain Vance",
        age=45,
        personality="Stern",
        context="On the battlements",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Respectful",
        intent=quest,
    )

    contract = builder.build_dialogue_contract(game_context, npc_context, dialogue_history=[])

    assert "player_options" in contract.output_schema["properties"]
    player_options_schema = contract.output_schema["properties"]["player_options"]["properties"]
    assert "accept" in player_options_schema
    assert "refuse" in player_options_schema
    assert "dialogue_options" not in player_options_schema


def test_build_quest_intent_without_choice_and_without_options(builder, game_context):
    quest = Quest(
        type="Quest",
        objective="Listen to the proclamation",
        has_choice=False,
        number_of_options=0,
    )
    npc_context = NPCContext(
        name="Herald",
        age=30,
        personality="Loud",
        context="Town square",
        talkativeness=Talkativeness.HIGH,
        main_character_relation="Neutral",
        intent=quest,
    )

    contract = builder.build_dialogue_contract(game_context, npc_context, dialogue_history=[])

    assert "player_options" not in contract.output_schema["properties"]
    assert contract.output_schema["required"] == ["dialogue"]


def test_build_with_history(builder, game_context, npc_dialogue_context):
    history = DialogueHistory()
    history.clear_dialogue_history()
    history.add_player_dialogue_to_history("Hello")
    history.add_npc_dialogue_to_history(
        ComposedDialogue(intent=npc_dialogue_context.intent, dialogue="Greetings, traveler.")
    )

    try:
        contract = builder.build_dialogue_contract(
            game_context, npc_dialogue_context, dialogue_history=history.get_dialogue_history()
        )

        assert "Player: Hello" in contract.user_prompt
        assert 'Eldrin: "Greetings, traveler."' in contract.user_prompt
    finally:
        history.clear_dialogue_history()


def test_build_with_empty_history(builder, game_context, npc_dialogue_context):
    contract = builder.build_dialogue_contract(game_context, npc_dialogue_context, dialogue_history=[])
    assert "# CONVERSATION HISTORY" not in contract.user_prompt


def test_build_dialogue_fairness_filter_toggle(builder, game_context, npc_dialogue_context):
    settings = Settings()
    original_filter = settings.fairness_filter

    try:
        settings.fairness_filter = True
        contract_with_filter = builder.build_dialogue_contract(game_context, npc_dialogue_context, [])
        assert FAIRNESS_BASE_RULES_PROMPT in contract_with_filter.system_prompt

        settings.fairness_filter = False
        contract_without_filter = builder.build_dialogue_contract(game_context, npc_dialogue_context, [])
        assert FAIRNESS_BASE_RULES_PROMPT not in contract_without_filter.system_prompt
    finally:
        settings.fairness_filter = original_filter


def test_build_invalid_intent_validation():
    with pytest.raises(ValidationError):
        NPCContext(
            name="Gorg",
            age=15,
            personality="Angry",
            context="Cave",
            talkativeness=Talkativeness.LOW,
            main_character_relation="Hostile",
            intent="Invalid Intent Type",  # type: ignore
        )


# --- build_judge_contract tests ---

def test_build_judge_contract_basic(builder, game_context, npc_dialogue_context):
    composed = ComposedDialogue(
        intent=npc_dialogue_context.intent,
        dialogue="Sit by the fire and rest.",
        player_options=DialogueOptionsSchema(dialogue_options=["Thank you", "Who are you?"]),
    )
    questions = [
        JudgeQuestion(id="q1", text="Does the dialogue fit the setting?"),
        JudgeQuestion(id="q2", text="Is the tone respectful?"),
    ]

    contract = builder.build_judge_contract(
        composed_dialogue=composed,
        game_context=game_context,
        npc_context=npc_dialogue_context,
        questions=questions,
    )

    assert isinstance(contract, Contract)
    assert JUDGE_BASE_PROMPT in contract.system_prompt
    assert JUDGE_RULES_PROMPT in contract.system_prompt
    assert "[Q1]" in contract.user_prompt
    assert "Does the dialogue fit the setting?" in contract.user_prompt
    assert "[Q2]" in contract.user_prompt
    assert "Sit by the fire and rest." in contract.user_prompt

    schema = contract.output_schema
    assert schema["type"] == "object"
    assert "answers" in schema["properties"]
    items_schema = schema["properties"]["answers"]["items"]
    assert items_schema["properties"]["id"]["enum"] == ["q1", "q2"]
    assert "reason" in items_schema["required"]
    assert "answer" in items_schema["required"]


def test_build_judge_contract_with_bad_reasons(builder, game_context, npc_dialogue_context):
    composed = ComposedDialogue(
        intent=npc_dialogue_context.intent,
        dialogue="Welcome to my camp.",
    )
    questions = [JudgeQuestion(id="q1", text="Is it lore compliant?")]
    bad_reasons_text = "The previous explanation did not cite the dialogue properly."

    contract = builder.build_judge_contract(
        composed_dialogue=composed,
        game_context=game_context,
        npc_context=npc_dialogue_context,
        questions=questions,
        bad_reasons=bad_reasons_text,
    )

    assert bad_reasons_text in contract.user_prompt


# --- build_healer_contract tests ---

def test_build_healer_contract(builder, game_context, npc_quest_context):
    composed = ComposedDialogue(
        intent=npc_quest_context.intent,
        dialogue="I lost something very precious.",
        player_options=QuestChoiceSchema(
            accept="I will look for it",
            refuse="Not my problem",
            dialogue_options=["What was it?"],
        ),
    )
    issues = [
        JudgeIssue(category="Lore", issue="Mentioned modern technology in a medieval setting."),
        JudgeIssue(category="Formatting", issue="Missing quest reward indication."),
    ]

    contract = builder.build_healer_contract(
        composed_dialogue=composed,
        game_context=game_context,
        npc_context=npc_quest_context,
        issues=issues,
    )

    assert isinstance(contract, Contract)
    assert HEALER_BASE_PROMPT in contract.system_prompt
    assert HEALER_RULES_PROMPT in contract.system_prompt
    assert "Mentioned modern technology" in contract.user_prompt
    assert "Missing quest reward indication" in contract.user_prompt
    assert "Elara" in contract.user_prompt
    assert "The Lost Ring" in contract.user_prompt
    # Output schema matches npc_context schema
    assert "player_options" in contract.output_schema["properties"]
    assert "accept" in contract.output_schema["properties"]["player_options"]["properties"]


# --- Static and helper methods isolated tests ---

def test_build_player_options_section_combinations():
    # Neither options nor choice
    assert ContractBuilder._build_player_options_section(False, False, 0) is None

    # Options only
    sec_opts = ContractBuilder._build_player_options_section(True, False, 3)
    assert sec_opts is not None
    assert "# PLAYER OPTIONS" in sec_opts
    assert "3" in sec_opts

    # Choice only
    sec_choice = ContractBuilder._build_player_options_section(False, True, 0)
    assert sec_choice is not None
    assert "# PLAYER OPTIONS" in sec_choice
    assert '"accept"' in sec_choice
    assert '"refuse"' in sec_choice

    # Both options and choice
    sec_both = ContractBuilder._build_player_options_section(True, True, 2)
    assert sec_both is not None
    assert "# PLAYER OPTIONS" in sec_both
    assert "2" in sec_both
    assert '"accept"' in sec_both
    assert '"refuse"' in sec_both


def test_build_task_section_combinations():
    # Plain dialogue, single deliverable
    sec_single = ContractBuilder._build_task_section(is_quest=False, has_options=False, has_choice=False, number_of_options=0)
    assert sec_single == TASK_DIALOGUE_PROMPT

    # Quest, multiple deliverables
    sec_quest = ContractBuilder._build_task_section(is_quest=True, has_options=True, has_choice=True, number_of_options=2)
    assert TASK_QUEST_PROMPT in sec_quest
    assert TASK_CHOICE_FIELD_PROMPT in sec_quest
    assert TASK_OPTIONS_FIELD_TEMPLATE.format(number_of_options=2) in sec_quest


def test_build_judge_output_schema_without_reason(builder):
    schema = builder._build_judge_output_schema(question_ids=["q1", "q2"], include_reason=False)
    items_props = schema["properties"]["answers"]["items"]["properties"]
    assert "reason" not in items_props
    assert "id" in items_props
    assert "answer" in items_props


def test_extend_player_options_schema_quest_no_choice(builder):
    quest = Quest(type="Quest", objective="Find ring", has_choice=False, number_of_options=0)
    extended = builder._extend_player_options_schema_quest(quest, base_schema=None)
    assert extended is None
