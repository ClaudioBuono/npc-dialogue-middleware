from unittest.mock import MagicMock
import pytest
from core.configuration.thresholds import (
    MAX_AGE,
    MAX_BACKSTORY_LENGTH,
    MAX_CONTEXT_LENGTH,
    MAX_ENVIRONMENT_LENGTH,
    MAX_EPOCH_LENGTH,
    MAX_LANGUAGE_ENTRIES,
    MAX_MAIN_CHARACTER_DESCRIPTION_LENGTH,
    MAX_MORE_INFO_LENGTH,
    MAX_MUST_USE_EXPRESSION_LENGTH,
    MAX_NAME_LENGTH,
    MAX_NUMBER_OF_OPTIONS_LENGTH,
    MAX_PERSONALITY_LENGTH,
    MAX_QUEST_DESCRIPTION_LENGTH,
    MAX_QUEST_LOCATION_LENGTH,
    MAX_QUEST_NAME_LENGTH,
    MAX_QUEST_OBJECTIVE_LENGTH,
    MAX_QUEST_REWARD_LENGTH,
    MAX_RECENT_PLOT_LENGTH,
    MAX_RELATION_LENGTH,
    MAX_VISUAL_DESCRIPTION_LENGTH,
    MAX_WORLD_STATE_LENGTH,
    MIN_AGE,
    MIN_EPOCH_LENGTH,
    MIN_NAME_LENGTH,
)
from core.generation.pre_processing import (
    _DialogueBaseFieldsResult,
    _IntentValidationResult,
    _validate_dialogue,
    _validate_dialogue_base_fields,
    _validate_quest,
    normalize_and_validate_game_context,
    normalize_and_validate_npc_context,
)
from core.tools.errors import PreProcessingError, ValidationErrorCode
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness


# --- Fixtures ---

@pytest.fixture
def valid_game_context():
    return GameContext(
        epoch="Medieval Fantasy",
        environment="Dense enchanted woodland with ancient ruins.",
        world_state="An uneasy armistice holds between rival city-states.",
        main_character_description="A weathered sellsword with scarred armor.",
    )


@pytest.fixture
def valid_dialogue():
    return Dialogue(
        type="Dialogue",
        must_use_expression="Keep your voice down.",
        more_info="Guards are patrolling nearby.",
        number_of_options=2,
    )


@pytest.fixture
def valid_quest():
    return Quest(
        type="Quest",
        objective="Infiltrate the keep and retrieve the ledger.",
        name="The Silent Infiltration",
        description="The local baron's taxes are kept locked in the upper vault.",
        location="Baron's Keep, Upper Tower",
        reward="250 gold pieces",
        has_choice=True,
        must_use_expression="Be quiet about this.",
        more_info="The baron's guards change shift at midnight.",
        number_of_options=3,
    )


@pytest.fixture
def valid_npc_context(valid_dialogue):
    return NPCContext(
        name="Garrick",
        age=45,
        personality="Gruff, pragmatic, cautious.",
        context="Leaning against a tavern barrel, dusk.",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Wary acquaintance",
        recent_plot="Smugglers were captured yesterday.",
        visual_description="Broad shoulders, wearing a worn wool tunic.",
        backstory="Former city watchman turned informant.",
        language=["Common", "Thieves' Cant"],
        intent=valid_dialogue,
    )


# ==============================================================================
# 1. normalize_and_validate_game_context Tests
# ==============================================================================

def test_validate_game_context_happy_path(valid_game_context):
    normalized = normalize_and_validate_game_context(valid_game_context)
    assert isinstance(normalized, GameContext)
    assert normalized.epoch == valid_game_context.epoch
    assert normalized.environment == valid_game_context.environment
    assert normalized.world_state == valid_game_context.world_state
    assert normalized.main_character_description == valid_game_context.main_character_description


def test_validate_game_context_strips_whitespace():
    context = GameContext(
        epoch="  Medieval  ",
        environment="  Forest  ",
        world_state="  Peaceful  ",
        main_character_description="  A knight  ",
    )
    normalized = normalize_and_validate_game_context(context)
    assert normalized.epoch == "Medieval"
    assert normalized.environment == "Forest"
    assert normalized.world_state == "Peaceful"
    assert normalized.main_character_description == "A knight"


def test_validate_game_context_none_main_character_description():
    context = GameContext(
        epoch="Medieval",
        environment="Forest",
        world_state="Peaceful",
        main_character_description=None,
    )
    normalized = normalize_and_validate_game_context(context)
    assert normalized.main_character_description is None


def test_validate_game_context_epoch_boundary_min_length():
    context = GameContext(
        epoch="a" * MIN_EPOCH_LENGTH,
        environment="Forest",
        world_state="Peaceful",
    )
    normalized = normalize_and_validate_game_context(context)
    assert len(normalized.epoch) == MIN_EPOCH_LENGTH


def test_validate_game_context_epoch_boundary_max_length():
    context = GameContext(
        epoch="a" * MAX_EPOCH_LENGTH,
        environment="Forest",
        world_state="Peaceful",
    )
    normalized = normalize_and_validate_game_context(context)
    assert len(normalized.epoch) == MAX_EPOCH_LENGTH


@pytest.mark.parametrize("invalid_epoch,expected_msg", [
    ("", "cannot be empty or whitespace only"),
    ("   ", "cannot be empty or whitespace only"),
    ("a", f"must be at least {MIN_EPOCH_LENGTH} characters long"),
    ("a" * (MAX_EPOCH_LENGTH + 1), f"must not exceed {MAX_EPOCH_LENGTH} characters"),
])
def test_validate_game_context_invalid_epoch(invalid_epoch, expected_msg):
    context = GameContext(
        epoch=invalid_epoch,
        environment="Forest",
        world_state="Peaceful",
    )
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_game_context(context)
    assert exc_info.value.code == ValidationErrorCode.INVALID_VALUE
    assert any(expected_msg in e for e in exc_info.value.errors)


@pytest.mark.parametrize("invalid_env,expected_msg", [
    ("", "cannot be empty or whitespace only"),
    ("   ", "cannot be empty or whitespace only"),
    ("a" * (MAX_ENVIRONMENT_LENGTH + 1), f"must not exceed {MAX_ENVIRONMENT_LENGTH} characters"),
])
def test_validate_game_context_invalid_environment(invalid_env, expected_msg):
    context = GameContext(
        epoch="Medieval",
        environment=invalid_env,
        world_state="Peaceful",
    )
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_game_context(context)
    assert any(expected_msg in e for e in exc_info.value.errors)


@pytest.mark.parametrize("invalid_world_state,expected_msg", [
    ("", "cannot be empty or whitespace only"),
    ("   ", "cannot be empty or whitespace only"),
    ("a" * (MAX_WORLD_STATE_LENGTH + 1), f"must not exceed {MAX_WORLD_STATE_LENGTH} characters"),
])
def test_validate_game_context_invalid_world_state(invalid_world_state, expected_msg):
    context = GameContext(
        epoch="Medieval",
        environment="Forest",
        world_state=invalid_world_state,
    )
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_game_context(context)
    assert any(expected_msg in e for e in exc_info.value.errors)


def test_validate_game_context_main_character_description_too_long():
    context = GameContext(
        epoch="Medieval",
        environment="Forest",
        world_state="Peaceful",
        main_character_description="a" * (MAX_MAIN_CHARACTER_DESCRIPTION_LENGTH + 1),
    )
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_game_context(context)
    assert any(f"must not exceed {MAX_MAIN_CHARACTER_DESCRIPTION_LENGTH} characters" in e for e in exc_info.value.errors)


def test_validate_game_context_collects_multiple_errors():
    context = GameContext(
        epoch="",
        environment="",
        world_state="",
    )
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_game_context(context)
    assert len(exc_info.value.errors) == 3


# ==============================================================================
# 2. normalize_and_validate_npc_context Tests
# ==============================================================================

def test_validate_npc_context_happy_path_dialogue(valid_npc_context):
    normalized = normalize_and_validate_npc_context(valid_npc_context)
    assert isinstance(normalized, NPCContext)
    assert normalized.name == valid_npc_context.name
    assert normalized.age == valid_npc_context.age
    assert normalized.intent.type == "Dialogue"


def test_validate_npc_context_happy_path_quest(valid_npc_context, valid_quest):
    valid_npc_context.intent = valid_quest
    normalized = normalize_and_validate_npc_context(valid_npc_context)
    assert isinstance(normalized, NPCContext)
    assert normalized.intent.type == "Quest"
    assert isinstance(normalized.intent, Quest)
    assert normalized.intent.objective == valid_quest.objective


def test_validate_npc_context_strips_all_string_fields(valid_dialogue):
    npc = NPCContext(
        name="  Garrick  ",
        age=30,
        personality="  Brave  ",
        context="  Castle  ",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="  Friend  ",
        recent_plot="  War started  ",
        visual_description="  Tall  ",
        backstory="  Knight  ",
        language=["  Common  ", "  Elvish  "],
        intent=valid_dialogue,
    )
    normalized = normalize_and_validate_npc_context(npc)
    assert normalized.name == "Garrick"
    assert normalized.personality == "Brave"
    assert normalized.context == "Castle"
    assert normalized.main_character_relation == "Friend"
    assert normalized.recent_plot == "War started"
    assert normalized.visual_description == "Tall"
    assert normalized.backstory == "Knight"
    assert normalized.language == ["Common", "Elvish"]


# --- Name boundaries & errors ---

def test_validate_npc_context_name_boundaries(valid_npc_context):
    valid_npc_context.name = "a" * MIN_NAME_LENGTH
    assert normalize_and_validate_npc_context(valid_npc_context).name == "a" * MIN_NAME_LENGTH

    valid_npc_context.name = "a" * MAX_NAME_LENGTH
    assert normalize_and_validate_npc_context(valid_npc_context).name == "a" * MAX_NAME_LENGTH


@pytest.mark.parametrize("invalid_name,expected_msg", [
    ("", "cannot be empty or whitespace only"),
    ("   ", "cannot be empty or whitespace only"),
    ("a", f"must be at least {MIN_NAME_LENGTH} characters long"),
    ("a" * (MAX_NAME_LENGTH + 1), f"must not exceed {MAX_NAME_LENGTH} characters"),
])
def test_validate_npc_context_invalid_name(valid_npc_context, invalid_name, expected_msg):
    valid_npc_context.name = invalid_name
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any(expected_msg in e for e in exc_info.value.errors)


# --- Age boundaries & errors ---

def test_validate_npc_context_age_boundaries(valid_npc_context):
    valid_npc_context.age = MIN_AGE
    assert normalize_and_validate_npc_context(valid_npc_context).age == MIN_AGE

    valid_npc_context.age = MAX_AGE
    assert normalize_and_validate_npc_context(valid_npc_context).age == MAX_AGE


@pytest.mark.parametrize("invalid_age", [MIN_AGE - 1, MAX_AGE + 1])
def test_validate_npc_context_invalid_age(valid_npc_context, invalid_age):
    valid_npc_context.age = invalid_age
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any(f"must be between {MIN_AGE} and {MAX_AGE}" in e for e in exc_info.value.errors)


# --- Personality, Context, Relation ---

@pytest.mark.parametrize("invalid_val,expected_msg", [
    ("", "cannot be empty or whitespace only"),
    ("   ", "cannot be empty or whitespace only"),
    ("a" * (MAX_PERSONALITY_LENGTH + 1), f"must not exceed {MAX_PERSONALITY_LENGTH} characters"),
])
def test_validate_npc_context_invalid_personality(valid_npc_context, invalid_val, expected_msg):
    valid_npc_context.personality = invalid_val
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any(expected_msg in e for e in exc_info.value.errors)


@pytest.mark.parametrize("invalid_val,expected_msg", [
    ("", "cannot be empty or whitespace only"),
    ("   ", "cannot be empty or whitespace only"),
    ("a" * (MAX_CONTEXT_LENGTH + 1), f"must not exceed {MAX_CONTEXT_LENGTH} characters"),
])
def test_validate_npc_context_invalid_context_field(valid_npc_context, invalid_val, expected_msg):
    valid_npc_context.context = invalid_val
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any(expected_msg in e for e in exc_info.value.errors)


@pytest.mark.parametrize("invalid_val,expected_msg", [
    ("", "cannot be empty or whitespace only"),
    ("   ", "cannot be empty or whitespace only"),
    ("a" * (MAX_RELATION_LENGTH + 1), f"must not exceed {MAX_RELATION_LENGTH} characters"),
])
def test_validate_npc_context_invalid_relation(valid_npc_context, invalid_val, expected_msg):
    valid_npc_context.main_character_relation = invalid_val
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any(expected_msg in e for e in exc_info.value.errors)


# --- Optional string fields ---

def test_validate_npc_context_optional_fields_too_long(valid_npc_context):
    valid_npc_context.recent_plot = "a" * (MAX_RECENT_PLOT_LENGTH + 1)
    valid_npc_context.visual_description = "a" * (MAX_VISUAL_DESCRIPTION_LENGTH + 1)
    valid_npc_context.backstory = "a" * (MAX_BACKSTORY_LENGTH + 1)

    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    errors = exc_info.value.errors
    assert any(f"must not exceed {MAX_RECENT_PLOT_LENGTH} characters" in e for e in errors)
    assert any(f"must not exceed {MAX_VISUAL_DESCRIPTION_LENGTH} characters" in e for e in errors)
    assert any(f"must not exceed {MAX_BACKSTORY_LENGTH} characters" in e for e in errors)


def test_validate_npc_context_language_none_is_valid(valid_npc_context):
    valid_npc_context.language = None
    normalized = normalize_and_validate_npc_context(valid_npc_context)
    assert normalized.language is None


@pytest.mark.parametrize("empty_languages", [[], ["  ", ""], ["   "]])
def test_validate_npc_context_language_empty_list_fails(valid_npc_context, empty_languages):
    valid_npc_context.language = empty_languages
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any("must contain at least one non-empty entry" in e for e in exc_info.value.errors)


def test_validate_npc_context_language_too_many_entries(valid_npc_context):
    valid_npc_context.language = [f"Lang{i}" for i in range(MAX_LANGUAGE_ENTRIES + 1)]
    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any(f"must not contain more than {MAX_LANGUAGE_ENTRIES} entries" in e for e in exc_info.value.errors)


# ==============================================================================
# 3. Intent & Helper Functions Tests
# ==============================================================================

def test_validate_dialogue_base_fields_happy_path(valid_dialogue):
    result = _validate_dialogue_base_fields(valid_dialogue)
    assert isinstance(result, _DialogueBaseFieldsResult)
    assert result.errors == []
    assert result.must_use_expression == "Keep your voice down."
    assert result.more_info == "Guards are patrolling nearby."


def test_validate_dialogue_base_fields_optional_lengths():
    dialogue = Dialogue(
        type="Dialogue",
        must_use_expression="a" * (MAX_MUST_USE_EXPRESSION_LENGTH + 1),
        more_info="a" * (MAX_MORE_INFO_LENGTH + 1),
        number_of_options=0,
    )
    result = _validate_dialogue_base_fields(dialogue)
    assert len(result.errors) == 2
    assert any(f"must not exceed {MAX_MUST_USE_EXPRESSION_LENGTH} characters" in e for e in result.errors)
    assert any(f"must not exceed {MAX_MORE_INFO_LENGTH} characters" in e for e in result.errors)


@pytest.mark.parametrize("invalid_options,expected_msg", [
    (True, "must be an integer"),
    (False, "must be an integer"),
    ("two", "must be an integer"),
    (-1, f"must be in the range 0 and {MAX_NUMBER_OF_OPTIONS_LENGTH}"),
    (MAX_NUMBER_OF_OPTIONS_LENGTH + 1, f"must be in the range 0 and {MAX_NUMBER_OF_OPTIONS_LENGTH}"),
])
def test_validate_dialogue_base_fields_invalid_number_of_options(invalid_options, expected_msg):
    dialogue = Dialogue.model_construct(
        type="Dialogue",
        must_use_expression=None,
        more_info=None,
        number_of_options=invalid_options,
    )
    result = _validate_dialogue_base_fields(dialogue)
    assert any(expected_msg in e for e in result.errors)


def test_validate_quest_happy_path(valid_quest):
    result = _validate_quest(valid_quest)
    assert isinstance(result, _IntentValidationResult)
    assert result.errors == []
    assert result.normalized.objective == valid_quest.objective
    assert result.normalized.name == valid_quest.name
    assert result.normalized.has_choice == valid_quest.has_choice


@pytest.mark.parametrize("invalid_objective,expected_msg", [
    ("", "cannot be empty or whitespace only"),
    ("   ", "cannot be empty or whitespace only"),
    ("a" * (MAX_QUEST_OBJECTIVE_LENGTH + 1), f"must not exceed {MAX_QUEST_OBJECTIVE_LENGTH} characters"),
])
def test_validate_quest_invalid_objective(valid_quest, invalid_objective, expected_msg):
    valid_quest.objective = invalid_objective
    result = _validate_quest(valid_quest)
    assert any(expected_msg in e for e in result.errors)


def test_validate_quest_optional_fields_too_long(valid_quest):
    valid_quest.name = "a" * (MAX_QUEST_NAME_LENGTH + 1)
    valid_quest.description = "a" * (MAX_QUEST_DESCRIPTION_LENGTH + 1)
    valid_quest.location = "a" * (MAX_QUEST_LOCATION_LENGTH + 1)
    valid_quest.reward = "a" * (MAX_QUEST_REWARD_LENGTH + 1)

    result = _validate_quest(valid_quest)
    assert len(result.errors) == 4
    assert any(f"must not exceed {MAX_QUEST_NAME_LENGTH} characters" in e for e in result.errors)
    assert any(f"must not exceed {MAX_QUEST_DESCRIPTION_LENGTH} characters" in e for e in result.errors)
    assert any(f"must not exceed {MAX_QUEST_LOCATION_LENGTH} characters" in e for e in result.errors)
    assert any(f"must not exceed {MAX_QUEST_REWARD_LENGTH} characters" in e for e in result.errors)


def test_validate_dialogue_happy_path(valid_dialogue):
    result = _validate_dialogue(valid_dialogue)
    assert isinstance(result, _IntentValidationResult)
    assert result.errors == []
    assert result.normalized.must_use_expression == "Keep your voice down."


# --- Intent Type Mismatch Branches ---

def test_validate_npc_context_intent_type_quest_but_not_quest_instance(valid_npc_context):
    mock_intent = MagicMock()
    mock_intent.type = "Quest"
    # mock_intent is not an instance of Quest
    valid_npc_context.intent = mock_intent

    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any("is 'Quest' but the object is not a Quest instance" in e for e in exc_info.value.errors)


def test_validate_npc_context_intent_type_dialogue_but_is_quest_instance(valid_npc_context, valid_quest):
    # Quest inherits from Dialogue, but type is explicitly set to Dialogue
    quest_as_dialogue = Quest.model_construct(
        type="Dialogue",
        objective="Do something",
        number_of_options=0,
    )
    valid_npc_context.intent = quest_as_dialogue

    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any("is 'Dialogue' but the object is not a plain Dialogue instance" in e for e in exc_info.value.errors)


def test_validate_npc_context_intent_unknown_type(valid_npc_context):
    mock_intent = MagicMock()
    mock_intent.type = "Bribe"
    valid_npc_context.intent = mock_intent

    with pytest.raises(PreProcessingError) as exc_info:
        normalize_and_validate_npc_context(valid_npc_context)
    assert any("Unknown value for field 'intent.type': 'Bribe'" in e for e in exc_info.value.errors)
