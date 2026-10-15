from pydantic import BaseModel
import pytest
from unittest.mock import MagicMock
from typing import Optional, Union
from core.routing.complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityScore,
    _ContextLengthScore,
    _COMMON_LENGTH_THRESHOLD_NAMES,
    _QUEST_ONLY_LENGTH_THRESHOLD_NAMES,
)
from core.types.contexts import Dialogue, Quest, GameContext, NPCContext, Talkativeness
from core.types.enums import ComplexityTier
from core.routing.helpers import classify_score_to_complexity_tier

class _RichnessProbe(BaseModel):
    """Minimal model: one optional text field and one boolean flag."""
    text_field: str | None = None
    flag_field: bool = False

# ------------------------------------------------------------------
# Initialization & Validation Tests
# ------------------------------------------------------------------

def test_complexity_analyzer_init_validation():
    """Verifies constructor validation logic for bounds, types, and required keys."""

    # Valid configuration with a single max_tokens float
    analyzer1 = ComplexityAnalyzer(min_tokens=100, max_tokens=500)
    assert analyzer1._min_tokens == 100
    assert analyzer1._max_tokens_by_intent == {"dialogue": 500, "quest": 500}

    # Valid configuration with mapping dictionary
    analyzer2 = ComplexityAnalyzer(min_tokens=100, max_tokens={"dialogue": 400, "quest": 800})
    assert analyzer2._max_tokens_by_intent == {"dialogue": 400, "quest": 800}

    # Missing required keys in max_tokens dictionary mapping
    with pytest.raises(ValueError) as excinfo:
        ComplexityAnalyzer(min_tokens=100, max_tokens={"dialogue": 400})
    assert "max_tokens dict must contain the keys 'dialogue' and 'quest'" in str(excinfo.value)

    # Boundary check: max_tokens <= min_tokens (single float)
    with pytest.raises(ValueError) as excinfo:
        ComplexityAnalyzer(min_tokens=200, max_tokens=150)
    assert "must be greater than min_tokens" in str(excinfo.value)

    # Boundary check: max_tokens <= min_tokens (dictionary mapping)
    with pytest.raises(ValueError) as excinfo:
        ComplexityAnalyzer(min_tokens=200, max_tokens={"dialogue": 150, "quest": 300})
    assert "must be greater than min_tokens" in str(excinfo.value)

    # Boundary check: chars_per_token must be strictly positive
    with pytest.raises(ValueError) as excinfo:
        ComplexityAnalyzer(min_tokens=50, max_tokens=100, chars_per_token=0)
    assert "chars_per_token must be positive" in str(excinfo.value)


# ------------------------------------------------------------------
# Dynamic Factory Test (from_thresholds)
# ------------------------------------------------------------------

def test_complexity_analyzer_from_thresholds_dynamic():
    """
    Verifies that from_thresholds dynamically extracts limits from the thresholds module.
    Calculations are self-adaptive to prevent test breakage if threshold lists change.
    """

    mock_thresholds = MagicMock()
    
    # Dynamically populate all threshold attributes with dummy values on our mock
    for const_name in _COMMON_LENGTH_THRESHOLD_NAMES.values():
        setattr(mock_thresholds, const_name, 100)
    for const_name in _QUEST_ONLY_LENGTH_THRESHOLD_NAMES.values():
        setattr(mock_thresholds, const_name, 200)

    # Calculate expected outcomes programmatically based on the actual dictionary sizes
    num_common_fields = len(_COMMON_LENGTH_THRESHOLD_NAMES)
    num_quest_fields = len(_QUEST_ONLY_LENGTH_THRESHOLD_NAMES)
    
    expected_common_chars = num_common_fields * 100
    expected_quest_chars = expected_common_chars + (num_quest_fields * 200)

    chars_per_token = 4.0
    realistic_ratio = 0.2

    expected_dialogue_max = (expected_common_chars / chars_per_token) * realistic_ratio
    expected_quest_max = (expected_quest_chars / chars_per_token) * realistic_ratio

    analyzer = ComplexityAnalyzer.from_thresholds(
        thresholds=mock_thresholds,
        realistic_max_ratio=realistic_ratio,
        min_tokens=10,
        chars_per_token=chars_per_token
    )

    assert analyzer._min_tokens == 10
    assert analyzer._max_tokens_by_intent["dialogue"] == expected_dialogue_max
    assert analyzer._max_tokens_by_intent["quest"] == expected_quest_max


# ------------------------------------------------------------------
# Text Field Parsing & Type Introspection Tests
# ------------------------------------------------------------------

def test_text_fields_length():
    """Verifies string-only character aggregation and exclusion filters."""

    analyzer = ComplexityAnalyzer()
    
    dialogue = Dialogue(
        must_use_expression="Say hello",  # len = 9
        more_info="Keep it simple",       # len = 14
        has_options=True                  # Boolean (should be bypassed in text length)
    )
    
    # Total character length check: 9 + 14 = 23 + type = "Dialogue" = 31
    assert analyzer._text_fields_length(dialogue) == 31

    # Total character length check with exclusion set (excluding type, which is always done inside the actual evaluation)
    assert analyzer._text_fields_length(dialogue, exclude={"type"}) == 23


def test_own_optional_fields_robust():
    """
    Verifies that optional fields are dynamically parsed from the schema.
    This test uses assertions on properties, keeping it completely immune to new schema additions.
    """

    analyzer = ComplexityAnalyzer()
    
    # Dialogue optional fields assertion
    dialogue_opts = analyzer._own_optional_fields(Dialogue)
    assert "must_use_expression" in dialogue_opts
    assert "more_info" in dialogue_opts
    assert "number_of_options" in dialogue_opts
    
    # Quest optional fields (excluding inherited fields from Dialogue)
    quest_opts = analyzer._own_optional_fields(Quest, exclude_inherited_from=Dialogue)
    assert "objective" not in quest_opts  # 'objective' is required, shouldn't be here
    assert "must_use_expression" not in quest_opts  # Should be excluded as it is inherited
    assert "name" in quest_opts
    assert "description" in quest_opts


def test_is_bool_field():
    """Verifies Pydantic field annotation parsing for boolean values (including Optional types)."""

    analyzer = ComplexityAnalyzer()
    
    assert analyzer._is_bool_field(bool) is True
    assert analyzer._is_bool_field(Optional[bool]) is True
    assert analyzer._is_bool_field(Union[bool, None]) is True
    
    assert analyzer._is_bool_field(str) is False
    assert analyzer._is_bool_field(Optional[str]) is False


# ------------------------------------------------------------------
# Intent Type Weighting Tests
# ------------------------------------------------------------------

def test_score_intent_behavioral():
    """Verifies that intent types retrieve weights from config mapping without hardcoding values."""

    analyzer = ComplexityAnalyzer()
    
    dialogue = Dialogue()
    quest = Quest(objective="Kill the beast")
    
    # Maps are dynamic, avoiding absolute number assertions
    assert analyzer._score_intent(dialogue) == analyzer._INTENT_COMPLEXITY.get("dialogue")
    assert analyzer._score_intent(quest) == analyzer._INTENT_COMPLEXITY.get("quest")
    
    # Fallback evaluation for unexpected/custom sub-classes
    class CustomIntent(Dialogue):
        pass
    assert analyzer._score_intent(CustomIntent()) == analyzer._DEFAULT_INTENT_COMPLEXITY


# ------------------------------------------------------------------
# Context Length Scoring & Debug Tests
# ------------------------------------------------------------------

def test_score_context_length_debug_and_saturation():
    """Verifies _score_context_length_debug attributes, calculation, and clamping behavior."""
    analyzer = ComplexityAnalyzer(
        min_tokens=10.0,
        max_tokens={"dialogue": 50.0, "quest": 100.0},
        chars_per_token=4.0
    )

    # 1. Under minimum tokens: score clamps to 0.0
    game_ctx_short = GameContext(epoch="E", environment="short", world_state="ok")
    npc_ctx_short = NPCContext(
        name="A", age=20, personality="P", context="C",
        intent=Dialogue(), talkativeness=Talkativeness.AVERAGE,
        main_character_relation="R"
    )
    # Total chars: len("E") + len("short") + len("ok") + len("A") + len("P") + len("C") + len("R") = 1+5+2+1+1+1+1 = 12
    # 12 / 4 = 3 tokens <= min_tokens (10.0) -> score 0.0
    debug_short = analyzer._score_context_length_debug(game_ctx_short, npc_ctx_short)
    assert isinstance(debug_short, _ContextLengthScore)
    assert debug_short.score == 0.0
    assert debug_short.intent_kind == "dialogue"
    assert debug_short.min_tokens == 10.0
    assert debug_short.max_tokens == 50.0
    assert analyzer._score_context_length(game_ctx_short, npc_ctx_short) == 0.0

    # 2. Over maximum tokens: score clamps to 1.0
    game_ctx_long = GameContext(epoch="E" * 200, environment="env" * 100, world_state="ws" * 100)
    npc_ctx_long = NPCContext(
        name="Name", age=30, personality="Personality", context="Context",
        intent=Dialogue(), talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Relation"
    )
    debug_long = analyzer._score_context_length_debug(game_ctx_long, npc_ctx_long)
    assert debug_long.score == 1.0
    assert debug_long.estimated_tokens > 50.0
    assert analyzer._score_context_length(game_ctx_long, npc_ctx_long) == 1.0

    # 3. Quest intent applies quest max_tokens and sets intent_kind to 'quest'
    npc_ctx_quest = NPCContext(
        name="A", age=20, personality="P", context="C",
        intent=Quest(objective="Do task"), talkativeness=Talkativeness.AVERAGE,
        main_character_relation="R"
    )
    debug_quest = analyzer._score_context_length_debug(game_ctx_short, npc_ctx_quest)
    assert debug_quest.intent_kind == "quest"
    assert debug_quest.max_tokens == 100.0


def test_context_length_score_frozen():
    """Verifies that _ContextLengthScore is a frozen dataclass and cannot be modified."""
    score_obj = _ContextLengthScore(
        score=0.5, total_chars=100, estimated_tokens=25.0,
        min_tokens=10.0, max_tokens=50.0, intent_kind="dialogue"
    )
    with pytest.raises((AttributeError, TypeError)):
        score_obj.score = 0.8


def test_max_tokens_for_intent():
    """Verifies that _max_tokens_for_intent retrieves the respective ceiling for Dialogue vs Quest."""
    analyzer = ComplexityAnalyzer(max_tokens={"dialogue": 250.0, "quest": 400.0})
    assert analyzer._max_tokens_for_intent(Dialogue()) == 250.0
    assert analyzer._max_tokens_for_intent(Quest(objective="Save village")) == 400.0


def test_text_fields_length_with_non_string_types_and_exclusions():
    """Verifies that non-string values (int, bool, Enum, None) are ignored and exclusions respected."""
    analyzer = ComplexityAnalyzer()
    game_ctx = GameContext(
        epoch="Modern", environment="City", world_state="Normal",
        main_character_description=None
    )
    # len("Modern") = 6, len("City") = 4, len("Normal") = 6 -> total = 16
    assert analyzer._text_fields_length(game_ctx) == 16
    # With exclusion of "environment": 6 + 6 = 12
    assert analyzer._text_fields_length(game_ctx, exclude={"environment"}) == 12


def test_score_richness_empty_fields():
    """Verifies that _score_richness safely returns 0.0 when no checkable fields exist."""
    analyzer = ComplexityAnalyzer()
    d = Dialogue()
    assert analyzer._score_richness(d, set()) == 0.0
    assert analyzer._score_richness(d, {"type"}) == 0.0


# ------------------------------------------------------------------
# Intent Richness Behavioral Tests
# ------------------------------------------------------------------

def test_score_intent_richness_behavioral():
    """
    Verifies structural richness calculations using algorithmic invariants:
    1. Zero-baseline: Empty optional parameters yield a score of 0.0.
    2. Monotonicity: Accumulating populated optional parameters increases the score.
    3. Weighted ratio: Boolean flags weigh flag_field_weight times a plain text field.
    4. Non-bool fields (e.g. number_of_options: int) weigh like text fields.
    """

    flag_weight = 2.0
    analyzer = ComplexityAnalyzer(flag_field_weight=flag_weight)

    # 1. Zero-baseline Invariant
    d_empty = Dialogue()
    assert analyzer._score_intent_richness(d_empty) == 0.0

    # 2. Monotonicity Invariant
    d_one_param = Dialogue(must_use_expression="Sarcastic")
    d_two_params = Dialogue(must_use_expression="Sarcastic", more_info="In a hurry")

    score_empty = analyzer._score_intent_richness(d_empty)
    score_low = analyzer._score_intent_richness(d_one_param)
    score_high = analyzer._score_intent_richness(d_two_params)

    assert score_empty == 0.0
    assert 0.0 < score_low < score_high <= 1.0

    # 3. Relative Weights Invariant (isolated from the real intent models)
    fields = {"text_field", "flag_field"}
    score_text = analyzer._score_richness(_RichnessProbe(text_field="x"), fields)    # 1 / (1 + 2)
    score_flag = analyzer._score_richness(_RichnessProbe(flag_field=True), fields)   # 2 / (1 + 2)

    assert score_flag == pytest.approx(score_text * flag_weight)

    # 4. Non-bool fields are NOT flags: number_of_options (int) weighs like a text field
    d_text_only = Dialogue(must_use_expression="Sarcastic")
    d_options_only = Dialogue(number_of_options=3)

    assert analyzer._score_intent_richness(d_options_only) == pytest.approx(
        analyzer._score_intent_richness(d_text_only)
    )


# ------------------------------------------------------------------
# End-to-End Integration Tests
# ------------------------------------------------------------------

def test_analyze_behavioral():
    """
    Evaluates the orchestrating analyze() function using robust behavioral invariants.
    Ensures mathematical mapping and categorization tier logic match our framework definitions.
    """
    
    analyzer = ComplexityAnalyzer(
        min_tokens=5,
        max_tokens={"dialogue": 30, "quest": 60},
        chars_per_token=3.0,
        flag_field_weight=2.0
    )

    game_ctx = GameContext(
        epoch="Ancient",
        environment="Gloomy Dungeon",
        world_state="Haunted"
    )

    dialogue = Dialogue(
        must_use_expression="Nervous",
        more_info="Wants to escape",
        number_of_options=3
    )

    npc_ctx = NPCContext(
        name="Bob",
        age=28,
        personality="Shy",
        context="Prison Cell",
        intent=dialogue,
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Neutral",
        recent_plot="Captured yesterday",
        visual_description="Wearing rags",
        backstory="A local peasant"
    )

    score: ComplexityScore = analyzer.analyze(game_ctx, npc_ctx)

    # Invariant 1: Score model outputs structure matches expectations
    assert isinstance(score, ComplexityScore)
    assert 0.0 <= score.value <= 1.0
    assert isinstance(score.tier, ComplexityTier)

    # Invariant 2: Mathematical correctness of weighted average (independent of weights values)
    expected_weighted_total = sum(
        score.breakdown[key] * analyzer._WEIGHTS[key] 
        for key in score.breakdown
    )
    assert score.value == round(expected_weighted_total, 3)

    # Invariant 3: Complexity tier matching criteria
    expected_tier = classify_score_to_complexity_tier(score.value)
    assert score.tier == expected_tier

    # Invariant 4: Completeness of breakdown details
    assert set(score.breakdown.keys()) == set(analyzer._WEIGHTS.keys())


def test_score_intent_richness_quest_behavioral():
    """
    Verifies that Quest intent richness accumulates both Dialogue and Quest-specific optional fields,
    and respects the flag_field_weight for boolean flags (e.g. has_choice).
    """
    analyzer = ComplexityAnalyzer(flag_field_weight=2.0)

    # 1. Quest with only required field (objective) -> 0.0 richness
    q_bare = Quest(objective="Find artifact")
    assert analyzer._score_intent_richness(q_bare) == 0.0

    # 2. Quest with Quest-specific text fields populated (name, location, reward)
    q_with_quest_fields = Quest(
        objective="Find artifact",
        name="The Ancient Relic",
        location="Lost Temple",
        reward="500 Gold"
    )
    score_quest_fields = analyzer._score_intent_richness(q_with_quest_fields)
    assert 0.0 < score_quest_fields < 1.0

    # 3. Quest with both Dialogue optional fields AND Quest optional fields
    q_rich = Quest(
        objective="Find artifact",
        must_use_expression="Be careful",
        more_info="Traps everywhere",
        number_of_options=3,
        name="The Ancient Relic",
        location="Lost Temple",
        reward="500 Gold",
        has_choice=True
    )
    score_rich = analyzer._score_intent_richness(q_rich)
    assert score_rich > score_quest_fields
    assert score_rich <= 1.0


def test_analyze_quest_integration():
    """Verifies analyze() execution when the request intent is a Quest."""
    analyzer = ComplexityAnalyzer(
        min_tokens=5,
        max_tokens={"dialogue": 30, "quest": 60},
        chars_per_token=3.0,
        flag_field_weight=2.0
    )

    game_ctx = GameContext(epoch="Fantasy", environment="Kingdom", world_state="War")
    quest = Quest(
        objective="Slay the dragon",
        name="Dragon Slayer",
        location="Dragon Peak",
        reward="Dragon Armor",
        has_choice=True
    )
    npc_ctx = NPCContext(
        name="Eldrin",
        age=60,
        personality="Wise",
        context="Throne Room",
        intent=quest,
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Ally"
    )

    score = analyzer.analyze(game_ctx, npc_ctx)

    assert isinstance(score, ComplexityScore)
    assert 0.0 <= score.value <= 1.0
    # Quest intent weight is 0.70 (compared to dialogue 0.30)
    assert score.breakdown["intent_weight"] == 0.70
    expected_weighted_total = sum(
        score.breakdown[key] * analyzer._WEIGHTS[key] for key in score.breakdown
    )
    assert score.value == round(expected_weighted_total, 3)
    assert score.tier == classify_score_to_complexity_tier(score.value)