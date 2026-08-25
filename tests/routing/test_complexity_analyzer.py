import pytest
from unittest.mock import MagicMock
from typing import Optional, Union
from core.routing.complexity_analyzer import (
    ComplexityAnalyzer,
    ComplexityScore,
    _COMMON_LENGTH_THRESHOLD_NAMES,
    _QUEST_ONLY_LENGTH_THRESHOLD_NAMES,
)
from core.types.contexts import Dialogue, Quest, GameContext, NPCContext, Talkativeness
from core.types.enums import ComplexityTier
from core.routing.helpers import classify_score_to_complexity_tier


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
    
    # Total character length check: 9 + 14 = 23
    assert analyzer._text_fields_length(dialogue) == 23

    # Total character length check with exclusion set
    assert analyzer._text_fields_length(dialogue, exclude={"more_info"}) == 9


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
    assert "has_options" in dialogue_opts
    
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
# Manual Flag / Relation Override Tests
# ------------------------------------------------------------------

def test_score_manual_flag_behavioral():
    """Verifies manual override scaling for high-importance narrative relationships."""

    analyzer = ComplexityAnalyzer()
    
    def create_npc_with_relation(relation: str):
        return NPCContext(
            name="Grog",
            age=35,
            personality="Angry",
            context="The Wilderness",
            intent=Dialogue(),
            talkativeness=Talkativeness.AVERAGE,
            main_character_relation=relation
        )

    # Important/critical narrative connections must force 1.0 (highest tier priority)
    for critical_relation in ["hostile", "romantic interest", "nemesis"]:
        assert analyzer._score_manual_flag(create_npc_with_relation(critical_relation)) == 1.0
        assert analyzer._score_manual_flag(create_npc_with_relation(critical_relation.upper())) == 1.0  # Case insensitivity

    # Standard/common connections must remain neutral at 0.0
    for standard_relation in ["friend", "neutral", "acquaintance"]:
        assert analyzer._score_manual_flag(create_npc_with_relation(standard_relation)) == 0.0


# ------------------------------------------------------------------
# Intent Richness Behavioral Tests
# ------------------------------------------------------------------

def test_score_intent_richness_behavioral():
    """
    Verifies structural richness calculations using algorithmic invariants:
    1. Zero-baseline: Empty optional parameters yield a score of 0.0.
    2. Monotonicity: Accumulating populated optional parameters increases the score.
    3. Weighted ratio: Boolean generation flags weigh strictly more than plain text inputs.
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

    # 3. Relative Weights Invariant
    # A single bool flag should yield exactly flag_field_weight * score of a single text field
    d_text_only = Dialogue(must_use_expression="Sarcastic")  # weight = 1.0
    d_flag_only = Dialogue(has_options=True)                 # weight = flag_field_weight (2.0)

    score_text = analyzer._score_intent_richness(d_text_only)
    score_flag = analyzer._score_intent_richness(d_flag_only)

    assert pytest.approx(score_flag) == score_text * flag_weight


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
        has_options=True
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