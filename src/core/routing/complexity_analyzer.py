import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import get_args, get_origin, Union
from pydantic import BaseModel
from core.config.thresholds import CHARS_PER_TOKEN, HIGH_THRESHOLD, LOW_THRESHOLD
from core.routing.helpers import classify_score_to_complexity_tier
from core.types.contexts import GameContext, NPCContext, Quest, Dialogue
from core.types.enums import ComplexityTier
from core.logger import to_json_format

logger = logging.getLogger(__name__)

# They are divided into two groups because the Intent is polymorphic (Dialogue or Quest): 
# Dialogue Intent
_COMMON_LENGTH_THRESHOLD_NAMES = {
    # GameContext
    "game_environment": "MAX_ENVIRONMENT_LENGTH",
    "game_world_state": "MAX_WORLD_STATE_LENGTH",
    "game_main_character_description": "MAX_MAIN_CHARACTER_DESCRIPTION_LENGTH",
    # NPCContext (excluding intent and language, as in _score_context_length)
    "npc_name": "MAX_NAME_LENGTH",
    "npc_personality": "MAX_PERSONALITY_LENGTH",
    "npc_context": "MAX_CONTEXT_LENGTH",
    "npc_relation": "MAX_RELATION_LENGTH",
    "npc_recent_plot": "MAX_RECENT_PLOT_LENGTH",
    "npc_visual_description": "MAX_VISUAL_DESCRIPTION_LENGTH",
    "npc_backstory": "MAX_BACKSTORY_LENGTH",
    # Dialogue (also common to Quest, which inherits from Dialogue)
    "dialogue_must_use_expression": "MAX_MUST_USE_EXPRESSION_LENGTH",
    "dialogue_more_info": "MAX_MORE_INFO_LENGTH",
}

# Additional fields present ONLY on Quest.
_QUEST_ONLY_LENGTH_THRESHOLD_NAMES = {
    "quest_name": "MAX_QUEST_NAME_LENGTH",
    "quest_objective": "MAX_QUEST_OBJECTIVE_LENGTH",
    "quest_description": "MAX_QUEST_DESCRIPTION_LENGTH",
    "quest_location": "MAX_QUEST_LOCATION_LENGTH",
    "quest_reward": "MAX_QUEST_REWARD_LENGTH",
}


@dataclass
class ComplexityScore:
    """Result of the complexity analysis, with a breakdown for debugging/logging."""

    value: float  # normalized score, 0.0 - 1.0
    tier: ComplexityTier
    breakdown: dict[str, float] = field(default_factory=dict)   # contribution of each feature


class ComplexityAnalyzer:
    """
    Estimates how complex a dialogue generation request is, based on GameContext and NPCContext
    """

    # Weights of individual features in the final score (must sum to 1.0)
    _WEIGHTS = {
        "context_length": 0.25,
        "intent_richness": 0.25,
        "intent_weight": 0.45,
        "manual_flag": 0.05,
    }

    # Known intents, with an intrinsic complexity weight (0.0 - 1.0, must sum to 1.0)
    _INTENT_COMPLEXITY = {
        "dialogue": 0.3,
        "quest": 0.7,
    }

    _DEFAULT_INTENT_COMPLEXITY = 0.5   # fallback for unmapped intents

    def __init__(
        self,
        min_tokens: float = 150,
        max_tokens: float | dict[str, float] = 800,
        chars_per_token: float = 4.0,
        flag_field_weight: float = 2.0, #TODO: should be adapted to the request options number
    ):
        """
        Args:
            min_tokens: below this estimated token threshold, context_length score = 0.0.
            max_tokens: above this estimated token threshold, context_length score = 1.0.
                Can be:
                - a single float, used for both intent types ("legacy" behavior, 
                  useful if you don't want to differentiate between dialogue/quest);
                - a dict {"dialogue": ..., "quest": ...} with a different ceiling for 
                  each Intent type.
            chars_per_token: conversion factor from characters -> estimated tokens.
            flag_field_weight: relative weight of optional boolean fields (e.g., has_options, 
                has_choice) compared to content fields (e.g., more_info, location) in 
                the intent richness calculation.
        """
        
        if isinstance(max_tokens, dict):
            missing = {"dialogue", "quest"} - set(max_tokens)
            if missing:
                raise ValueError(
                    f"max_tokens dict must contain the keys 'dialogue' and 'quest' "
                    f"(missing: {sorted(missing)})"
                )
            max_tokens_by_intent = dict(max_tokens)
        else:
            max_tokens_by_intent = {"dialogue": max_tokens, "quest": max_tokens}

        for intent_key, value in max_tokens_by_intent.items():
            if value <= min_tokens:
                raise ValueError(
                    f"max_tokens['{intent_key}'] ({value}) must be greater "
                    f"than min_tokens ({min_tokens})"
                )
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")

        self._min_tokens = min_tokens
        self._max_tokens_by_intent = max_tokens_by_intent
        self._chars_per_token = chars_per_token
        self._flag_field_weight = flag_field_weight

    @classmethod
    def from_thresholds(
        cls,
        thresholds: ModuleType,
        realistic_max_ratio: float = 0.15,
        min_tokens: float = 0.0,
        chars_per_token: float = CHARS_PER_TOKEN,
        flag_field_weight: float = 2.0,
    ) -> "ComplexityAnalyzer":
        """
        Derives max_tokens from validation constants in thresholds.py instead of hardcoding them.

        Process:
        1. Sums common text fields (GameContext + NPCContext + Dialogue) for the Dialogue worst-case.
        2. Adds Quest-specific fields to that total for the Quest worst-case.
        3. Since intents are mutually exclusive, separate max_tokens are stored for "dialogue" 
           and "quest" to avoid overestimating pure dialogue complexity.
        4. Applies `realistic_max_ratio` to the worst-case token count because maximizing all 
           fields simultaneously is statistically unrealistic.

        Args:
            thresholds: The thresholds.py module containing the MAX_*_LENGTH constants.
            realistic_max_ratio: Fraction of the theoretical worst-case considered "maximum 
                practical complexity" (e.g., 0.15). Beyond this, the score saturates to 1.0.
            min_tokens: Absolute token threshold below which the complexity score is 0.0.
            chars_per_token: Conversion factor from characters to estimated tokens (default: 4.0).
            flag_field_weight: Relative weight of boolean flags vs string content fields.
        
        """
        common_chars = sum(
            getattr(thresholds, const_name)
            for const_name in _COMMON_LENGTH_THRESHOLD_NAMES.values()
        )
        quest_extra_chars = sum(
            getattr(thresholds, const_name)
            for const_name in _QUEST_ONLY_LENGTH_THRESHOLD_NAMES.values()
        )

        dialogue_worst_case_tokens = common_chars / chars_per_token
        quest_worst_case_tokens = (common_chars + quest_extra_chars) / chars_per_token

        max_tokens = {
            "dialogue": dialogue_worst_case_tokens * realistic_max_ratio,
            "quest": quest_worst_case_tokens * realistic_max_ratio,
        }

        return cls(
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            chars_per_token=chars_per_token,
            flag_field_weight=flag_field_weight,
        )

    def analyze(self, game_context: GameContext, npc_context: NPCContext) -> ComplexityScore:
        """
        Calculates the overall complexity score and assigns a routing tier.

        Computes a weighted average of four features:
        - text_content_length: Total rough token estimate of all text fields.
        - intent_richness: The density and structure of populated optional intent fields.
        - intrinsic_intent_complexity: Base baseline difficulty of the intent type (e.g., Quest vs Dialogue, Quest must have higher complexity).
        - manual_narrative_importance_flag: Override based on critical NPC relationships.
        """

        breakdown = {
            "context_length": self._score_context_length(game_context, npc_context),
            "intent_richness": self._score_intent_richness(npc_context.intent),
            "intent_weight": self._score_intent(npc_context.intent),
            "manual_flag": self._score_manual_flag(npc_context),
        }


        # Adds weights (fixed) to the scores calculated in the breakdown
        total = sum(breakdown[key] * self._WEIGHTS[key] for key in breakdown)

        # Classifies the value to a tier label
        tier = classify_score_to_complexity_tier(total)
        
        logger.debug(f"Complexity analysis breakdown:\n{to_json_format(breakdown)}\n-> Total: {total:.3f}, Tier: {tier}")

        return ComplexityScore(value=round(total, 3), tier=tier, breakdown=breakdown)

    # ------------------------------------------------------------------
    # Context length scoring
    # ------------------------------------------------------------------

    def _score_context_length(self, game_context: GameContext, npc_context: NPCContext) -> float:
        """Longer world/NPC/intent text content → higher complexity. Rough token estimate."""

        normalized, _debug = self._score_context_length_debug(game_context, npc_context)
        return normalized


    def _max_tokens_for_intent(self, intent: Quest | Dialogue) -> float:
        """
        Returns the correct max_tokens based on the concrete type of Intent.
        Quest inherits from Dialogue and has additional fields, so its theoretical 
        worst-case scenario (and consequently its max_tokens) is higher: using the 
        same max_tokens for both would underestimate the complexity of a rich Quest 
        or overestimate that of a "simple" Dialogue.
        """
        key = "quest" if isinstance(intent, Quest) else "dialogue"
        return self._max_tokens_by_intent[key]

    def _text_fields_length(self, model, exclude: set[str] = frozenset()) -> int:
        """
        Sums the length of all top-level string fields on a Pydantic model,
        skipping excluded field names and non-string values (int, bool, enum, None, etc.).
        """
        data = model.model_dump()
        total = 0
        for name, value in data.items():
            if name in exclude:
                continue
            if isinstance(value, str):
                total += len(value)
        return total

    # ------------------------------------------------------------------
    # Intent richness scoring
    # ------------------------------------------------------------------

    def _score_intent_richness(self, intent: Quest | Dialogue) -> float:
        """
        More populated optional fields / more generation flags → higher complexity.
        Quest inherits from Dialogue: the score always starts from the Dialogue 
        component (common to both) and, if the intent is also a Quest, it is 
        extended with the additional component specific to Quest.

        Self-adaptive: optional fields are read dynamically from the Pydantic 
        model, so adding new optional fields to Dialogue/Quest is automatically 
        picked up without modifying this method. Boolean fields (generation flags, 
        e.g., has_options, has_choice) weigh more than content fields (strings).
        """

        dialogue_fields = self._own_optional_fields(Dialogue)
        score = self._score_richness(intent, dialogue_fields)

        if isinstance(intent, Quest):
            quest_fields = self._own_optional_fields(Quest, exclude_inherited_from=Dialogue)
            score += self._score_richness(intent, quest_fields)

        return min(score, 1.0)

    def _own_optional_fields(self, model_cls: type[BaseModel], exclude_inherited_from: type[BaseModel] | None = None) -> set[str]:
        """
        Names of optional fields (not required) declared on model_cls, 
        optionally excluding those inherited from a base class.
        """
        names = {
            name for name, info in model_cls.model_fields.items()
            if not info.is_required()
        }
        if exclude_inherited_from is not None:
            names -= set(exclude_inherited_from.model_fields.keys())
        return names

    def _score_richness(self, model: BaseModel, field_names: set[str]) -> float:
        """
        Fraction of "achievable complexity" given by populated optional fields.
        Boolean fields (generation flags, e.g., has_options, has_choice) weigh 
        flag_field_weight times a normal content field.
        """
        model_cls = type(model)
        total_weight = 0.0
        achieved = 0.0
        for name in field_names:
            info = model_cls.model_fields[name]
            weight = self._flag_field_weight if self._is_bool_field(info.annotation) else 1.0
            total_weight += weight
            if getattr(model, name):
                achieved += weight
        return achieved / total_weight if total_weight else 0.0

    @staticmethod
    def _is_bool_field(annotation) -> bool:
        """True for `bool` or `Optional[bool]` (Union[bool, None])."""
        if annotation is bool:
            return True
        if get_origin(annotation) is Union:
            return bool in get_args(annotation)
        return False

    # ------------------------------------------------------------------
    # Intent type weighting
    # ------------------------------------------------------------------

    def _score_intent(self, intent: Quest | Dialogue) -> float:
        intent_name = type(intent).__name__.lower()
        return self._INTENT_COMPLEXITY.get(intent_name, self._DEFAULT_INTENT_COMPLEXITY)

    # ------------------------------------------------------------------
    # Manual override flag
    # ------------------------------------------------------------------

    def _score_manual_flag(self, npc_context: NPCContext) -> float:
        """
        Allows explicit override for narratively important NPCs/relations,
        e.g. a hostile or romantic relation might warrant a more capable model.
        """
        important_relations = {"hostile", "romantic interest", "nemesis"} #TODO: obtain this list from input or saved words
        return 1.0 if npc_context.main_character_relation.lower() in important_relations else 0.0
    
    # ------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------

    def _score_context_length_debug(
    self, game_context: GameContext, npc_context: NPCContext) -> tuple[float, dict]:
        """
        Like _score_context_length, but also returns intermediate values 
        (total_chars, estimated_tokens, min_tokens, max_tokens, intent_type) to 
        understand why a score results in 0.0 or 1.0 (saturation).
        """
        total_chars = (
            self._text_fields_length(game_context)
            + self._text_fields_length(npc_context, exclude={"intent", "language"})
            + self._text_fields_length(npc_context.intent)
        )
        estimated_tokens = total_chars / self._chars_per_token

        max_tokens = self._max_tokens_for_intent(npc_context.intent)
        token_range = max_tokens - self._min_tokens
        normalized = (estimated_tokens - self._min_tokens) / token_range
        normalized = min(max(normalized, 0.0), 1.0)

        debug_info = {
            "total_chars": total_chars,
            "estimated_tokens": round(estimated_tokens, 1),
            "min_tokens": round(self._min_tokens, 1),
            "max_tokens": round(max_tokens, 1),
            "intent_type": "quest" if isinstance(npc_context.intent, Quest) else "dialogue",
        }
        return normalized, debug_info