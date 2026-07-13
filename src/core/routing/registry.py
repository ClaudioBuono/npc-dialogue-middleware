from typing import List, Optional
from core.routing.models import ModelConfig, IntendedTier
from core.routing.profiler import BaseProfiler, SelfAssessmentProfiler, BenchmarkProfiler, RankedModel, _build_client, _TIER_TO_SCORE
from tools.errors import RoutingConfigError, RoutingConfigErrorCode


class ModelRegistry:
    """Holds the profiled, ranked list of models available for routing."""

    _ranked_models: List[RankedModel] = []

    @classmethod
    def set_models(cls, models: List[ModelConfig], profiler: bool = True) -> None:
        """Profiles and registers a new set of models, replacing any previous ones.

        Args:
            models: The model configurations to register.
            profiler: If True (default), models are automatically profiled
                via BenchmarkProfiler. If False,
                `intended_tier` is trusted as-is — in which case every
                model MUST declare it (see Raises below). Ignored
                entirely when only one model is provided.

        Raises:
            RoutingConfigError: If profiler=False and one or more models
                are missing `intended_tier`. Self-assessment has no other
                signal to rely on, so this is treated as a misconfiguration
                rather than silently defaulting to a guessed tier.
        """
        if len(models) == 1:
            cls._ranked_models = [cls._register_without_profiling(models[0])]
            return

        if not profiler:
            cls._validate_tiers_declared(models)

        strategy: BaseProfiler = BenchmarkProfiler() if profiler else SelfAssessmentProfiler()
        ranked = strategy.profile(models)
        cls._ranked_models = sorted(ranked, key=lambda r: r.score, reverse=True)

    @classmethod
    def get_ranked_models(cls) -> List[RankedModel]:
        return cls._ranked_models

    @classmethod
    def get_best_for_tier(cls, min_tier: IntendedTier) -> Optional[RankedModel]:
        """Returns the highest-scoring model whose declared tier is at
        least `min_tier`. Models with no declared tier are treated as
        LOW for this comparison."""
        tier_order = {IntendedTier.LOW: 0, IntendedTier.MEDIUM: 1, IntendedTier.HIGH: 2}
        candidates = [
            r for r in cls._ranked_models
            if tier_order[r.config.intended_tier or IntendedTier.LOW] >= tier_order[min_tier]
        ]
        if candidates:
            return candidates[0]

        return cls._ranked_models[0] if cls._ranked_models else None

    @staticmethod
    def _validate_tiers_declared(models: List[ModelConfig]) -> None:
        """
        Ensures every model declares `intended_tier` when self-assessment
        (profiler=False) is requested, since it's the only signal available.
        """
        missing = [m.id for m in models if m.intended_tier is None]
        if missing:
            raise RoutingConfigError(
                code=RoutingConfigErrorCode.MISSING_INTENDED_TIER,
                errors=[
                    f"Model '{model_id}' is missing 'intended_tier', required when profiler=False."
                    for model_id in missing
                ],
            )

    @staticmethod
    def _register_without_profiling(model: ModelConfig) -> RankedModel:
        """
        Registers a single model with zero profiling overhead. Falls
        back to a neutral score if `intended_tier` was not declared.
        """
        score = _TIER_TO_SCORE[model.intended_tier] if model.intended_tier is not None else _TIER_TO_SCORE[IntendedTier.MEDIUM]
        return RankedModel(config=model, score=score, client=_build_client(model))
    