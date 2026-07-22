import logging
from typing import List, Optional
from core.routing.models import ModelConfig
from core.routing.profiler import BaseProfiler, SelfAssessmentProfiler, BenchmarkProfiler, RankedModel, build_client, _TIER_TO_SCORE
from core.types.enums import ComplexityTier
from tools.errors import RoutingConfigError, RoutingConfigErrorCode
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Holds the profiled, ranked list of models available for routing.

    Implemented as a singleton: there is only ever one registry of
    registered models at runtime, so multiple independent instances
    would only be a source of confusion (which one is "the" registry?).
    Use `ModelRegistry()` anywhere — it always returns the same instance.
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ranked_models = []
        return cls._instance

    def set_models(self, models: List[ModelConfig], profiler: bool = True) -> None:
        """Profiles and registers a new set of models, replacing any previous ones.

        Args:
            models: The model configurations to register.
            profiler: If True (default), models are automatically profiled
                via BenchmarkProfiler. If False, `intended_tier` is trusted
                as-is — in which case every model MUST declare it (see
                Raises below). Ignored entirely when only one model is
                provided.

        Raises:
            RoutingConfigError: If profiler=False and one or more models
                are missing `intended_tier`. Self-assessment has no other
                signal to rely on, so this is treated as a misconfiguration
                rather than silently defaulting to a guessed tier.
        """
        if len(models) == 1:
            self._ranked_models = [self._register_without_profiling(models[0])]
            return

        if not profiler:
            self._validate_tiers_declared(models)

        strategy: BaseProfiler = BenchmarkProfiler() if profiler else SelfAssessmentProfiler()
        ranked = strategy.profile(models)
        self._ranked_models = sorted(ranked, key=lambda r: r.score, reverse=True)
        self._print_ranked_models()

    def _print_ranked_models(self) -> None:
        """Prints the currently ranked models as a formatted table.

        Displays rank, id, computed score, declared/effective tier, endpoint,
        and max context tokens for each registered model.
        """
        if not self._ranked_models:
            logger.info("No models registered.")
            return

        headers = ["#", "ID", "Score", "Tier", "Endpoint", "Max Ctx"]

        rows = []
        for i, ranked in enumerate(self._ranked_models, start=1):
            cfg = ranked.config
            tier = cfg.intended_tier.value if cfg.intended_tier else "-"
            max_ctx = f"{cfg.max_context_tokens:,}" if cfg.max_context_tokens else "-"
            rows.append([
                str(i),
                cfg.id,
                f"{ranked.score:.3f}",
                tier,
                cfg.endpoint,
                max_ctx,
            ])

        col_widths = [
            max(len(headers[c]), max((len(row[c]) for row in rows), default=0))
            for c in range(len(headers))
        ]

        def format_row(values: List[str]) -> str:
            return "  ".join(v.ljust(col_widths[i]) for i, v in enumerate(values))

        separator = "-" * (sum(col_widths) + 2 * (len(headers) - 1))

        output_lines = [
            f"Ranked models ({len(rows)} total):",
            separator,
            format_row(headers),
            separator
        ]
        for row in rows:
            output_lines.append(format_row(row))
        output_lines.append(separator)
        logger.info("\n".join(output_lines))

    def get_ranked_models(self) -> List[RankedModel]:
        return self._ranked_models

    def get_best_for_tier(self, tier: ComplexityTier) -> Optional[RankedModel]:
        """Returns the highest-scoring model whose tier matches `tier`.
        Models with no declared tier are treated as LOW for this comparison."""
        candidates = [
            r for r in self._ranked_models
            if r.config.intended_tier == tier
        ]

        return candidates[0] if candidates else None

    @staticmethod
    def _validate_tiers_declared(models: List[ModelConfig]) -> None:
        """Ensures every model declares `intended_tier` when self-assessment
        (profiler=False) is requested, since it's the only signal available."""
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
        """Registers a single model with zero profiling overhead. Falls
        back to a neutral score if `intended_tier` was not declared."""
        score = _TIER_TO_SCORE[model.intended_tier] if model.intended_tier is not None else _TIER_TO_SCORE[ComplexityTier.MEDIUM]
        return RankedModel(config=model, score=score, client=build_client(model))