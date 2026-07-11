# TODO: IMPLEMENT THE ACTUAL PROFILING. This one right here is only a placeholder to make the code structure work. We still have to discuss the profiling metrics.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from core.routing.models import ModelConfig, IntendedTier
from core.llm.llm_base_client import BaseLLMClient
from core.llm.openai_client import OpenAICompatibleClient

# Placeholder score mapping used by self-assessment and as a fallback.
_TIER_TO_SCORE = {
    IntendedTier.LOW: 30.0,
    IntendedTier.MEDIUM: 60.0,
    IntendedTier.HIGH: 90.0,
}

# Used only as a last-resort fallback when a score is needed but no tier
# is available at all.
_DEFAULT_FALLBACK_SCORE = _TIER_TO_SCORE[IntendedTier.MEDIUM]


@dataclass
class RankedModel:
    """A model configuration paired with its computed power score and
    a ready-to-use client instance."""

    config: ModelConfig
    score: float
    client: BaseLLMClient


class BaseProfiler(ABC):
    """Abstract interface for model profiling strategies."""

    @abstractmethod
    def profile(self, models: List[ModelConfig]) -> List[RankedModel]:
        """Assigns a power score to each model and builds its client."""
        ...


class SelfAssessmentProfiler(BaseProfiler):
    """
    Profiler that trusts the developer-declared `intended_tier` as-is,
    without performing any real measurement. Used when profiler=False.
    """

    def profile(self, models: List[ModelConfig]) -> List[RankedModel]:
        return [
            RankedModel(
                config=model,
                score=_TIER_TO_SCORE[model.intended_tier],
                client=_build_client(model),
            )
            for model in models
        ]


class BenchmarkProfiler(BaseProfiler):
    """
    Automatic profiler used when profiler=True: measures real model
    performance and derives a power score from the results, instead of
    relying on a developer-declared tier.

    NOT YET IMPLEMENTED.
    """

    _TEST_PROMPT = "Reply with the single word: ready."

    def profile(self, models: List[ModelConfig]) -> List[RankedModel]:
        """
        Builds a client for each model and computes its benchmark score.
        """
        ranked: List[RankedModel] = []
        for model in models:
            client = _build_client(model)
            score = self._compute_score(client, model)
            ranked.append(RankedModel(config=model, score=score, client=client))
        return ranked

    def _compute_score(self, client: BaseLLMClient, model: ModelConfig) -> float:
        """
        Aggregates the individual benchmark measurements into a single
        power score.
        """
        raise NotImplementedError(
            "BenchmarkProfiler._compute_score is not implemented yet. "
            "Implement _measure_time_to_first_token, _measure_throughput, "
            "and _measure_output_quality, then combine them here."
        )

    def _measure_time_to_first_token(self, client: BaseLLMClient) -> float:
        """
        Measures how long the model takes to start responding.
        """
        raise NotImplementedError("Time-to-first-token measurement is not implemented yet.")

    def _measure_throughput(self, client: BaseLLMClient) -> float:
        """
        Measures sustained generation throughput.
        """
        raise NotImplementedError("Throughput measurement is not implemented yet.")

    def _measure_output_quality(self, client: BaseLLMClient) -> float:
        """
        Measures how well the model follows instructions/schema on a
        representative test prompt (e.g. checking the response actually
        matches the expected JSON schema, correct format, etc).
        """
        raise NotImplementedError("Output quality measurement is not implemented yet.")


def _build_client(model: ModelConfig) -> BaseLLMClient:
    """Builds the appropriate BaseLLMClient for a given model configuration."""
    return OpenAICompatibleClient(
        endpoint=model.endpoint,
        api_key=model.api_key,
        model_identifier=model.id,
    )