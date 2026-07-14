import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from core.config.thresholds import CHARS_PER_TOKEN
from core.routing.models import ModelConfig
from core.llm.llm_base_client import BaseLLMClient
from core.llm.openai_client import OpenAICompatibleClient
from core.types.contract import Contract
from core.types.enums import ComplexityTier
from tools.errors import LLMClientError, LLMClientErrorCode

# Placeholder score mapping used by self-assessment and as a fallback.
_TIER_TO_SCORE = {
    ComplexityTier.LOW: 30.0,
    ComplexityTier.MEDIUM: 60.0,
    ComplexityTier.HIGH: 90.0,
}
_DEFAULT_FALLBACK_SCORE = _TIER_TO_SCORE[ComplexityTier.MEDIUM]

# Hard limit for profiling execution and its penalty fallback.
_TIMEOUT_PENALTY_SCORE = 5.0  # Minimal score assigned when a model times out during profiling

# Relative importance of each metric in the final score.
# TODO: tune these weights against real benchmark data once available.
_WEIGHT_COMPLETION_TIME = 0.45
_WEIGHT_TTFT = 0.35
_WEIGHT_THROUGHPUT = 0.20

# Normalization scales — TODO: tune based on real hardware/model data.
_COMPLETION_TIME_SCALE_SECONDS = 8.0   # ~8s -> score halved
_TTFT_SCALE_SECONDS = 2.0              # ~2s -> score halved
_THROUGHPUT_REFERENCE_TOKENS_PER_SEC = 40.0  # throughput at/above this -> full score

_TEST_PROMPT = "Reply with the single word: ready."


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
    """Profiler that trusts the developer-declared `intended_tier` as-is,
    without performing any real measurement. Used when profiler=False.
    """

    def profile(self, models: List[ModelConfig]) -> List[RankedModel]:
        return [
            RankedModel(config=model, score=_TIER_TO_SCORE[model.intended_tier], client=build_client(model))
            for model in models
        ]


class BenchmarkProfiler(BaseProfiler):
    """Automatic profiler used when profiler=True: measures real model
    performance and derives a power score from the results.

    Metrics, in order of importance: total completion time, time to
    first token, and throughput.
    """

    def profile(self, models: List[ModelConfig]) -> List[RankedModel]:
        """Ranks the given models by using the profiling metrics of the class."""
        ranked: List[RankedModel] = []
        for model in models:
            client = build_client(model)
            score = self._compute_score(client, model)
            ranked.append(RankedModel(config=model, score=score, client=client))
        return ranked

    def _compute_score(self, client: BaseLLMClient, model: ModelConfig) -> float:
        """Runs all measurements and combines them into a single weighted score.

        If a measurement fails outright (e.g. unreachable model), falls
        back to the declared tier's score, or a neutral default if no
        tier was declared either.
        """
        try:
            completion_time, ttft, throughput = self._measure_time_to_first_token_and_throughput(client)
        except TimeoutError:
            # Hard limit violation: return a severely penalized score directly, 
            # bypassing any declared intent since the model is unresponsive.
            return _TIMEOUT_PENALTY_SCORE
        except Exception:
            return _TIER_TO_SCORE[model.intended_tier] if model.intended_tier is not None else _DEFAULT_FALLBACK_SCORE

        completion_time_score = self._normalize_lower_is_better(completion_time, _COMPLETION_TIME_SCALE_SECONDS)
        ttft_score = self._normalize_lower_is_better(ttft, _TTFT_SCALE_SECONDS)
        throughput_score = self._normalize_higher_is_better(throughput, _THROUGHPUT_REFERENCE_TOKENS_PER_SEC)

        weighted_sum = (
            completion_time_score * _WEIGHT_COMPLETION_TIME
            + ttft_score * _WEIGHT_TTFT
            + throughput_score * _WEIGHT_THROUGHPUT
        )
        total_weight = _WEIGHT_COMPLETION_TIME + _WEIGHT_TTFT + _WEIGHT_THROUGHPUT

        return round(weighted_sum / total_weight, 2)

    def _measure_time_to_first_token_and_throughput(self, client: BaseLLMClient) -> tuple[float, float, float]:
        """Runs a single streaming probe call, measuring total completion
        time, time to first token, and throughput together — avoids
        running separate calls purely for profiling, which would waste
        time on top of the model's already-costly inference.

        Returns:
            (total_completion_time_seconds, time_to_first_token_seconds, throughput_tokens_per_sec)

        Raises:
            TimeoutError: If the underlying client library triggers a timeout during streaming.
            LLMClientError: If the model never returns any tokens.
        """
        contract = Contract(system_prompt="You are a test probe.", user_prompt=_TEST_PROMPT)

        start = time.perf_counter()
        first_token_time: Optional[float] = None
        full_text = ""
    
        try:
            for chunk in client.generate_streaming(contract, temperature=0.0):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += chunk
        except Exception as e:
            # Intercept native HTTP/SDK timeouts or network interruptions 
            # thrown during the streaming iteration.
            raise TimeoutError(f"Profiling stream interrupted or timed out natively: {e}") from e

        end = time.perf_counter()

        if first_token_time is None:
            raise LLMClientError(
                code=LLMClientErrorCode.EMPTY_RESPONSE,
                message="Model returned no tokens during profiling probe.",
            )

        total_completion_time = end - start
        time_to_first_token = first_token_time - start
        generation_time = max(end - first_token_time, 1e-6)  # avoid division by zero
        token_estimate = self._estimate_token_count(full_text)
        throughput = token_estimate / generation_time

        return total_completion_time, time_to_first_token, throughput

    @staticmethod
    def _estimate_token_count(text: str) -> float:
        """Rough token count approximation."""
        return float(len(text)) / CHARS_PER_TOKEN

    @staticmethod
    def _normalize_lower_is_better(value: float, scale: float) -> float:
        """Maps a 'lower is better' raw measurement to a 0-100 score."""
        return max(0.0, min(100.0, 100.0 * scale / (scale + value)))

    @staticmethod
    def _normalize_higher_is_better(value: float, reference: float) -> float:
        """Maps a 'higher is better' raw measurement to a 0-100 score."""
        return max(0.0, min(100.0, 100.0 * value / reference))


def build_client(model: ModelConfig) -> BaseLLMClient:
    """Builds the appropriate BaseLLMClient for a given model configuration."""
    return OpenAICompatibleClient(endpoint=model.endpoint, api_key=model.api_key, model_identifier=model.id)