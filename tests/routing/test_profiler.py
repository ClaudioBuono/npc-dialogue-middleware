import pytest
from unittest.mock import MagicMock, patch
from core.routing.models import ModelConfig
from core.routing.profiler import (
    SelfAssessmentProfiler,
    BenchmarkProfiler,
    build_client,
    _TIER_TO_SCORE,
    _DEFAULT_FALLBACK_SCORE,
    _TIMEOUT_PENALTY_SCORE,
)
from core.types.enums import ComplexityTier
from core.llm.openai_client import OpenAICompatibleClient
from tools.errors import LLMClientError


# ------------------------------------------------------------------
# SelfAssessmentProfiler Tests
# ------------------------------------------------------------------

def test_self_assessment_profiler():
    """Verifies that static self-assessment correctly assigns scores based on intended tiers."""

    profiler = SelfAssessmentProfiler()
    models = [
        ModelConfig(id="m1", endpoint="http://ep1", intended_tier=ComplexityTier.LOW),
        ModelConfig(id="m2", endpoint="http://ep2", intended_tier=ComplexityTier.HIGH)
    ]
    
    ranked = profiler.profile(models)
    
    assert len(ranked) == 2
    assert ranked[0].config.id == "m1"
    assert ranked[0].score == _TIER_TO_SCORE[ComplexityTier.LOW]
    
    assert ranked[1].config.id == "m2"
    assert ranked[1].score == _TIER_TO_SCORE[ComplexityTier.HIGH]


# ------------------------------------------------------------------
# BenchmarkProfiler - Math Helpers Tests
# ------------------------------------------------------------------

def test_benchmark_profiler_helpers():
    """Verifies mathematical invariants for token estimation and normalization formulas."""

    # Test _estimate_token_count (Assuming default chars_per_token=4.0)
    assert BenchmarkProfiler._estimate_token_count("abcd") == 1.0
    assert BenchmarkProfiler._estimate_token_count("") == 0.0

    # Test _normalize_lower_is_better
    # Formula invariant: max(0.0, min(1.0, 1.0 * scale / (scale + value)))
    assert BenchmarkProfiler._normalize_lower_is_better(0.0, 2.0) == 1.0 # Best case
    assert BenchmarkProfiler._normalize_lower_is_better(2.0, 2.0) == 0.5 # Mid case
    assert 0.0 <= BenchmarkProfiler._normalize_lower_is_better(100.0, 2.0) < 0.1 # Worst case

    # Test _normalize_higher_is_better
    # Formula invariant: max(0.0, min(1.0, 1.0 * value / reference))
    assert BenchmarkProfiler._normalize_higher_is_better(0.0, 40.0) == 0.0   # Worst case
    assert BenchmarkProfiler._normalize_higher_is_better(20.0, 40.0) == 0.5  # Mid case
    assert BenchmarkProfiler._normalize_higher_is_better(80.0, 40.0) == 1.0  # Best case (capped)


# ------------------------------------------------------------------
# BenchmarkProfiler - Measurement Core Tests
# ------------------------------------------------------------------

def test_measure_time_to_first_token_and_throughput_success():
    """Verifies successful stream processing and correct time/throughput calculations."""

    profiler = BenchmarkProfiler()
    mock_client = MagicMock()
    mock_client.generate_streaming.return_value = iter(["abcd", "efgh"]) # 8 chars = 2.0 tokens

    # Simulate: Start (10.0) -> First Token (11.0) -> End (12.0)
    with patch("time.perf_counter", side_effect=[10.0, 11.0, 12.0]):
        total_time, ttft, throughput = profiler._measure_time_to_first_token_and_throughput(mock_client)

    assert total_time == 2.0 # 12.0 - 10.0
    assert ttft == 1.0       # 11.0 - 10.0
    # generation time = 1.0s, tokens = 2.0 -> throughput = 2.0
    assert throughput == 2.0


def test_measure_time_to_first_token_and_throughput_empty_response():
    """Verifies that an empty stream yields a specific LLMClientError."""

    profiler = BenchmarkProfiler()
    mock_client = MagicMock()
    mock_client.generate_streaming.return_value = iter([])

    with patch("time.perf_counter", return_value=10.0):
        with pytest.raises(LLMClientError, match="Model returned no tokens during profiling probe"):
            profiler._measure_time_to_first_token_and_throughput(mock_client)


def test_measure_time_to_first_token_and_throughput_native_timeout():
    """Verifies handling of stream interruption/timeouts."""

    profiler = BenchmarkProfiler()
    mock_client = MagicMock()

    def generator_raising_exception():
        yield "chunk1"
        raise TimeoutError("Network Timeout") # Changed to TimeoutError to match standard behavior

    mock_client.generate_streaming.return_value = generator_raising_exception()

    with patch("time.perf_counter", side_effect=[10.0, 11.0]):
        with pytest.raises(TimeoutError, match="Profiling stream interrupted or timed out natively"):
            profiler._measure_time_to_first_token_and_throughput(mock_client)


# ------------------------------------------------------------------
# BenchmarkProfiler - Scoring Tests
# ------------------------------------------------------------------

def test_compute_score_success():
    """Verifies that raw metrics are correctly weighted into a final score."""

    profiler = BenchmarkProfiler()
    mock_client = MagicMock()
    model_cfg = ModelConfig(id="m1", endpoint="http://ep1")

    # Mock measurement to return exact scaling values (score 0.5, 0.5, 1.0)
    # Assuming weights: completion_time=0.45, ttft=0.35, throughput=0.20
    # (0.5 * 0.45) + (0.5 * 0.35) + (1.0 * 0.20) = 0.60
    with patch.object(profiler, "_measure_time_to_first_token_and_throughput", return_value=(8.0, 2.0, 40.0)):
        score = profiler._compute_score(mock_client, model_cfg)
    
    assert score == 0.60


def test_compute_score_timeout():
    """Verifies that a TimeoutError during measurement assigns the penalty score."""

    profiler = BenchmarkProfiler()
    mock_client = MagicMock()
    model_cfg = ModelConfig(id="m1", endpoint="http://ep1")

    with patch.object(profiler, "_measure_time_to_first_token_and_throughput", side_effect=TimeoutError("Timed out")):
        score = profiler._compute_score(mock_client, model_cfg)
    
    assert score == _TIMEOUT_PENALTY_SCORE


def test_compute_score_generic_error():
    """Verifies fallback behaviors when measurement fails with a generic error."""

    profiler = BenchmarkProfiler()
    mock_client = MagicMock()
    
    # 1. Fallback to intended_tier if available
    model_cfg_with_tier = ModelConfig(id="m1", endpoint="http://ep1", intended_tier=ComplexityTier.HIGH)
    with patch.object(profiler, "_measure_time_to_first_token_and_throughput", side_effect=RuntimeError("Some error")):
        score = profiler._compute_score(mock_client, model_cfg_with_tier)
    assert score == _TIER_TO_SCORE[ComplexityTier.HIGH]

    # 2. Total fallback to default score if no tier is available
    model_cfg_no_tier = ModelConfig(id="m2", endpoint="http://ep2")
    with patch.object(profiler, "_measure_time_to_first_token_and_throughput", side_effect=RuntimeError("Some error")):
        score = profiler._compute_score(mock_client, model_cfg_no_tier)
    assert score == _DEFAULT_FALLBACK_SCORE


# ------------------------------------------------------------------
# BenchmarkProfiler - Integration & Helpers
# ------------------------------------------------------------------

def test_benchmark_profiler_profile():
    """Verifies orchestration logic and automatic intended_tier assignment."""

    profiler = BenchmarkProfiler()
    models = [
        ModelConfig(id="m1", endpoint="http://ep1"),
        ModelConfig(id="m2", endpoint="http://ep2")
    ]

    mock_client1 = MagicMock()
    mock_client2 = MagicMock()
    
    with patch("core.routing.profiler.build_client", side_effect=[mock_client1, mock_client2]), \
         patch.object(profiler, "_compute_score", side_effect=[0.4, 0.8]):
        
        ranked = profiler.profile(models)

    assert len(ranked) == 2
    assert ranked[0].config.id == "m1"
    assert ranked[0].score == 0.4
    assert isinstance(ranked[0].config.intended_tier, ComplexityTier) # Dynamic check
    
    assert ranked[1].config.id == "m2"
    assert ranked[1].score == 0.8
    assert isinstance(ranked[1].config.intended_tier, ComplexityTier) # Dynamic check


def test_build_client_helper():
    """Verifies that model configurations are correctly passed to the client instance."""
     
    model_cfg = ModelConfig(id="m1", endpoint="http://localhost:8000/v1", api_key="my-key")
    client = build_client(model_cfg)
    
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model_name == "m1"