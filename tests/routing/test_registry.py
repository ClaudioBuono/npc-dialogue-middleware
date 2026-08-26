import pytest
from unittest.mock import MagicMock, patch
from core.routing.models import ModelConfig
from core.routing.registry import ModelRegistry
from core.routing.profiler import RankedModel
from core.types.enums import ComplexityTier
from tools.errors import RoutingConfigError

@pytest.fixture(autouse=True)
def clean_registry():
    # ModelRegistry is a singleton, so clear state before and after each test
    registry = ModelRegistry()
    registry._ranked_models = []
    yield
    registry._ranked_models = []


def test_registry_singleton():
    r1 = ModelRegistry()
    r2 = ModelRegistry()
    assert r1 is r2


def test_set_models_single_model():
    registry = ModelRegistry()
    model = ModelConfig(id="single-model", endpoint="http://localhost:8000/v1", intended_tier=ComplexityTier.HIGH)
    
    registry.set_models([model])
    
    ranked = registry.get_ranked_models()
    assert len(ranked) == 1
    assert ranked[0].config.id == "single-model"
    # Registration without profiling assigns a default high score
    assert ranked[0].score == 0.9


def test_set_models_without_profiling_success():
    registry = ModelRegistry()
    models = [
        ModelConfig(id="model-low", endpoint="http://localhost", intended_tier=ComplexityTier.LOW),
        ModelConfig(id="model-high", endpoint="http://localhost", intended_tier=ComplexityTier.HIGH)
    ]
    
    # set profiler=False
    registry.set_models(models, profiler=False)
    
    ranked = registry.get_ranked_models()
    assert len(ranked) == 2
    # Should be sorted reverse by score (high first, then low)
    assert ranked[0].config.id == "model-high"
    assert ranked[1].config.id == "model-low"


def test_set_models_without_profiling_missing_tier():
    registry = ModelRegistry()
    # Missing intended_tier
    models = [
        ModelConfig(id="model-low", endpoint="http://localhost", intended_tier=ComplexityTier.LOW),
        ModelConfig(id="model-missing", endpoint="http://localhost")
    ]
    
    with pytest.raises(RoutingConfigError) as excinfo:
        registry.set_models(models, profiler=False)
    assert "missing 'intended_tier'" in str(excinfo.value)


def test_set_models_with_profiling():
    registry = ModelRegistry()
    models = [
        ModelConfig(id="m1", endpoint="http://localhost"),
        ModelConfig(id="m2", endpoint="http://localhost")
    ]

    mock_ranked = [
        RankedModel(config=models[0], score=0.4, client=MagicMock()),
        RankedModel(config=models[1], score=0.8, client=MagicMock())
    ]

    # Mock BenchmarkProfiler.profile to return our ranked list
    with patch("core.routing.registry.BenchmarkProfiler.profile", return_value=mock_ranked):
        registry.set_models(models, profiler=True)
    
    ranked = registry.get_ranked_models()
    assert len(ranked) == 2
    # Registry sorts by score descending, so m2 (0.8) should be first, then m1 (0.4)
    assert ranked[0].config.id == "m2"
    assert ranked[1].config.id == "m1"


def test_get_best_for_tier():
    registry = ModelRegistry()
    m_low_1 = ModelConfig(id="low-1", endpoint="http://localhost", intended_tier=ComplexityTier.LOW)
    m_low_2 = ModelConfig(id="low-2", endpoint="http://localhost", intended_tier=ComplexityTier.LOW)
    m_med = ModelConfig(id="med", endpoint="http://localhost", intended_tier=ComplexityTier.MEDIUM)
    
    # Manually populate ranked models with explicit mock scores:
    # low-1: score 0.35, low-2: score 0.25 (low-1 is best for LOW)
    client = MagicMock()
    registry._ranked_models = [
        RankedModel(config=m_med, score=0.6, client=client),
        RankedModel(config=m_low_1, score=0.35, client=client),
        RankedModel(config=m_low_2, score=0.25, client=client),
    ]

    best_low = registry.get_best_for_tier(ComplexityTier.LOW)
    assert best_low is not None
    assert best_low.config.id == "low-1"

    best_med = registry.get_best_for_tier(ComplexityTier.MEDIUM)
    assert best_med is not None
    assert best_med.config.id == "med"

    # No HIGH models registered
    best_high = registry.get_best_for_tier(ComplexityTier.HIGH)
    assert best_high is None
