import pytest
from unittest.mock import MagicMock, patch
from core.routing.models import ModelConfig
from core.routing.registry import ModelRegistry
from core.routing.profiler import RankedModel
from core.types.enums import ComplexityTier
from core.tools.errors import RoutingConfigError

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


def test_set_models_empty_list():
    """Verifies that passing an empty list clears ranked models and does not error."""
    registry = ModelRegistry()
    registry.set_models([])
    assert registry.get_ranked_models() == []


def test_set_models_none_uses_internal_configs():
    """Verifies that calling set_models with None uses the pre-loaded _models_configs."""
    registry = ModelRegistry()
    default_cfg = ModelConfig(id="default-model", endpoint="http://localhost", intended_tier=ComplexityTier.LOW)
    registry._models_configs = [default_cfg]

    registry.set_models(None)
    ranked = registry.get_ranked_models()
    assert len(ranked) == 1
    assert ranked[0].config.id == "default-model"


def test_set_models_single_model_without_intended_tier():
    """Verifies fallback score (0.6 - MEDIUM) when a single model lacks intended_tier."""
    registry = ModelRegistry()
    model = ModelConfig(id="single-no-tier", endpoint="http://localhost:8000/v1")

    registry.set_models([model])
    ranked = registry.get_ranked_models()
    assert len(ranked) == 1
    assert ranked[0].config.id == "single-no-tier"
    assert ranked[0].score == 0.6


def test_validate_tiers_declared_multiple_missing():
    """Verifies error reporting when multiple models are missing intended_tier."""
    registry = ModelRegistry()
    models = [
        ModelConfig(id="m1-missing", endpoint="http://ep1"),
        ModelConfig(id="m2-missing", endpoint="http://ep2"),
    ]
    with pytest.raises(RoutingConfigError) as excinfo:
        registry.set_models(models, profiler=False)

    error_str = str(excinfo.value)
    assert "m1-missing" in error_str
    assert "m2-missing" in error_str


def test_print_ranked_models_empty_and_populated():
    """Verifies _print_ranked_models formatting with empty state and optional fields."""
    registry = ModelRegistry()

    # 1. Empty state
    registry._ranked_models = []
    registry._print_ranked_models()

    # 2. Populated state with None fields and formatted max context tokens
    client = MagicMock()
    m1 = ModelConfig(id="m1", endpoint="http://ep1", intended_tier=None, max_context_tokens=None)
    m2 = ModelConfig(id="m2", endpoint="http://ep2", intended_tier=ComplexityTier.HIGH, max_context_tokens=128000)
    registry._ranked_models = [
        RankedModel(config=m2, score=0.9, client=client),
        RankedModel(config=m1, score=0.6, client=client),
    ]
    registry._print_ranked_models()


def test_get_best_for_tier_empty_registry():
    """Verifies that querying an empty registry returns None."""
    registry = ModelRegistry()
    registry._ranked_models = []
    assert registry.get_best_for_tier(ComplexityTier.LOW) is None
    assert registry.get_best_for_tier(ComplexityTier.MEDIUM) is None
    assert registry.get_best_for_tier(ComplexityTier.HIGH) is None


def test_registry_new_initialization():
    """Verifies that __new__ correctly resolves config and initializes attributes when _instance is None."""
    orig_instance = ModelRegistry._instance
    try:
        ModelRegistry._instance = None
        with patch("core.routing.registry.resolve_config_file", return_value="dummy.yaml"), \
             patch("core.routing.registry.load_config_from_file", return_value=[]):
            inst = ModelRegistry()
            assert inst._ranked_models == []
            assert inst._models_configs == []
    finally:
        ModelRegistry._instance = orig_instance

