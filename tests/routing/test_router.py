import pytest
from unittest.mock import MagicMock, patch
from core.routing.models import ModelConfig
from core.routing.registry import ModelRegistry
from core.routing.profiler import RankedModel
from core.routing.router import LLMRouter
from core.routing.complexity_analyzer import ComplexityScore
from core.types.enums import ComplexityTier
from core.types.contexts import GameContext, NPCContext, Dialogue, Talkativeness
from tools.errors import LLMClientError

@pytest.fixture(autouse=True)
def clean_registry():
    ModelRegistry()._ranked_models = []
    yield
    ModelRegistry()._ranked_models = []


def test_router_select_model_empty_registry():
    router = LLMRouter()
    game_ctx = MagicMock(spec=GameContext)
    npc_ctx = MagicMock(spec=NPCContext)
    
    with pytest.raises(LLMClientError) as excinfo:
        router.select_model(game_ctx, npc_ctx)
    assert "No models are currently registered" in str(excinfo.value)


def test_router_select_model_single_model():
    router = LLMRouter()
    game_ctx = MagicMock(spec=GameContext)
    npc_ctx = MagicMock(spec=NPCContext)
    
    # Register only one model
    model = ModelConfig(id="only-one", endpoint="http://localhost")
    client = MagicMock()
    ModelRegistry()._ranked_models = [RankedModel(config=model, score=0.6, client=client)]
    
    # Should directly return the client without calling complexity analyzer
    with patch.object(router.complexity_analyzer, "analyze") as mock_analyze:
        selected_client = router.select_model(game_ctx, npc_ctx)
        mock_analyze.assert_not_called()
        assert selected_client is client


def test_router_handle_low_complexity():
    router = LLMRouter()
    
    m_low = RankedModel(config=ModelConfig(id="low", endpoint="http://l", intended_tier=ComplexityTier.LOW), score=0.3, client=MagicMock())
    m_med = RankedModel(config=ModelConfig(id="med", endpoint="http://m", intended_tier=ComplexityTier.MEDIUM), score=0.6, client=MagicMock())
    m_high = RankedModel(config=ModelConfig(id="high", endpoint="http://h", intended_tier=ComplexityTier.HIGH), score=0.9, client=MagicMock())

    # 1. LOW model available
    ModelRegistry()._ranked_models = [m_high, m_med, m_low]
    assert router._handle_low_complexity() is m_low

    # 2. No LOW model, but MEDIUM model available
    ModelRegistry()._ranked_models = [m_high, m_med]
    assert router._handle_low_complexity() is m_med

    # 3. Only HIGH model available
    ModelRegistry()._ranked_models = [m_high]
    assert router._handle_low_complexity() is m_high


def test_router_handle_high_complexity():
    router = LLMRouter()
    
    m_low = RankedModel(config=ModelConfig(id="low", endpoint="http://l", intended_tier=ComplexityTier.LOW), score=0.3, client=MagicMock())
    m_med = RankedModel(config=ModelConfig(id="med", endpoint="http://m", intended_tier=ComplexityTier.MEDIUM), score=0.6, client=MagicMock())
    m_high = RankedModel(config=ModelConfig(id="high", endpoint="http://h", intended_tier=ComplexityTier.HIGH), score=0.9, client=MagicMock())

    # 1. HIGH model available
    ModelRegistry()._ranked_models = [m_high, m_med, m_low]
    assert router._handle_high_complexity() is m_high

    # 2. No HIGH model, but MEDIUM model available
    ModelRegistry()._ranked_models = [m_med, m_low]
    assert router._handle_high_complexity() is m_med

    # 3. Only LOW model available
    ModelRegistry()._ranked_models = [m_low]
    assert router._handle_high_complexity() is m_low


def test_router_handle_medium_complexity():
    router = LLMRouter()
    
    m_low = RankedModel(config=ModelConfig(id="low", endpoint="http://l", intended_tier=ComplexityTier.LOW), score=0.3, client=MagicMock())
    m_med = RankedModel(config=ModelConfig(id="med", endpoint="http://m", intended_tier=ComplexityTier.MEDIUM), score=0.6, client=MagicMock())
    m_high = RankedModel(config=ModelConfig(id="high", endpoint="http://h", intended_tier=ComplexityTier.HIGH), score=0.9, client=MagicMock())

    # 1. MEDIUM model available -> should return MEDIUM model regardless of score
    ModelRegistry()._ranked_models = [m_high, m_med, m_low]
    assert router._handle_medium_complexity(0.40) is m_med
    assert router._handle_medium_complexity(0.60) is m_med

    # 2. No MEDIUM model, score closer to HIGH (e.g. 0.60)
    # Under LOW_THRESHOLD=0.35, HIGH_THRESHOLD=0.70:
    # low_diff = 0.60 - 0.35 = 0.25
    # high_diff = 0.70 - 0.60 = 0.10
    # high_diff < low_diff is True -> Closer to HIGH
    # Should check HIGH first, then LOW.
    ModelRegistry()._ranked_models = [m_high, m_low]
    assert router._handle_medium_complexity(0.60) is m_high
    
    ModelRegistry()._ranked_models = [m_low]
    assert router._handle_medium_complexity(0.60) is m_low

    # 3. No MEDIUM model, score closer to LOW (e.g. 0.40)
    # Under LOW_THRESHOLD=0.35, HIGH_THRESHOLD=0.70:
    # low_diff = 0.40 - 0.35 = 0.05
    # high_diff = 0.70 - 0.40 = 0.30
    # high_diff < low_diff is False -> Closer to LOW
    # Should check LOW first, then HIGH.
    ModelRegistry()._ranked_models = [m_high, m_low]
    assert router._handle_medium_complexity(0.40) is m_low
    
    ModelRegistry()._ranked_models = [m_high]
    assert router._handle_medium_complexity(0.40) is m_high


def test_router_select_model_integration():
    router = LLMRouter()
    game_ctx = GameContext(epoch="Medieval", environment="Forest", world_state="Peaceful")
    npc_ctx = NPCContext(
        name="John", age=25, personality="Kind", context="Cabin",
        intent=Dialogue(must_use_expression="Hi"), talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Neutral"
    )

    m_low = RankedModel(config=ModelConfig(id="low", endpoint="http://l", intended_tier=ComplexityTier.LOW), score=0.3, client=MagicMock())
    m_high = RankedModel(config=ModelConfig(id="high", endpoint="http://h", intended_tier=ComplexityTier.HIGH), score=0.9, client=MagicMock())
    ModelRegistry()._ranked_models = [m_high, m_low]

    # Analyze mock returning LOW tier
    score_low = ComplexityScore(value=0.2, tier=ComplexityTier.LOW, breakdown={})
    with patch.object(router.complexity_analyzer, "analyze", return_value=score_low):
        client = router.select_model(game_ctx, npc_ctx)
        assert client is m_low.client

    # Analyze mock returning HIGH tier
    score_high = ComplexityScore(value=0.8, tier=ComplexityTier.HIGH, breakdown={})
    with patch.object(router.complexity_analyzer, "analyze", return_value=score_high):
        client = router.select_model(game_ctx, npc_ctx)
        assert client is m_high.client
