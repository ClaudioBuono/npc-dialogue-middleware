import pytest
from unittest.mock import MagicMock, patch
from core.orchestrator import Orchestrator
from core.types.contexts import GameContext, NPCContext, Dialogue, Talkativeness
from api.schemas import ComposedDialogue
from core.types.enums import MiddlewareState

@pytest.fixture(autouse=True)
def reset_orchestrator():
    Orchestrator.reset_instance()
    yield
    Orchestrator.reset_instance()

@pytest.fixture
def orchestrator():
    return Orchestrator.get_instance()

@pytest.fixture
def mock_dependencies(orchestrator):
    orchestrator.contract_builder = MagicMock()
    orchestrator.llm_router = MagicMock()
    orchestrator.dialogue_generator = MagicMock()
    orchestrator.dialogue_history = MagicMock()
    orchestrator.dialogue_composer = MagicMock()
    orchestrator.guardrail = MagicMock()
    return orchestrator

@pytest.fixture
def game_context():
    return GameContext(environment="Town", epoch="Modern", world_state="Peace", main_character_description="Hero")

@pytest.fixture
def npc_context():
    return NPCContext(
        name="Bob",
        age="30",
        personality="Calm",
        context="Shop",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Neutral",
        intent=Dialogue(type="Dialogue", has_options=False)
    )

def test_singleton():
    o1 = Orchestrator.get_instance()
    o2 = Orchestrator.get_instance()
    assert o1 is o2

@patch("core.orchestrator.StateManager")
def test_set_game_context(mock_state_manager, orchestrator, game_context):
    manager_instance = mock_state_manager.return_value
    orchestrator.set_game_context(game_context)
    assert orchestrator.game_context == game_context
    assert manager_instance.transition_to.call_count == 2
    manager_instance.transition_to.assert_any_call(MiddlewareState.SETTING_CONTEXT)
    manager_instance.transition_to.assert_any_call(MiddlewareState.IDLE)

@patch("core.orchestrator.StateManager")
@patch("core.orchestrator.Settings")
def test_generate_dialogue(mock_settings, mock_state_manager, mock_dependencies, game_context, npc_context):
    # Setup mocks
    mock_settings.return_value.profanity_filter = True
    mock_dependencies.game_context = game_context
    
    contract_mock = MagicMock()
    mock_dependencies.contract_builder.build.return_value = contract_mock
    
    client_mock = MagicMock()
    mock_dependencies.llm_router.select_model.return_value = client_mock
    
    mock_dependencies.dialogue_generator.generate.return_value = '{"dialogue": "Hi"}'
    
    composed = ComposedDialogue(intent=npc_context.intent, dialogue="Hi")
    mock_dependencies.dialogue_composer.compose_dialogue.return_value = composed
    
    mock_dependencies.guardrail.validate_composed_output.return_value = True

    # Execution
    result = mock_dependencies.generate_dialogue(npc_context, "Hello NPC")

    # Assertions
    assert result == composed
    mock_dependencies.dialogue_history.add_player_dialogue_to_history.assert_called_with("Hello NPC")
    mock_dependencies.dialogue_history.add_npc_dialogue_to_history.assert_called_with(composed)
    mock_dependencies.contract_builder.build.assert_called_once()
    mock_dependencies.llm_router.select_model.assert_called_once()
    mock_dependencies.dialogue_generator.set_client.assert_called_with(client_mock)
    mock_dependencies.dialogue_generator.generate.assert_called_with(contract_mock)
    mock_dependencies.dialogue_composer.compose_dialogue.assert_called_once()
    mock_dependencies.guardrail.validate_composed_output.assert_called_once()

@patch("core.orchestrator.StateManager")
@patch("core.orchestrator.Settings")
def test_generate_dialogue_refused(mock_settings, mock_state_manager, mock_dependencies, game_context, npc_context):
    mock_settings.return_value.profanity_filter = True
    mock_dependencies.game_context = game_context
    mock_dependencies.guardrail.validate_composed_output.return_value = False
    
    result = mock_dependencies.generate_dialogue(npc_context, None)
    
    assert result is None
    mock_dependencies.dialogue_history.add_npc_dialogue_to_history.assert_not_called()

@patch("core.orchestrator.StateManager")
@patch("core.orchestrator.Settings")
def test_generate_dialogue_stream(mock_settings, mock_state_manager, mock_dependencies, game_context, npc_context):
    mock_settings.return_value.profanity_filter = False
    mock_dependencies.game_context = game_context
    
    mock_dependencies.dialogue_generator.generate_stream.return_value = iter(["Hel", "lo"])
    
    composed = ComposedDialogue(intent=npc_context.intent, dialogue="Hello")
    mock_dependencies.dialogue_composer.compose_dialogue.return_value = composed
    
    stream = mock_dependencies.generate_dialogue_stream(npc_context, None)
    chunks = list(stream)
    
    assert chunks == ["Hel", "lo"]
    mock_dependencies.dialogue_composer.compose_dialogue.assert_called_with(npc_context, "Hello")
    mock_dependencies.dialogue_history.add_npc_dialogue_to_history.assert_called_with(composed)
