from unittest.mock import MagicMock
import pytest
from api.schemas import ComposedDialogue
from core.generation.generate_service import GenerateService
from core.generation.history import DialogueHistory
from core.infrastructure.state_manager import StateManager
from core.pipeline.guardrail import Guardrail
from core.pipeline.orchestrator import Orchestrator
from core.tools.errors import MiddlewareError, MiddlewareErrorCode, PreProcessingError
from core.types.contexts import Dialogue, GameContext, NPCContext, Talkativeness
from core.types.dataclasses import DialogueStream
from core.types.enums import MiddlewareState


@pytest.fixture(autouse=True)
def reset_state():
    """Ensure StateManager is clean and in IDLE state for each test."""
    StateManager.reset_instance()
    state_mgr = StateManager()
    state_mgr.transition_to(MiddlewareState.IDLE)
    yield
    StateManager.reset_instance()


@pytest.fixture
def valid_game_context():
    return GameContext(
        epoch="Medieval Fantasy",
        environment="Dense enchanted woodland",
        world_state="Peaceful kingdom",
        main_character_description="A brave adventurer",
    )


@pytest.fixture
def valid_npc_context():
    dialogue = Dialogue(type="Dialogue", number_of_options=2)
    return NPCContext(
        name="Rowan",
        age=28,
        personality="Cheerful and curious",
        context="Tending to herbs in a sunlit garden",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Warm and welcoming",
        intent=dialogue,
    )


@pytest.fixture
def mock_orchestrator(valid_game_context):
    orchestrator = MagicMock(spec=Orchestrator)
    orchestrator.game_context = valid_game_context
    return orchestrator


@pytest.fixture
def mock_guardrail():
    guardrail = MagicMock(spec=Guardrail)
    guardrail.validate_game_context.return_value = True
    guardrail.validate_npc_context.return_value = True
    return guardrail


@pytest.fixture
def mock_dialogue_history():
    history = MagicMock(spec=DialogueHistory)
    history.is_empty.return_value = False
    return history


@pytest.fixture
def service(mock_orchestrator, mock_guardrail, mock_dialogue_history):
    return GenerateService(
        orchestrator=mock_orchestrator,
        guardrail=mock_guardrail,
        dialogue_history=mock_dialogue_history,
    )


# ==============================================================================
# 1. set_game_context Tests
# ==============================================================================

def test_set_game_context_happy_path(service, mock_orchestrator, mock_guardrail, valid_game_context):
    service.set_game_context(valid_game_context)

    mock_guardrail.validate_game_context.assert_called_once_with(valid_game_context)
    mock_orchestrator.set_game_context.assert_called_once_with(valid_game_context)


def test_set_game_context_guardrail_refused(service, mock_orchestrator, mock_guardrail, valid_game_context):
    mock_guardrail.validate_game_context.return_value = False

    with pytest.raises(MiddlewareError) as exc_info:
        service.set_game_context(valid_game_context)

    assert exc_info.value.code == MiddlewareErrorCode.REFUSED
    assert "The middleware refused the game context." in exc_info.value.errors
    mock_orchestrator.set_game_context.assert_not_called()


def test_set_game_context_preprocessing_failure(service, mock_orchestrator, valid_game_context):
    valid_game_context.epoch = ""  # Will fail semantic validation

    with pytest.raises(PreProcessingError):
        service.set_game_context(valid_game_context)

    mock_orchestrator.set_game_context.assert_not_called()


# ==============================================================================
# 2. generate_dialogue Precondition Tests
# ==============================================================================

def test_generate_dialogue_context_not_set(service, mock_orchestrator, valid_npc_context):
    mock_orchestrator.game_context = None

    with pytest.raises(MiddlewareError) as exc_info:
        service.generate_dialogue(valid_npc_context, last_player_choice=None)

    assert exc_info.value.code == MiddlewareErrorCode.CONTEXT_NOT_SET
    assert "Game context is not set." in exc_info.value.errors


def test_generate_dialogue_middleware_not_idle(service, valid_npc_context):
    StateManager().transition_to(MiddlewareState.GENERATING)

    with pytest.raises(MiddlewareError) as exc_info:
        service.generate_dialogue(valid_npc_context, last_player_choice=None)

    assert exc_info.value.code == MiddlewareErrorCode.GENERATING
    assert "The middleware is busy generating." in exc_info.value.errors


def test_generate_dialogue_guardrail_refused_npc(service, mock_guardrail, valid_npc_context):
    mock_guardrail.validate_npc_context.return_value = False

    with pytest.raises(MiddlewareError) as exc_info:
        service.generate_dialogue(valid_npc_context, last_player_choice=None)

    assert exc_info.value.code == MiddlewareErrorCode.REFUSED
    assert "The middleware refused the npc context." in exc_info.value.errors


def test_generate_dialogue_continue_history_empty(service, mock_dialogue_history, valid_npc_context):
    mock_dialogue_history.is_empty.return_value = True

    with pytest.raises(MiddlewareError) as exc_info:
        service.generate_dialogue(valid_npc_context, last_player_choice="Tell me more")

    assert exc_info.value.code == MiddlewareErrorCode.REFUSED
    assert "Dialogue history is empty." in exc_info.value.errors


# ==============================================================================
# 3. generate_dialogue Execution Tests
# ==============================================================================

def test_generate_dialogue_new_conversation_happy_path(
    service, mock_orchestrator, mock_dialogue_history, valid_npc_context
):
    expected_dialogue = ComposedDialogue(
        intent=valid_npc_context.intent,
        dialogue="Hello traveler, need some herbs?",
    )
    mock_orchestrator.generate_dialogue.return_value = expected_dialogue

    result = service.generate_dialogue(valid_npc_context, last_player_choice=None)

    assert result == expected_dialogue
    mock_dialogue_history.clear_dialogue_history.assert_called_once()
    mock_orchestrator.generate_dialogue.assert_called_once_with(valid_npc_context, None)


def test_generate_dialogue_continue_conversation_happy_path(
    service, mock_orchestrator, mock_dialogue_history, valid_npc_context
):
    expected_dialogue = ComposedDialogue(
        intent=valid_npc_context.intent,
        dialogue="These herbs are from the Whispering Woods.",
    )
    mock_orchestrator.generate_dialogue.return_value = expected_dialogue

    result = service.generate_dialogue(valid_npc_context, last_player_choice="Where did you get these?")

    assert result == expected_dialogue
    mock_dialogue_history.clear_dialogue_history.assert_not_called()
    mock_orchestrator.generate_dialogue.assert_called_once_with(
        valid_npc_context, "Where did you get these?"
    )


def test_generate_dialogue_orchestrator_returns_none(service, mock_orchestrator, valid_npc_context):
    mock_orchestrator.generate_dialogue.return_value = None

    with pytest.raises(MiddlewareError) as exc_info:
        service.generate_dialogue(valid_npc_context, last_player_choice=None)

    assert exc_info.value.code == MiddlewareErrorCode.REFUSED
    assert "Dialogue generation was refused." in exc_info.value.errors


# ==============================================================================
# 4. generate_dialogue_stream Tests
# ==============================================================================

def test_generate_dialogue_stream_new_conversation(
    service, mock_orchestrator, mock_dialogue_history, valid_npc_context
):
    def fake_stream(npc, choice):
        yield "Hello "
        yield "there!"

    mock_orchestrator.generate_dialogue_stream.side_effect = fake_stream

    stream_result = service.generate_dialogue_stream(valid_npc_context, last_player_choice=None)

    assert isinstance(stream_result, DialogueStream)
    mock_dialogue_history.clear_dialogue_history.assert_called_once()
    chunks = list(stream_result.chunks)
    assert chunks == ["Hello ", "there!"]


def test_generate_dialogue_stream_continue_conversation(
    service, mock_orchestrator, mock_dialogue_history, valid_npc_context
):
    def fake_stream(npc, choice):
        yield "Continuing "
        yield "story."

    mock_orchestrator.generate_dialogue_stream.side_effect = fake_stream

    stream_result = service.generate_dialogue_stream(valid_npc_context, last_player_choice="Go on")

    assert isinstance(stream_result, DialogueStream)
    mock_dialogue_history.clear_dialogue_history.assert_not_called()
    chunks = list(stream_result.chunks)
    assert chunks == ["Continuing ", "story."]


def test_generate_dialogue_stream_precondition_failures(service, mock_orchestrator, valid_npc_context):
    mock_orchestrator.game_context = None

    with pytest.raises(MiddlewareError) as exc_info:
        service.generate_dialogue_stream(valid_npc_context, last_player_choice=None)

    assert exc_info.value.code == MiddlewareErrorCode.CONTEXT_NOT_SET


def test_safe_stream_handles_orchestrator_exception(service, mock_orchestrator, valid_npc_context):
    def failing_stream(npc, choice):
        yield "Initial chunk"
        raise RuntimeError("LLM network disconnected mid-stream")

    mock_orchestrator.generate_dialogue_stream.side_effect = failing_stream

    stream_result = service.generate_dialogue_stream(valid_npc_context, last_player_choice=None)
    chunks = list(stream_result.chunks)

    assert chunks == ["Initial chunk", "\n[STREAM_ERROR]\n"]
