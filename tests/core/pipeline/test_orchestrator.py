from contextlib import contextmanager
import pytest
from unittest.mock import MagicMock, call, patch
from api.schemas import ComposedDialogue
from core.configuration.settings import AppSettings, Settings
from core.pipeline import orchestrator as orchestrator_module
from core.pipeline.orchestrator import Orchestrator
from core.types.contexts import Dialogue, GameContext, NPCContext, Talkativeness
from core.types.enums import MiddlewareState


@pytest.fixture(autouse=True)
def reset_orchestrator():
    """Ensure every test starts and ends with a clean Singleton state."""
    Orchestrator.reset_instance()
    yield
    Orchestrator.reset_instance()


@pytest.fixture
def mock_dependencies():
    return {
        "guardrail": MagicMock(),
        "dialogue_history": MagicMock(),
        "contract_builder": MagicMock(),
        "llm_router": MagicMock(),
        "dialogue_generator": MagicMock(),
        "dialogue_composer": MagicMock(),
        "refiner": MagicMock(),
    }


@pytest.fixture
def mock_state_manager():
    with patch.object(orchestrator_module, "StateManager") as m:
        yield m


@pytest.fixture
def orchestrator(mock_dependencies):
    return Orchestrator(**mock_dependencies)


@pytest.fixture
def game_context():
    return GameContext(
        environment="Town Square",
        epoch="Medieval Fantasy",
        world_state="Peaceful day",
        main_character_description="A brave adventurer",
    )


@pytest.fixture
def npc_context():
    return NPCContext(
        name="Bob",
        age="30",
        personality="Calm and friendly",
        context="At his shop",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Neutral customer",
        intent=Dialogue(type="Dialogue", has_options=False),
    )

@pytest.fixture
def set_app_settings(monkeypatch):
    def _set(**overrides):
        app = AppSettings(**overrides)
        monkeypatch.setattr(Settings, "_settings", app)   # restored after the tests
        return app
    return _set

# --- Singleton Pattern & Lifecycle Tests ---

def test_singleton_get_instance(mock_dependencies):
    """Orchestrator.get_instance returns the exact same Singleton instance."""
    o1 = Orchestrator(**mock_dependencies)
    o2 = Orchestrator.get_instance()
    assert o1 is o2


def test_singleton_reset_instance(mock_dependencies):
    """reset_instance clears the Singleton, allowing a fresh instance to be constructed."""
    o1 = Orchestrator(**mock_dependencies)
    Orchestrator.reset_instance()
    o2 = Orchestrator(**mock_dependencies)
    assert o1 is not o2


def test_singleton_init_idempotency(mock_dependencies):
    """Calling __init__ a second time on an already-initialized instance does not overwrite attributes."""
    o = Orchestrator(**mock_dependencies)
    original_builder = o.contract_builder

    second_mock = MagicMock()
    deps2 = {**mock_dependencies, "contract_builder": second_mock}
    o.__init__(**deps2)

    assert o.contract_builder is original_builder


# --- set_game_context Tests ---

def test_set_game_context(mock_state_manager, orchestrator, game_context):
    """set_game_context transitions StateManager through SETTING_CONTEXT and IDLE, and sets property."""
    manager_instance = mock_state_manager.return_value
    orchestrator.set_game_context(game_context)

    assert orchestrator.game_context == game_context
    assert manager_instance.transition_to.call_count == 2
    manager_instance.transition_to.assert_has_calls([
        call(MiddlewareState.SETTING_CONTEXT),
        call(MiddlewareState.IDLE),
    ])


def test_set_game_context_debug_logging(mock_state_manager, orchestrator, game_context):
    """set_game_context logs debug output when logger is enabled for DEBUG."""
    with patch.object(orchestrator_module.logger, "isEnabledFor", return_value=True), \
         patch.object(orchestrator_module.logger, "debug") as mock_debug:
        orchestrator.set_game_context(game_context)
        mock_debug.assert_called_once()
        assert "Game context:" in mock_debug.call_args[0][0]


# --- generate_dialogue Happy Path & Feature Flag Tests ---

def test_generate_dialogue_with_refiner_enabled(
    set_app_settings, mock_state_manager, orchestrator, game_context, npc_context
):
    """When refine_dialogue is enabled, Refiner is executed and Guardrail censor is bypassed."""
    app_settings = set_app_settings(refine_dialogue=True, profanity_filter=False)

    orchestrator.game_context = game_context
    contract_mock = MagicMock()
    orchestrator.contract_builder.build_dialogue_contract.return_value = contract_mock
    client_mock = MagicMock()
    orchestrator.llm_router.select_model.return_value = client_mock
    orchestrator.dialogue_generator.generate.return_value = '{"dialogue": "Raw text"}'

    composed_dialogue = ComposedDialogue(intent=npc_context.intent, dialogue="Composed text")
    refined_dialogue = ComposedDialogue(intent=npc_context.intent, dialogue="Refined text")
    orchestrator.dialogue_composer.compose_dialogue.return_value = composed_dialogue
    orchestrator.refiner.refine_dialogue.return_value = refined_dialogue

    result = orchestrator.generate_dialogue(npc_context, "Hello NPC")

    assert result == refined_dialogue
    orchestrator.dialogue_history.add_player_dialogue_to_history.assert_called_once_with("Hello NPC")
    orchestrator.contract_builder.build_dialogue_contract.assert_called_once_with(
        game_context, npc_context, orchestrator.dialogue_history.get_dialogue_history()
    )
    orchestrator.llm_router.select_model.assert_called_once_with(
        game_context=game_context, npc_context=npc_context
    )
    orchestrator.dialogue_generator.set_client.assert_called_once_with(client_mock)
    orchestrator.dialogue_generator.generate.assert_called_once_with(contract_mock)
    orchestrator.dialogue_composer.compose_dialogue.assert_called_once_with(npc_context, '{"dialogue": "Raw text"}')

    orchestrator.refiner.set_client.assert_called_once_with(client_mock)
    orchestrator.refiner.refine_dialogue.assert_called_once_with(composed_dialogue, npc_context, game_context)
    orchestrator.guardrail.censor_composed_dialogue.assert_not_called()
    orchestrator.dialogue_history.add_npc_dialogue_to_history.assert_called_once_with(refined_dialogue)


def test_generate_dialogue_without_refiner_with_profanity_filter(
    set_app_settings, mock_state_manager, orchestrator, game_context, npc_context
):
    """When refine_dialogue=False and profanity_filter=True, Guardrail censors the composed dialogue."""
    app_settings = set_app_settings(refine_dialogue=False, profanity_filter=True)

    orchestrator.game_context = game_context
    composed = ComposedDialogue(intent=npc_context.intent, dialogue="Dirty text")
    censored = ComposedDialogue(intent=npc_context.intent, dialogue="Censored text")

    orchestrator.dialogue_composer.compose_dialogue.return_value = composed
    orchestrator.guardrail.censor_composed_dialogue.return_value = censored

    result = orchestrator.generate_dialogue(npc_context, None)

    assert result == censored
    orchestrator.refiner.refine_dialogue.assert_not_called()
    orchestrator.guardrail.censor_composed_dialogue.assert_called_once_with(composed)
    orchestrator.dialogue_history.add_npc_dialogue_to_history.assert_called_once_with(censored)
    # last_player_choice was None, so add_player_dialogue_to_history is skipped
    orchestrator.dialogue_history.add_player_dialogue_to_history.assert_not_called()


def test_generate_dialogue_neither_refiner_nor_filter(
    set_app_settings, mock_state_manager, orchestrator, game_context, npc_context
):
    """When both refine_dialogue and profanity_filter are disabled, composed dialogue is stored unmodified."""
    app_settings = set_app_settings(refine_dialogue=False, profanity_filter=False)

    orchestrator.game_context = game_context
    composed = ComposedDialogue(intent=npc_context.intent, dialogue="Raw dialogue")
    orchestrator.dialogue_composer.compose_dialogue.return_value = composed

    result = orchestrator.generate_dialogue(npc_context, None)

    assert result == composed
    orchestrator.refiner.refine_dialogue.assert_not_called()
    orchestrator.guardrail.censor_composed_dialogue.assert_not_called()
    orchestrator.dialogue_history.add_npc_dialogue_to_history.assert_called_once_with(composed)


# --- generate_dialogue Error Handling & State Transitions ---

def test_generate_dialogue_state_transitions(
    set_app_settings, mock_state_manager, orchestrator, game_context, npc_context
):
    """generate_dialogue transitions state to GENERATING and returns to IDLE in finally block."""
    app_settings = set_app_settings(refine_dialogue=False, profanity_filter=False)

    orchestrator.game_context = game_context
    manager_instance = mock_state_manager.return_value

    orchestrator.generate_dialogue(npc_context, None)

    manager_instance.transition_to.assert_has_calls([
        call(MiddlewareState.GENERATING),
        call(MiddlewareState.IDLE),
    ])


def test_generate_dialogue_exception_handling_reverts_state(
    set_app_settings, mock_state_manager, orchestrator, game_context, npc_context
):
    """When an exception occurs during dialogue generation, state reverts to IDLE and exception is raised."""
    app_settings = set_app_settings(refine_dialogue=False, profanity_filter=False)

    orchestrator.game_context = game_context
    manager_instance = mock_state_manager.return_value
    orchestrator.dialogue_generator.generate.side_effect = RuntimeError("Generation failed")

    with pytest.raises(RuntimeError, match="Generation failed"):
        orchestrator.generate_dialogue(npc_context, "Player choice")

    # State still restored to IDLE
    manager_instance.transition_to.assert_has_calls([
        call(MiddlewareState.GENERATING),
        call(MiddlewareState.IDLE),
    ])
    orchestrator.dialogue_history.add_npc_dialogue_to_history.assert_not_called()


# --- generate_dialogue_stream Tests ---

def test_generate_dialogue_stream_happy_path_without_scanner(
    monkeypatch, mock_state_manager, orchestrator, game_context, npc_context
):
    """generate_dialogue_stream yields stream chunks, updates history, and transitions state cleanly."""
    orchestrator.game_context = game_context
    orchestrator.guardrail.get_streaming_scanner.return_value = None
    orchestrator.dialogue_generator.generate_stream.return_value = iter(["Hello", " ", "traveler"])

    composed = ComposedDialogue(intent=npc_context.intent, dialogue="Hello traveler")
    orchestrator.dialogue_composer.compose_dialogue.return_value = composed

    chunks = list(orchestrator.generate_dialogue_stream(npc_context, "Greeting"))

    assert chunks == ["Hello", " ", "traveler"]
    orchestrator.dialogue_history.add_player_dialogue_to_history.assert_called_once_with("Greeting")
    orchestrator.dialogue_composer.compose_dialogue.assert_called_once_with(npc_context, "Hello traveler")
    orchestrator.dialogue_history.add_npc_dialogue_to_history.assert_called_once_with(composed)

    manager_instance = mock_state_manager.return_value
    manager_instance.transition_to.assert_has_calls([
        call(MiddlewareState.GENERATING),
        call(MiddlewareState.IDLE),
    ])


def test_generate_dialogue_stream_with_scanner_sliding_window_and_redaction(
    orchestrator, game_context, npc_context
):
    """Guarded streaming with max_len buffers characters and redacts detected matches."""
    orchestrator.game_context = game_context

    mock_scanner = MagicMock()
    mock_scanner._max_len = 3
    # First chunk clean, second chunk triggers match, flush has no pending matches
    mock_scanner.feed.side_effect = [None, ["curse"], None]
    mock_scanner.flush.return_value = None
    orchestrator.guardrail.get_streaming_scanner.return_value = mock_scanner
    orchestrator.guardrail.redact_terms.side_effect = lambda buf, terms: buf.replace("curse", "[CENSORED]")

    raw_chunks = ["Hello ", "curse word", "!"]
    orchestrator.dialogue_generator.generate_stream.return_value = iter(raw_chunks)

    released = list(orchestrator.generate_dialogue_stream(npc_context, None))
    full_output = "".join(released)

    assert "[CENSORED]" in full_output
    assert "curse" not in full_output
    assert orchestrator.guardrail.redact_terms.called


def test_generate_dialogue_stream_with_scanner_flush_match(
    orchestrator, game_context, npc_context
):
    """Guarded streaming redacts matches identified when flushing at the end of the stream."""
    orchestrator.game_context = game_context

    mock_scanner = MagicMock()
    mock_scanner._max_len = 10
    mock_scanner.feed.return_value = None
    mock_scanner.flush.return_value = ["profane"]
    orchestrator.guardrail.get_streaming_scanner.return_value = mock_scanner
    orchestrator.guardrail.redact_terms.side_effect = lambda buf, terms: buf.replace("profane", "[CENSORED]")

    orchestrator.dialogue_generator.generate_stream.return_value = iter(["End with profane"])

    released = list(orchestrator.generate_dialogue_stream(npc_context, None))
    full_output = "".join(released)

    assert "[CENSORED]" in full_output
    assert "profane" not in full_output


def test_generate_dialogue_stream_skips_empty_chunks(
    orchestrator, game_context, npc_context
):
    """Guarded stream filters out empty string chunks without affecting emitted text."""
    orchestrator.game_context = game_context
    orchestrator.guardrail.get_streaming_scanner.return_value = None

    raw_chunks = ["chunk1", "", "chunk2", ""]
    orchestrator.dialogue_generator.generate_stream.return_value = iter(raw_chunks)

    released = list(orchestrator.generate_dialogue_stream(npc_context, None))
    assert released == ["chunk1", "chunk2"]


def test_generate_dialogue_stream_exception_during_iteration(
    mock_state_manager, orchestrator, game_context, npc_context
):
    """When stream generator raises an error mid-stream, StateManager transitions back to IDLE."""
    orchestrator.game_context = game_context
    orchestrator.guardrail.get_streaming_scanner.return_value = None

    def failing_stream():
        yield "Chunk 1"
        raise RuntimeError("Stream broken")

    orchestrator.dialogue_generator.generate_stream.return_value = failing_stream()

    stream = orchestrator.generate_dialogue_stream(npc_context, None)
    assert next(stream) == "Chunk 1"

    with pytest.raises(RuntimeError, match="Stream broken"):
        next(stream)

    manager_instance = mock_state_manager.return_value
    manager_instance.transition_to.assert_has_calls([
        call(MiddlewareState.GENERATING),
        call(MiddlewareState.IDLE),
    ])

def test_generate_dialogue_stream_no_censoring_when_profanity_filter_off(
    set_app_settings, mock_state_manager, orchestrator, game_context, npc_context
):
    """With profanity_filter off, the stream is released untouched."""
    set_app_settings(profanity_filter=False)
    orchestrator.game_context = game_context

    scanner = MagicMock()
    scanner._max_len = 3
    scanner.feed.return_value = ["curse"]      # lo scanner troverebbe qualcosa...
    scanner.flush.return_value = ["curse"]
    orchestrator.guardrail.get_streaming_scanner.return_value = scanner
    orchestrator.dialogue_generator.generate_stream.return_value = iter(["Hello ", "curse", " word"])

    released = list(orchestrator.generate_dialogue_stream(npc_context, None))

    assert "".join(released) == "Hello curse word"
    orchestrator.guardrail.redact_terms.assert_not_called()


# --- _save_streamed_dialogue Tests ---

def test_save_streamed_dialogue_handles_composer_exception(
    orchestrator, npc_context
):
    """_save_streamed_dialogue catches and logs composer exceptions without raising."""
    orchestrator.dialogue_composer.compose_dialogue.side_effect = ValueError("Corrupted stream")

    with patch.object(orchestrator_module.logger, "exception") as mock_log_exc:
        orchestrator._save_streamed_dialogue(npc_context, "Corrupted dialogue text")
        mock_log_exc.assert_called_once_with("Failed to compose dialogue after stream")

    orchestrator.dialogue_history.add_npc_dialogue_to_history.assert_not_called()