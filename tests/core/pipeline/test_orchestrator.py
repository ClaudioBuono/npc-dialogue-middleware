import pytest
from unittest.mock import MagicMock, patch
from core.configuration.settings import AppSettings
from core.pipeline import orchestrator as orchestrator_module
from core.pipeline.orchestrator import Orchestrator
from core.types.contexts import GameContext, NPCContext, Dialogue, Talkativeness
from api.schemas import ComposedDialogue
from core.types.enums import MiddlewareState

@pytest.fixture(autouse=True)
def reset_orchestrator():
    Orchestrator.reset_instance()
    yield
    Orchestrator.reset_instance()

@pytest.fixture
def orchestrator(mock_dependencies, mock_state_manager, mock_settings):
    return Orchestrator(**mock_dependencies)

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
def app_settings():
    return AppSettings()

@pytest.fixture
def mock_settings():
    with patch.object(orchestrator_module, "Settings") as m:
        yield m

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
    with patch.object(Orchestrator, "__init__", return_value=None) as mock_init:
        o1 = Orchestrator.get_instance()
        o2 = Orchestrator.get_instance()

    assert o1 is o2
    mock_init.assert_called_once()

def test_set_game_context(mock_state_manager, orchestrator, game_context):
    manager_instance = mock_state_manager.return_value
    orchestrator.set_game_context(game_context)
    assert orchestrator.game_context == game_context
    assert manager_instance.transition_to.call_count == 2
    manager_instance.transition_to.assert_any_call(MiddlewareState.SETTING_CONTEXT)
    manager_instance.transition_to.assert_any_call(MiddlewareState.IDLE)

def test_generate_dialogue(mock_settings, mock_state_manager, orchestrator, game_context, npc_context):
    mock_settings.return_value.profanity_filter = True
    orchestrator.game_context = game_context

    contract_mock = MagicMock()
    orchestrator.contract_builder.build_dialogue_contract.return_value = contract_mock

    client_mock = MagicMock()
    orchestrator.llm_router.select_model.return_value = client_mock

    orchestrator.dialogue_generator.generate.return_value = '{"dialogue": "Hi"}'

    composed = ComposedDialogue(intent=npc_context.intent, dialogue="Hi")
    orchestrator.dialogue_composer.compose_dialogue.return_value = composed
    orchestrator.refiner.refine_dialogue.return_value = composed
    orchestrator.guardrail.validate_composed_output.return_value = True

    result = orchestrator.generate_dialogue(npc_context, "Hello NPC")

    assert result == composed
    orchestrator.dialogue_history.add_player_dialogue_to_history.assert_called_with("Hello NPC")
    orchestrator.dialogue_history.add_npc_dialogue_to_history.assert_called_with(composed)
    orchestrator.contract_builder.build_dialogue_contract.assert_called_once()
    orchestrator.llm_router.select_model.assert_called_once()
    orchestrator.dialogue_generator.set_client.assert_called_with(client_mock)
    orchestrator.dialogue_generator.generate.assert_called_with(contract_mock)
    orchestrator.dialogue_composer.compose_dialogue.assert_called_once()
    orchestrator.refiner.set_client.assert_called_with(client_mock)
    orchestrator.refiner.refine_dialogue.assert_called_once_with(composed, npc_context, game_context)
    orchestrator.guardrail.validate_composed_output.assert_called_once()

def test_generate_dialogue_refused(mock_settings, mock_state_manager, orchestrator, game_context, npc_context):
    mock_settings.return_value.profanity_filter = True
    orchestrator.game_context = game_context
    orchestrator.guardrail.validate_composed_output.return_value = False

    result = orchestrator.generate_dialogue(npc_context, None)

    assert result is None
    orchestrator.dialogue_history.add_npc_dialogue_to_history.assert_not_called()


def test_generate_dialogue_stream(mock_state_manager, orchestrator, game_context, npc_context):
    orchestrator.game_context = game_context
    orchestrator.guardrail.get_streaming_scanner.return_value = None

    orchestrator.dialogue_generator.generate_stream.return_value = iter(["Hel", "lo"])

    composed = ComposedDialogue(intent=npc_context.intent, dialogue="Hello")
    orchestrator.dialogue_composer.compose_dialogue.return_value = composed

    chunks = list(orchestrator.generate_dialogue_stream(npc_context, None))

    assert chunks == ["Hel", "lo"]
    orchestrator.dialogue_composer.compose_dialogue.assert_called_with(npc_context, "Hello")
    orchestrator.dialogue_history.add_npc_dialogue_to_history.assert_called_with(composed)

# TODO: Test streaming censorship in generate_dialogue_stream
#   - Profanity modes: parametrize over ProfanityMode (except DISABLED) with a scanner
#     mock that reports a match; verify redact_terms is called and the censored term
#     never appears in the emitted chunks
#   - DISABLED mode: no redaction, chunks emitted as-is
#   - Buffer logic: with scanner._max_len > 0, chunks are held back by max_len chars
#     and the residual buffer is flushed at the end of the stream (nothing lost)
#   - Empty chunks are skipped and don't alter the result
#   - Player choice is added to the history at the start of the stream
#   - After a redaction: verify what ends up in the history and what is passed to
#     compose_dialogue (censored or original text)
#   - Integration: one test with a real Guardrail (patched _load_derogatory_terms)
#     and a fake generator emitting a bad word, to check scanner and orchestrator
#     work together
#
# TODO: Test the non-streaming generate_dialogue
#   - Profanity DISABLED: validate_composed_output is not called, dialogue is saved
#   - No player choice: add_player_dialogue_to_history is not called
#   - Refiner modifies the dialogue: the refined version goes to history and return value
#   - State transitions: GENERATING then IDLE (assert_has_calls), also on refusal
#   - Exception mid-pipeline (e.g. generate raises): is the state left on GENERATING?
#     Possible real bug, there is no try/finally
#
# TODO: Test set_game_context when something raises: does the state return to IDLE?

# NOTE: detection of terms split across chunks belongs to the scanner's own tests