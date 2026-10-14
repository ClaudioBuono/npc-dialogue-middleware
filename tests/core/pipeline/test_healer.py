import logging
import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel, ValidationError
from api.schemas import ComposedDialogue, QuestChoiceSchema
from core.configuration.settings import AppSettings, LLMSettings, Settings
from core.generation.contract_builder import ContractBuilder
from core.generation.output_composer import DialogueOutputComposer
from core.pipeline import healer as healer_module
from core.pipeline.healer import Healer
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import PreProcessingError, ValidationErrorCode
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness
from core.types.dataclasses import JudgeIssue, Contract


@pytest.fixture
def mock_contract_builder():
    builder = MagicMock(spec=ContractBuilder)
    builder.build_healer_contract.return_value = Contract(system_prompt="sys", user_prompt="user", output_schema={})
    return builder


@pytest.fixture
def mock_dialogue_composer():
    composer = MagicMock(spec=DialogueOutputComposer)
    composer.compose_dialogue.return_value = ComposedDialogue(intent=Dialogue(), dialogue="Healed dialogue")
    return composer


@pytest.fixture
def mock_client():
    client = MagicMock(spec=OpenAICompatibleClient)
    client.generate.return_value = '{"dialogue": "Healed dialogue"}'
    return client


@pytest.fixture
def healer(mock_contract_builder, mock_dialogue_composer, mock_client):
    h = Healer(contract_builder=mock_contract_builder, dialogue_composer=mock_dialogue_composer)
    h.set_client(mock_client)
    return h


@pytest.fixture
def dummy_contexts():
    game_ctx = GameContext(epoch="Test", environment="Test", world_state="Test")
    npc_ctx = MagicMock(spec=NPCContext)
    npc_ctx.name = "Test NPC"
    npc_ctx.intent = Quest(
        objective="Test",
        name="Test",
        description="Test",
        has_choice=False,
    )
    return game_ctx, npc_ctx


def make_validation_error() -> ValidationError:
    """Build a real pydantic ValidationError instance for use as a side_effect."""
    class Dummy(BaseModel):
        val: int

    try:
        Dummy(val="not-an-int")
    except ValidationError as e:
        return e


# --- Initialization & Client Injection Tests ---

def test_healer_initialization(mock_contract_builder, mock_dialogue_composer):
    """Healer saves provided contract_builder and dialogue_composer."""
    h = Healer(contract_builder=mock_contract_builder, dialogue_composer=mock_dialogue_composer)
    assert h.contract_builder is mock_contract_builder
    assert h.dialogue_composer is mock_dialogue_composer
    assert not hasattr(h, "_client")


def test_healer_set_client(mock_contract_builder, mock_dialogue_composer, mock_client):
    """set_client injects the LLM client instance."""
    h = Healer(contract_builder=mock_contract_builder, dialogue_composer=mock_dialogue_composer)
    h.set_client(mock_client)
    assert h._client is mock_client


# --- Happy Path Execution Tests ---

def test_heal_dialogue_success(healer, mock_contract_builder, mock_dialogue_composer, mock_client, dummy_contexts):
    """Happy path: contract built, LLM called, raw output composed back into ComposedDialogue."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]

    result = healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    assert result.dialogue == "Healed dialogue"
    mock_contract_builder.build_healer_contract.assert_called_once_with(original_dialogue, game_ctx, npc_ctx, issues)
    mock_client.generate.assert_called_once()
    mock_dialogue_composer.compose_dialogue.assert_called_once_with(npc_ctx, '{"dialogue": "Healed dialogue"}')


def test_heal_dialogue_calls_client_with_correct_args(healer, mock_contract_builder, mock_client, dummy_contexts):
    """Verify that client.generate() is invoked with the built contract and Settings().llm.healer_temperature."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]

    healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    expected_contract = mock_contract_builder.build_healer_contract.return_value
    mock_client.generate.assert_called_once_with(
        expected_contract, temperature=Settings().llm.healer_temperature
    )


def test_heal_dialogue_with_custom_temperature(
    monkeypatch, healer, mock_contract_builder, mock_client, dummy_contexts
):
    """Verify that client.generate() respects a custom healer_temperature configuration."""
    custom_temp = 1.15
    app_settings = AppSettings(llm=LLMSettings(healer_temperature=custom_temp))
    monkeypatch.setattr(healer_module, "Settings", lambda: app_settings)

    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")

    healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, [])

    expected_contract = mock_contract_builder.build_healer_contract.return_value
    mock_client.generate.assert_called_once_with(expected_contract, temperature=custom_temp)


def test_heal_dialogue_with_no_issues(healer, mock_contract_builder, mock_dialogue_composer, dummy_contexts):
    """An empty issues list is handled properly by healer and passed to contract builder."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")

    result = healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, [])

    mock_contract_builder.build_healer_contract.assert_called_once_with(original_dialogue, game_ctx, npc_ctx, [])
    assert result.dialogue == "Healed dialogue"


def test_heal_dialogue_with_multiple_issues(healer, mock_contract_builder, dummy_contexts):
    """Multiple issues across categories are forwarded intact to contract builder."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [
        JudgeIssue(category="faithfulness", issue="Bad fact"),
        JudgeIssue(category="entity_check", issue="Unknown entity"),
        JudgeIssue(category="Accept/Refuse", issue="Missing Accept/Refuse options"),
    ]

    healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    mock_contract_builder.build_healer_contract.assert_called_once_with(original_dialogue, game_ctx, npc_ctx, issues)


def test_heal_dialogue_with_quest_choice_options(healer, mock_contract_builder, mock_dialogue_composer, dummy_contexts):
    """Healing dialogues containing QuestChoiceSchema returns the composed model properly."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(
        intent=Quest(type="Quest", objective="Find ring", has_choice=True),
        dialogue="Find my ring",
        player_options=QuestChoiceSchema(accept="Yes", refuse="No"),
    )
    issues = [JudgeIssue(category="persona_consistency", issue="Tone too brief")]
    expected_healed = ComposedDialogue(
        intent=Quest(type="Quest", objective="Find ring", has_choice=True),
        dialogue="Please kind traveler, find my lost heirloom ring!",
        player_options=QuestChoiceSchema(accept="I will search for it.", refuse="No time."),
    )
    mock_dialogue_composer.compose_dialogue.return_value = expected_healed

    result = healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    assert result == expected_healed
    assert result.player_options.accept == "I will search for it."


# --- Validation Error & PreProcessingError Tests ---

def test_heal_dialogue_validation_error(healer, mock_dialogue_composer, dummy_contexts):
    """When compose_dialogue raises ValidationError, healer translates it into PreProcessingError."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]

    mock_dialogue_composer.compose_dialogue.side_effect = make_validation_error()

    with pytest.raises(PreProcessingError) as exc_info:
        healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    assert exc_info.value.code == ValidationErrorCode.INVALID_VALUE


def test_heal_dialogue_validation_error_includes_raw_output(
    healer, mock_dialogue_composer, mock_client, dummy_contexts
):
    """The translated PreProcessingError preserves raw LLM response in its errors list."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]
    mock_client.generate.return_value = '{"malformed": "output"}'
    mock_dialogue_composer.compose_dialogue.side_effect = make_validation_error()

    with pytest.raises(PreProcessingError) as exc_info:
        healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    assert any('{"malformed": "output"}' in err for err in exc_info.value.errors)


def test_heal_dialogue_validation_error_still_calls_upstream_steps(
    healer, mock_contract_builder, mock_client, mock_dialogue_composer, dummy_contexts
):
    """Earlier pipeline steps (contract building, client generation) execute before compose error."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]
    mock_dialogue_composer.compose_dialogue.side_effect = make_validation_error()

    with pytest.raises(PreProcessingError):
        healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    mock_contract_builder.build_healer_contract.assert_called_once()
    mock_client.generate.assert_called_once()


# --- Exception Propagation & Network Failures Tests ---

@pytest.mark.parametrize("exception_cls,msg", [
    (RuntimeError, "API service unavailable"),
    (TimeoutError, "Request timed out after 30s"),
    (ConnectionError, "Remote host refused connection"),
])
def test_heal_dialogue_propagates_client_exceptions(healer, mock_client, dummy_contexts, exception_cls, msg):
    """heal_dialogue propagates network, timeout, or runtime exceptions from LLM client unchanged."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]
    mock_client.generate.side_effect = exception_cls(msg)

    with pytest.raises(exception_cls, match=msg):
        healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)