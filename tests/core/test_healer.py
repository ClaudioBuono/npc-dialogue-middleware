import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel, ValidationError

from api.schemas import ComposedDialogue
from core.config.settings import Settings
from core.contract_builder import ContractBuilder
from core.healer import Healer
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import PreProcessingError, ValidationErrorCode
from core.tools.output_composer import DialogueOutputComposer
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest
from core.types.dataclasses import JudgeIssue, Contract

@pytest.fixture
def mock_contract_builder():
    builder = MagicMock(spec=ContractBuilder)
    builder.build_healer_contract.return_value = Contract(system_prompt="sys", user_prompt="user", output_schema={})
    return builder

@pytest.fixture
def mock_dialogue_composer():
    composer = MagicMock(spec=DialogueOutputComposer)
    composer.compose_dialogue.return_value = ComposedDialogue(intent = Dialogue(), dialogue="Healed dialogue")
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
    npc_ctx.intent = Quest(
        objective="Test",
        name="Test",
        description="Test",
        has_choice=False
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

def test_heal_dialogue_success(healer, mock_contract_builder, mock_dialogue_composer, mock_client, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent = Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]

    result = healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)
    
    assert result.dialogue == "Healed dialogue"
    mock_contract_builder.build_healer_contract.assert_called_once_with(original_dialogue, game_ctx, npc_ctx, issues)
    mock_client.generate.assert_called_once()
    mock_dialogue_composer.compose_dialogue.assert_called_once_with(npc_ctx, '{"dialogue": "Healed dialogue"}')

def test_heal_dialogue_calls_client_with_correct_args(healer, mock_contract_builder, mock_client, dummy_contexts):
    """Pin down exactly what is passed to client.generate(): the built
    contract and the configured temperature. A previous assert only checked
    that generate() was called once, not with what."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]

    healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    expected_contract = mock_contract_builder.build_healer_contract.return_value
    mock_client.generate.assert_called_once_with(expected_contract, temperature=Settings().llm.temperature)

def test_heal_dialogue_with_no_issues(healer, mock_contract_builder, mock_dialogue_composer, dummy_contexts):
    """An empty issues list is a legitimate edge case (e.g. called
    defensively even when the judge found nothing) and must not break."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")

    result = healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, [])

    mock_contract_builder.build_healer_contract.assert_called_once_with(original_dialogue, game_ctx, npc_ctx, [])
    assert result.dialogue == "Healed dialogue"

def test_heal_dialogue_with_multiple_issues(healer, mock_contract_builder, dummy_contexts):
    """A list with several issues must be forwarded to the contract builder
    intact (not truncated, deduplicated, or reordered)."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [
        JudgeIssue(category="faithfulness", issue="Bad fact"),
        JudgeIssue(category="entity_check", issue="Unknown entity"),
        JudgeIssue(category="Accept/Refuse", issue="Missing Accept/Refuse options"),
    ]

    healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    mock_contract_builder.build_healer_contract.assert_called_once_with(original_dialogue, game_ctx, npc_ctx, issues)

def test_heal_dialogue_validation_error(healer, mock_dialogue_composer, mock_client, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent = Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]

    # Simulate validation error from compose_dialogue
    mock_dialogue_composer.compose_dialogue.side_effect = make_validation_error()

    with pytest.raises(PreProcessingError) as exc_info:
        healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)
    
    assert exc_info.value.code == ValidationErrorCode.INVALID_VALUE

def test_heal_dialogue_validation_error_includes_raw_output(healer, mock_dialogue_composer, mock_client, dummy_contexts):
    """The error payload should preserve the raw LLM output for debugging;
    only the error code was checked before, not its content."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]
    mock_client.generate.return_value = '{"dialogue": "broken"}'
    mock_dialogue_composer.compose_dialogue.side_effect = make_validation_error()

    with pytest.raises(PreProcessingError) as exc_info:
        healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    assert any('{"dialogue": "broken"}' in err for err in exc_info.value.errors)

def test_heal_dialogue_validation_error_still_calls_upstream_steps(healer, mock_contract_builder, mock_client, mock_dialogue_composer, dummy_contexts):
    """When compose_dialogue fails, the earlier steps (contract building,
    LLM generation) must still have run exactly once before the failure --
    the error happens only at the final parsing step."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]
    mock_dialogue_composer.compose_dialogue.side_effect = make_validation_error()

    with pytest.raises(PreProcessingError):
        healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)

    mock_contract_builder.build_healer_contract.assert_called_once()
    mock_client.generate.assert_called_once()

def test_heal_dialogue_propagates_client_errors(healer, mock_client, dummy_contexts):
    """heal_dialogue does not catch errors raised by the LLM client itself
    (e.g. network/API failures); they must propagate unchanged. This pins
    down current behavior so a future broad try/except doesn't silently
    swallow them without a test noticing."""
    game_ctx, npc_ctx = dummy_contexts
    original_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Original dialogue")
    issues = [JudgeIssue(category="faithfulness", issue="Bad fact")]
    mock_client.generate.side_effect = RuntimeError("API down")

    with pytest.raises(RuntimeError):
        healer.heal_dialogue(original_dialogue, game_ctx, npc_ctx, issues)