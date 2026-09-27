import pytest
from unittest.mock import MagicMock
from api.schemas import ComposedDialogue, QuestChoiceSchema
from core.config.settings import Settings
from core.contract_builder import ContractBuilder
from core.judger import Judger
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import PreProcessingError, MiddlewareError, MiddlewareErrorCode, ValidationErrorCode
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest
from core.types.dataclasses import Contract

@pytest.fixture
def mock_contract_builder():
    builder = MagicMock(spec=ContractBuilder)
    builder.build_judge_contract.return_value = Contract(system_prompt="sys", user_prompt="user", output_schema={})
    return builder

@pytest.fixture
def mock_client():
    client = MagicMock(spec=OpenAICompatibleClient)
    # Default valid response covering standard questions
    client.generate.return_value = '''{
        "answers": [
            {"id": "faithfulness", "answer": true, "reason": ""},
            {"id": "consistency", "answer": true, "reason": ""},
            {"id": "persona_consistency", "answer": true, "reason": ""},
            {"id": "entity_check", "answer": false, "reason": "Bad entity"},
            {"id": "language", "answer": true, "reason": ""},
            {"id": "fairness", "answer": true, "reason": ""}
        ]
    }'''
    return client

@pytest.fixture
def all_true_response():
    """Reusable fully-positive semantic judge response, so tests can isolate static issues."""
    return '''{
        "answers": [
            {"id": "faithfulness", "answer": true, "reason": ""},
            {"id": "consistency", "answer": true, "reason": ""},
            {"id": "persona_consistency", "answer": true, "reason": ""},
            {"id": "entity_check", "answer": true, "reason": ""},
            {"id": "language", "answer": true, "reason": ""},
            {"id": "fairness", "answer": true, "reason": ""}
        ]
    }'''

@pytest.fixture
def judger(mock_contract_builder, mock_client):
    j = Judger(contract_builder=mock_contract_builder)
    j.set_client(mock_client)
    return j

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

def test_judge_dialogue_success(judger, mock_client, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Test")
     
    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    
    # entity_check is false in mock_client, so it should be mapped to an issue
    assert len(issues) == 1
    assert issues[0].category == "entity_check"
    assert issues[0].issue == "Bad entity"
    
def test_judge_dialogue_validation_error(judger, mock_client, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Test")
    
    mock_client.generate.return_value = "invalid json"
    
    with pytest.raises(PreProcessingError) as exc_info:
        judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    
    assert exc_info.value.code == ValidationErrorCode.INVALID_VALUE

def test_judge_dialogue_missing_questions(judger, mock_client, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Test")
    
    # Missing some expected question IDs from the default set
    mock_client.generate.return_value = '''{
        "answers": [
            {"id": "faithfulness", "answer": true, "reason": ""}
        ]
    }'''
    
    with pytest.raises(MiddlewareError) as exc_info:
        judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
        
    assert exc_info.value.code == MiddlewareErrorCode.INVALID_RESPONSE

def test_judge_dialogue_mismatched_question_ids(judger, mock_client, dummy_contexts):
    """Same number of answers as expected, but the ids don't match the
    known question set (e.g. the judge invented or misspelled an id).
    This must be rejected just like a length mismatch."""
    game_ctx, npc_ctx = dummy_contexts
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Test")

    mock_client.generate.return_value = '''{
        "answers": [
            {"id": "faithfulness", "answer": true, "reason": ""},
            {"id": "consistency", "answer": true, "reason": ""},
            {"id": "persona_consistency", "answer": true, "reason": ""},
            {"id": "entity_check", "answer": true, "reason": ""},
            {"id": "language", "answer": true, "reason": ""},
            {"id": "UNKNOWN_ID", "answer": true, "reason": ""}
        ]
    }'''

    with pytest.raises(MiddlewareError) as exc_info:
        judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)

    assert exc_info.value.code == MiddlewareErrorCode.INVALID_RESPONSE

def test_build_questions_returns_expected_ids(judger):
    """_build_questions is the source of truth for _check_valid_response;
    pin down its ids so a change here doesn't silently break validation."""
    questions = judger._build_questions("English")
    ids = {q.id for q in questions}

    assert ids == {
        "faithfulness",
        "consistency",
        "persona_consistency",
        "entity_check",
        "language",
        "fairness",
    }
    # language question should embed the requested language
    language_question = next(q for q in questions if q.id == "language")
    assert "English" in language_question.text

def test_judge_static_format_must_use_expression(judger, mock_client, dummy_contexts, all_true_response):
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = "by the gods"
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Hello there")
    
    # Override client to return fully positive answers so we only see static issues
    mock_client.generate.return_value = all_true_response
    
    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert any(i.category == "Must use expression" for i in issues)

def test_judge_static_format_expression_present_no_issue(judger, mock_client, dummy_contexts, all_true_response):
    """Mandatory expression IS present in the dialogue: no issue should be raised."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = "by the gods"
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Well met, by the gods!")

    mock_client.generate.return_value = all_true_response

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Must use expression" for i in issues)
    assert issues == []

def test_judge_static_format_no_expression_required_no_issue(judger, mock_client, dummy_contexts, all_true_response):
    """When must_use_expression is falsy/None, the check should never fire,
    regardless of dialogue content."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Anything goes here")

    mock_client.generate.return_value = all_true_response

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Must use expression" for i in issues)

def test_judge_static_format_wrong_option_count(judger, mock_client, dummy_contexts, all_true_response):
    """dialogue_options length differs from Settings().number_of_options."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    npc_ctx.intent.has_choice = False

    expected_count = Settings().number_of_options
    wrong_count_options = ["option"] * (expected_count + 1)

    composed_dialogue = ComposedDialogue(
        intent=Dialogue(),
        dialogue="Test",
        player_options=QuestChoiceSchema(
            accept="",
            refuse="",
            dialogue_options=wrong_count_options,
        ),
    )

    mock_client.generate.return_value = all_true_response

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert any(i.category == "Number of options" for i in issues)

def test_judge_static_format_correct_option_count_no_issue(judger, mock_client, dummy_contexts, all_true_response):
    """dialogue_options length matches Settings().number_of_options exactly: no issue."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    npc_ctx.intent.has_choice = False

    expected_count = Settings().number_of_options
    correct_options = ["option"] * expected_count

    composed_dialogue = ComposedDialogue(
        intent=Dialogue(),
        dialogue="Test",
        player_options=QuestChoiceSchema(
            accept="",
            refuse="",
            dialogue_options=correct_options,
        ),
    )

    mock_client.generate.return_value = all_true_response

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Number of options" for i in issues)

def test_judge_static_format_no_dialogue_options_skips_count_check(judger, mock_client, dummy_contexts, all_true_response):
    """When dialogue_options is empty/falsy, the count check should be skipped
    entirely rather than flagged as incorrect."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    npc_ctx.intent.has_choice = False

    composed_dialogue = ComposedDialogue(
        intent=Dialogue(),
        dialogue="Test",
        player_options=QuestChoiceSchema(
            accept="",
            refuse="",
            dialogue_options=[],
        ),
    )

    mock_client.generate.return_value = all_true_response

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Number of options" for i in issues)

def test_judge_static_format_missing_accept_refuse(judger, mock_client, dummy_contexts, all_true_response):
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    npc_ctx.intent.has_choice = True
    composed_dialogue = ComposedDialogue(
        intent = Quest(
            type="Quest",
            objective="Find the lost sword of Elendil",
            has_choice=True
        ),
        dialogue="Will you help me?",
        player_options=QuestChoiceSchema(
            accept="",
            refuse="",
            dialogue_options=[]
        )
    )
    
    mock_client.generate.return_value = all_true_response
    
    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert any(i.category == "Accept/Refuse" for i in issues)

def test_judge_static_format_accept_refuse_present_no_issue(judger, mock_client, dummy_contexts, all_true_response):
    """has_choice is True and both accept/refuse are populated: no issue expected."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    npc_ctx.intent.has_choice = True
    composed_dialogue = ComposedDialogue(
        intent=Quest(
            type="Quest",
            objective="Find the lost sword of Elendil",
            has_choice=True
        ),
        dialogue="Will you help me?",
        player_options=QuestChoiceSchema(
            accept="I accept",
            refuse="I refuse",
            dialogue_options=[]
        )
    )

    mock_client.generate.return_value = all_true_response

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Accept/Refuse" for i in issues)

def test_judge_static_format_no_player_options_with_choice(judger, mock_client, dummy_contexts, all_true_response):
    """has_choice is True but player_options is None entirely (not just empty
    accept/refuse strings) - should still be flagged."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    npc_ctx.intent.has_choice = True
    composed_dialogue = ComposedDialogue(
        intent=Quest(
            type="Quest",
            objective="Find the lost sword of Elendil",
            has_choice=True
        ),
        dialogue="Will you help me?",
        player_options=None
    )

    mock_client.generate.return_value = all_true_response

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert any(i.category == "Accept/Refuse" for i in issues)

def test_judge_static_format_no_choice_required_no_accept_refuse_issue(judger, mock_client, dummy_contexts, all_true_response):
    """has_choice is False: missing accept/refuse should never be flagged,
    even if player_options is None."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    npc_ctx.intent.has_choice = False
    composed_dialogue = ComposedDialogue(
        intent=Dialogue(),
        dialogue="Just some flavor text",
        player_options=None
    )

    mock_client.generate.return_value = all_true_response

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Accept/Refuse" for i in issues)

def test_judge_dialogue_combines_semantic_and_static_issues(judger, mock_client, dummy_contexts):
    """When both an LLM-detected issue and a static-format violation occur,
    both must be present in the result (judge_dialogue must not drop either)."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = "by the gods"
    npc_ctx.intent.has_choice = False
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Hello there")

    # entity_check fails (semantic) AND must_use_expression is absent (static)
    mock_client.generate.return_value = '''{
        "answers": [
            {"id": "faithfulness", "answer": true, "reason": ""},
            {"id": "consistency", "answer": true, "reason": ""},
            {"id": "persona_consistency", "answer": true, "reason": ""},
            {"id": "entity_check", "answer": false, "reason": "Bad entity"},
            {"id": "language", "answer": true, "reason": ""},
            {"id": "fairness", "answer": true, "reason": ""}
        ]
    }'''

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)

    assert len(issues) == 2
    categories = {i.category for i in issues}
    assert categories == {"entity_check", "Must use expression"}
    