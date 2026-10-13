import json

import pytest
from unittest.mock import ANY, MagicMock
from api.schemas import ComposedDialogue, QuestChoiceSchema
from core.configuration.settings import AppSettings, Settings
from core.generation.contract_builder import ContractBuilder
from core.pipeline.guardrail import Guardrail
from core.pipeline.judger import Judger
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import PreProcessingError, MiddlewareError, MiddlewareErrorCode, ValidationErrorCode
from core.tools.judge_questions import build_judge_questions
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest
from core.types.dataclasses import Contract
from core.types.enums import Language

dummy_number_of_options = 3
BASE_SETTINGS = dict(fairness_filter=False, profanity_filter=False, language=Language.ENGLISH)
COMPOSED = lambda text="Test": ComposedDialogue(intent=Dialogue(), dialogue=text)

# --- settings ---------------------------------------------------------------
@pytest.fixture
def set_settings(monkeypatch):
    def _set(**overrides):
        app = AppSettings(**{**BASE_SETTINGS, **overrides})
        monkeypatch.setattr("core.pipeline.judger.Settings", lambda: app)
        return app
    return _set

@pytest.fixture(autouse=True)
def _baseline_settings(set_settings):
    set_settings()

# --- mocks ------------------------------------------------------------------
@pytest.fixture
def mock_contract_builder():
    builder = MagicMock(spec=ContractBuilder)
    builder.build_judge_contract.return_value = Contract(system_prompt="sys", user_prompt="user", output_schema={})
    return builder

def _judge_response(builder, failing=None):
    """Judge response build by questions actually answered by the judge."""
    failing = failing or {}
    def _generate(contract, temperature):
        questions = builder.build_judge_contract.call_args.args[3]
        return json.dumps({"answers": [
            {"id": q.id, "answer": q.id not in failing, "reason": failing.get(q.id, "Condition satisfied")}
            for q in questions
        ]})
    return _generate

@pytest.fixture
def mock_client(mock_contract_builder):
    client = MagicMock(spec=OpenAICompatibleClient)
    client.generate.side_effect = _judge_response(mock_contract_builder)   # default: tutto true
    return client

@pytest.fixture
def judge_answers(mock_client, mock_contract_builder):
    """judge_answers({"entity_check": "reason with at least 15 characters"}) makes that questions fail."""
    def _configure(failing=None):
        mock_client.generate.side_effect = _judge_response(mock_contract_builder, failing)
    return _configure

@pytest.fixture
def mock_guardrail():
    g = MagicMock(spec=Guardrail)
    g.retrieve_banned_words_in_composed_dialogue.return_value = []
    return g

@pytest.fixture
def judger(mock_contract_builder, mock_guardrail, mock_client):
    j = Judger(contract_builder=mock_contract_builder, guardrail=mock_guardrail)
    j.set_client(mock_client)
    return j

@pytest.fixture
def dummy_contexts():
    game_ctx = GameContext(epoch="Test", environment="Test", world_state="Test")
    npc_ctx = MagicMock(spec=NPCContext)
    npc_ctx.name = "Test"          # serve a _check_language
    npc_ctx.intent = Quest(objective="Test", name="Test", description="Test", has_choice=False)
    return game_ctx, npc_ctx

def test_judge_dialogue_success(judger, mock_client, judge_answers, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Test")

    judge_answers({"entity_check": "The entity does not exist in the world"})
    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    
    # entity_check is false in mock_client, so it should be mapped to an issue
    assert len(issues) == 1
    assert issues[0].category == "entity_check"
    assert issues[0].issue == "The entity does not exist in the world"
    
def test_judge_dialogue_validation_error(judger, mock_client, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Test")
    
    mock_client.generate.side_effect = lambda *a, **k: "invalid json"
    
    with pytest.raises(PreProcessingError) as exc_info:
        judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    
    assert exc_info.value.code == ValidationErrorCode.INVALID_VALUE

def test_judge_dialogue_missing_questions(judger, mock_client, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Test")
    
    # Missing some expected question IDs from the default set
    mock_client.generate.side_effect = lambda *a, **k: '{"answers": [{"id": "faithfulness", "answer": true, "reason": "Condition satisfied"}]}'
    
    with pytest.raises(MiddlewareError) as exc_info:
        judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
        
    assert exc_info.value.code == MiddlewareErrorCode.INVALID_RESPONSE

def test_judge_dialogue_mismatched_question_ids(judger, mock_client, mock_contract_builder, dummy_contexts):
    """Same number of answers as expected, but the ids don't match the
    known question set (e.g. the judge invented or misspelled an id).
    This must be rejected just like a length mismatch."""
    game_ctx, npc_ctx = dummy_contexts
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Test")

    def _generate(contract, temperature):
        # the questions that the Judger has actually posted (4th argument of build_judge_contract)
        questions = mock_contract_builder.build_judge_contract.call_args.args[3]
        ids = [q.id for q in questions][:-1] + ["UNKNOWN_ID"]   # same number of answers, last id wrong
        return json.dumps({"answers": [
            {"id": i, "answer": True, "reason": "Condition satisfied"} for i in ids
        ]})

    mock_client.generate.side_effect = _generate

    with pytest.raises(MiddlewareError) as exc_info:
        judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)

    assert exc_info.value.code == MiddlewareErrorCode.INVALID_RESPONSE

def test_build_judge_questions_ids(dummy_contexts):
    _, npc_ctx = dummy_contexts
    base = {q.id for q in build_judge_questions(npc_context=npc_ctx, fairness=False, profanity=False)}
    assert base == {"faithfulness", "consistency", "persona_consistency", "entity_check", "no_narration"}  # verifica con il codice reale
    with_fairness = {q.id for q in build_judge_questions(npc_context=npc_ctx, fairness=True, profanity=False)}
    with_profanity = {q.id for q in build_judge_questions(npc_context=npc_ctx, fairness=False, profanity=True)}
    assert with_fairness - base == {"fairness"}
    assert with_profanity - base == {"profanity"}

def test_judge_static_format_must_use_expression(judger, mock_client, dummy_contexts,):
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = "by the gods"
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Hello there")
    
    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert any(i.category == "Must use expression" for i in issues)

def test_judge_static_format_expression_present_no_issue(judger, mock_client, dummy_contexts):
    """Mandatory expression IS present in the dialogue: no issue should be raised."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = "by the gods"
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Well met, by the gods!")

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Must use expression" for i in issues)
    assert issues == []

def test_judge_static_format_no_expression_required_no_issue(judger, mock_client, dummy_contexts):
    """When must_use_expression is falsy/None, the check should never fire,
    regardless of dialogue content."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Anything goes here")

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Must use expression" for i in issues)

def test_judge_static_format_wrong_option_count(judger, mock_client, dummy_contexts):
    """dialogue_options length differs from dummy_number_of_options"""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = None
    npc_ctx.intent.has_choice = False

    expected_count = dummy_number_of_options
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

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert any(i.category == "Number of options" for i in issues)

def test_judge_static_format_correct_option_count_no_issue(judger, mock_client, dummy_contexts):
    """dialogue_options length matches dummy_number_of_options exactly: no issue."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.number_of_options = dummy_number_of_options
    npc_ctx.intent.must_use_expression = None
    npc_ctx.intent.has_choice = False

    expected_count = dummy_number_of_options
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

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Number of options" for i in issues)

def test_judge_static_format_no_dialogue_options_skips_count_check(judger, mock_client, dummy_contexts):
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

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Number of options" for i in issues)

def test_judge_static_format_missing_accept_refuse(judger, mock_client, dummy_contexts):
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
    
    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert any(i.category == "Accept/Refuse" for i in issues)

def test_judge_static_format_accept_refuse_present_no_issue(judger, mock_client, dummy_contexts):
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

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Accept/Refuse" for i in issues)

def test_judge_static_format_no_player_options_with_choice(judger, mock_client, dummy_contexts):
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

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert any(i.category == "Accept/Refuse" for i in issues)

def test_judge_static_format_no_choice_required_no_accept_refuse_issue(judger, mock_client, dummy_contexts):
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

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Accept/Refuse" for i in issues)

def test_judge_static_format_dialogue_intent_without_has_choice(judger, mock_client, dummy_contexts):
    """Regression test: Dialogue (unlike Quest) has no has_choice field.
    _check_accept_refuse must not raise AttributeError when npc_context.intent
    is a plain Dialogue -- it should simply treat it as "no choice required"."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent = Dialogue()  # no has_choice field on this type
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Just some flavor text")

    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)
    assert not any(i.category == "Accept/Refuse" for i in issues)

def test_judge_dialogue_combines_semantic_and_static_issues(judger, mock_client, judge_answers, dummy_contexts):
    """When both an LLM-detected issue and a static-format violation occur,
    both must be present in the result (judge_dialogue must not drop either)."""
    game_ctx, npc_ctx = dummy_contexts
    npc_ctx.intent.must_use_expression = "by the gods"
    npc_ctx.intent.has_choice = False
    composed_dialogue = ComposedDialogue(intent=Dialogue(), dialogue="Hello there")

    judge_answers({"entity_check": "The entity does not exist in the world"})
    issues = judger.judge_dialogue(composed_dialogue, npc_ctx, game_ctx)

    assert len(issues) == 2
    categories = {i.category for i in issues}
    assert categories == {"entity_check", "Must use expression"}

# --- settings flags -> judge questions -----------------------------------
@pytest.mark.parametrize("flag, qid", [
    ("profanity_filter", "profanity"),
    ("fairness_filter", "fairness"),
])
@pytest.mark.parametrize("enabled", [True, False])
def test_settings_flag_controls_judge_question(
    judger, set_settings, mock_contract_builder, dummy_contexts, flag, qid, enabled,
):
    game_ctx, npc_ctx = dummy_contexts
    set_settings(**{flag: enabled})

    judger.judge_dialogue(COMPOSED(), npc_ctx, game_ctx)

    asked = {q.id for q in mock_contract_builder.build_judge_contract.call_args.args[3]}
    assert (qid in asked) == enabled

def test_judger_temperature_comes_from_settings(judger, set_settings, mock_client, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    app = set_settings()
    judger.judge_dialogue(COMPOSED(), npc_ctx, game_ctx)
    mock_client.generate.assert_called_with(ANY, app.llm.judger_temperature)

# --- static checks that are influenced by settings ---------------------------------------
def test_banned_words_reported_when_profanity_filter_on(judger, set_settings, mock_guardrail, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    set_settings(profanity_filter=True)
    mock_guardrail.retrieve_banned_words_in_composed_dialogue.return_value = ["badword"]

    issues = judger.judge_dialogue(COMPOSED(), npc_ctx, game_ctx)

    assert any(i.category == "Banned words" and "badword" in i.issue for i in issues)

def test_banned_words_skipped_when_profanity_filter_off(judger, mock_guardrail, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts   # baseline: profanity_filter=False
    judger.judge_dialogue(COMPOSED(), npc_ctx, game_ctx)
    mock_guardrail.retrieve_banned_words_in_composed_dialogue.assert_not_called()

def test_language_mismatch_reported(judger, monkeypatch, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    # find_foreign_words already has its own tests: here we only verify the integration in Judger
    monkeypatch.setattr("core.pipeline.judger.find_foreign_words", lambda text, expected, ignore: ["ciao"])

    issues = judger.judge_dialogue(COMPOSED(), npc_ctx, game_ctx)

    mismatch = [i for i in issues if i.category == "Language mismatch"]
    assert len(mismatch) == 1
    assert "dialogue" in mismatch[0].issue

@pytest.mark.parametrize("text", ["*sighs* Hello there", "(smiles) Hello there", "[draws sword] Hello there"])
def test_stage_directions_reported(judger, dummy_contexts, text):
    game_ctx, npc_ctx = dummy_contexts
    issues = judger.judge_dialogue(COMPOSED(text), npc_ctx, game_ctx)
    assert any(i.category == "Stage directions" for i in issues)

# --- retry on reasons being too short -----------------------------------------------
def test_judge_retries_when_reason_too_short(judger, mock_client, mock_contract_builder, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    gens = [
        _judge_response(mock_contract_builder, {"entity_check": "bad"}),                       # < 15 characters
        _judge_response(mock_contract_builder, {"entity_check": "The entity does not exist"}),
    ]
    mock_client.generate.side_effect = lambda c, t: gens[min(mock_client.generate.call_count - 1, 1)](c, t)

    issues = judger.judge_dialogue(COMPOSED(), npc_ctx, game_ctx)

    assert mock_client.generate.call_count == 2
    assert 'id="entity_check"' in mock_contract_builder.build_judge_contract.call_args.args[4]  # feedback on 2nd attempt
    assert [i.category for i in issues] == ["entity_check"]

def test_judge_fails_after_max_attempts_with_short_reasons(judger, mock_client, judge_answers, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    judge_answers({"entity_check": "bad"})   # always too short

    with pytest.raises(MiddlewareError) as exc_info:
        judger.judge_dialogue(COMPOSED(), npc_ctx, game_ctx)

    assert exc_info.value.code == MiddlewareErrorCode.INVALID_RESPONSE
    assert mock_client.generate.call_count == 3   # MAX_JUDGE_ATTEMPTS

def test_llm_exception_is_propagated(judger, mock_client, dummy_contexts):
    game_ctx, npc_ctx = dummy_contexts
    mock_client.generate.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        judger.judge_dialogue(COMPOSED(), npc_ctx, game_ctx)