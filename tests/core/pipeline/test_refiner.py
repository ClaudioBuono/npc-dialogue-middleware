import pytest
from unittest.mock import MagicMock, call
from api.schemas import ComposedDialogue, QuestChoiceSchema
from core.configuration.settings import AppSettings
from core.pipeline import refiner as refiner_module
from core.pipeline.healer import Healer
from core.pipeline.judger import Judger
from core.pipeline.refiner import Refiner
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import MiddlewareError, MiddlewareErrorCode, PreProcessingError, ValidationErrorCode
from core.types.contexts import Dialogue, GameContext, NPCContext, Quest, Talkativeness
from core.types.dataclasses import JudgeIssue


@pytest.fixture
def mock_healer():
    return MagicMock(spec=Healer)


@pytest.fixture
def mock_judger():
    return MagicMock(spec=Judger)


@pytest.fixture
def refiner(mock_healer, mock_judger):
    return Refiner(healer=mock_healer, judger=mock_judger)


@pytest.fixture
def dummy_game_context():
    return GameContext(
        environment="Town Square",
        epoch="Medieval Fantasy",
        world_state="Peaceful day",
        main_character_description="A traveling rogue",
    )


@pytest.fixture
def dummy_npc_context():
    return NPCContext(
        name="Garrick",
        age="45",
        personality="Gruff blacksmith",
        context="At his forge",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Neutral customer",
        intent=Dialogue(type="Dialogue", has_options=False),
    )


@pytest.fixture
def sample_dialogue():
    return ComposedDialogue(
        intent=Dialogue(type="Dialogue", has_options=False),
        dialogue="Welcome to my smithy. What do you need?",
    )


# --- Initialization & Client Injection Tests ---

def test_refiner_initialization(mock_healer, mock_judger):
    """Refiner stores references to the provided Healer and Judger."""
    ref = Refiner(healer=mock_healer, judger=mock_judger)
    assert ref.healer is mock_healer
    assert ref.judger is mock_judger


def test_refiner_set_client(refiner, mock_healer, mock_judger):
    """set_client propagates the OpenAI-compatible client to both Healer and Judger."""
    mock_client = MagicMock(spec=OpenAICompatibleClient)
    refiner.set_client(mock_client)

    mock_healer.set_client.assert_called_once_with(mock_client)
    mock_judger.set_client.assert_called_once_with(mock_client)


# --- Happy Path Execution Tests ---

def test_refine_dialogue_clean_on_first_iteration(
    refiner, mock_healer, mock_judger, sample_dialogue, dummy_npc_context, dummy_game_context
):
    """When Judger returns no issues on iteration 1, the original dialogue is returned immediately."""
    mock_judger.judge_dialogue.return_value = []

    result = refiner.refine_dialogue(sample_dialogue, dummy_npc_context, dummy_game_context)

    assert result == sample_dialogue
    mock_judger.judge_dialogue.assert_called_once_with(sample_dialogue, dummy_npc_context, dummy_game_context)
    mock_healer.heal_dialogue.assert_not_called()


def test_refine_dialogue_repaired_on_second_iteration(
    refiner, mock_healer, mock_judger, sample_dialogue, dummy_npc_context, dummy_game_context
):
    """When Judger flags issues on iteration 1, Healer fixes it, and Judger accepts it on iteration 2."""
    issue_1 = JudgeIssue(category="consistency", issue="Contradicts game context")
    healed_dialogue = ComposedDialogue(
        intent=sample_dialogue.intent,
        dialogue="Welcome to my smithy, stranger.",
    )

    # First judge finds issues, second judge accepts healed dialogue
    mock_judger.judge_dialogue.side_effect = [[issue_1], []]
    mock_healer.heal_dialogue.return_value = healed_dialogue

    result = refiner.refine_dialogue(sample_dialogue, dummy_npc_context, dummy_game_context)

    assert result == healed_dialogue
    assert mock_judger.judge_dialogue.call_count == 2
    mock_judger.judge_dialogue.assert_has_calls([
        call(sample_dialogue, dummy_npc_context, dummy_game_context),
        call(healed_dialogue, dummy_npc_context, dummy_game_context),
    ])
    mock_healer.heal_dialogue.assert_called_once_with(
        sample_dialogue, dummy_game_context, dummy_npc_context, [issue_1]
    )


def test_refine_dialogue_repaired_on_last_iteration(
    monkeypatch, refiner, mock_healer, mock_judger, sample_dialogue, dummy_npc_context, dummy_game_context
):
    """When dialogue passes on exactly max_iterations, it succeeds and returns the latest healed version."""
    max_iter = 3
    app_settings = AppSettings(refiner_max_iterations=max_iter)
    monkeypatch.setattr(refiner_module, "Settings", lambda: app_settings)

    issue_1 = JudgeIssue(category="faithfulness", issue="Fabricated detail")
    issue_2 = JudgeIssue(category="persona_consistency", issue="Out of character")
    healed_v1 = ComposedDialogue(intent=sample_dialogue.intent, dialogue="Healed once")
    healed_v2 = ComposedDialogue(intent=sample_dialogue.intent, dialogue="Healed twice")

    mock_judger.judge_dialogue.side_effect = [[issue_1], [issue_2], []]
    mock_healer.heal_dialogue.side_effect = [healed_v1, healed_v2]

    result = refiner.refine_dialogue(sample_dialogue, dummy_npc_context, dummy_game_context)

    assert result == healed_v2
    assert mock_judger.judge_dialogue.call_count == 3
    assert mock_healer.heal_dialogue.call_count == 2
    mock_healer.heal_dialogue.assert_has_calls([
        call(sample_dialogue, dummy_game_context, dummy_npc_context, [issue_1]),
        call(healed_v1, dummy_game_context, dummy_npc_context, [issue_2]),
    ])


# --- Iteration Exhaustion & Failure Tests ---

def test_refine_dialogue_exhausts_max_iterations_raises_middleware_error(
    monkeypatch, refiner, mock_healer, mock_judger, sample_dialogue, dummy_npc_context, dummy_game_context
):
    """When Judger continually returns issues, Refiner exhausts max_iterations and raises MiddlewareError.REFUSED."""
    max_iter = 3
    app_settings = AppSettings(refiner_max_iterations=max_iter)
    monkeypatch.setattr(refiner_module, "Settings", lambda: app_settings)

    persistent_issue = JudgeIssue(category="faithfulness", issue="Persistent hallucination")
    mock_judger.judge_dialogue.return_value = [persistent_issue]

    healed_v1 = ComposedDialogue(intent=sample_dialogue.intent, dialogue="Attempt 1")
    healed_v2 = ComposedDialogue(intent=sample_dialogue.intent, dialogue="Attempt 2")
    mock_healer.heal_dialogue.side_effect = [healed_v1, healed_v2]

    with pytest.raises(MiddlewareError) as exc_info:
        refiner.refine_dialogue(sample_dialogue, dummy_npc_context, dummy_game_context)

    assert exc_info.value.code == MiddlewareErrorCode.REFUSED
    assert exc_info.value.errors == ["Could not refine the dialogue."]
    # For N max_iterations: N judge calls, and N - 1 heal calls (breaks on last iteration)
    assert mock_judger.judge_dialogue.call_count == max_iter
    assert mock_healer.heal_dialogue.call_count == max_iter - 1


# --- Boundary Values Tests ---

def test_refine_dialogue_boundary_max_iterations_one_success(
    monkeypatch, refiner, mock_healer, mock_judger, sample_dialogue, dummy_npc_context, dummy_game_context
):
    """Boundary: max_iterations=1 with immediate pass returns dialogue without healing."""
    app_settings = AppSettings(refiner_max_iterations=1)
    monkeypatch.setattr(refiner_module, "Settings", lambda: app_settings)

    mock_judger.judge_dialogue.return_value = []

    result = refiner.refine_dialogue(sample_dialogue, dummy_npc_context, dummy_game_context)

    assert result == sample_dialogue
    assert mock_judger.judge_dialogue.call_count == 1
    mock_healer.heal_dialogue.assert_not_called()


def test_refine_dialogue_boundary_max_iterations_one_failure(
    monkeypatch, refiner, mock_healer, mock_judger, sample_dialogue, dummy_npc_context, dummy_game_context
):
    """Boundary: max_iterations=1 with issues breaks immediately without any healing attempt."""
    app_settings = AppSettings(refiner_max_iterations=1)
    monkeypatch.setattr(refiner_module, "Settings", lambda: app_settings)

    mock_judger.judge_dialogue.return_value = [JudgeIssue(category="fairness", issue="Unfair comment")]

    with pytest.raises(MiddlewareError) as exc_info:
        refiner.refine_dialogue(sample_dialogue, dummy_npc_context, dummy_game_context)

    assert exc_info.value.code == MiddlewareErrorCode.REFUSED
    assert mock_judger.judge_dialogue.call_count == 1
    # On iteration 1 (which equals max_iterations), break is executed, so healer is never invoked
    mock_healer.heal_dialogue.assert_not_called()


def test_refine_dialogue_boundary_max_iterations_zero_raises(
    monkeypatch, refiner, mock_healer, mock_judger, sample_dialogue, dummy_npc_context, dummy_game_context
):
    """Boundary: max_iterations=0 executes 0 iterations and raises MiddlewareError."""
    app_settings = AppSettings(refiner_max_iterations=0)
    monkeypatch.setattr(refiner_module, "Settings", lambda: app_settings)

    with pytest.raises(MiddlewareError) as exc_info:
        refiner.refine_dialogue(sample_dialogue, dummy_npc_context, dummy_game_context)

    assert exc_info.value.code == MiddlewareErrorCode.REFUSED
    mock_judger.judge_dialogue.assert_not_called()
    mock_healer.heal_dialogue.assert_not_called()


# --- Error Handling & Exception Propagation Tests ---

def test_refine_dialogue_propagates_judger_exception(
    refiner, mock_healer, mock_judger, sample_dialogue, dummy_npc_context, dummy_game_context
):
    """Exceptions raised by Judger propagate immediately and halt refinement."""
    mock_judger.judge_dialogue.side_effect = RuntimeError("Judger connection timed out")

    with pytest.raises(RuntimeError, match="Judger connection timed out"):
        refiner.refine_dialogue(sample_dialogue, dummy_npc_context, dummy_game_context)

    mock_healer.heal_dialogue.assert_not_called()


def test_refine_dialogue_propagates_healer_exception(
    refiner, mock_healer, mock_judger, sample_dialogue, dummy_npc_context, dummy_game_context
):
    """Exceptions raised by Healer propagate immediately and abort remaining iterations."""
    mock_judger.judge_dialogue.return_value = [JudgeIssue(category="faithfulness", issue="Bad fact")]
    mock_healer.heal_dialogue.side_effect = PreProcessingError(
        code=ValidationErrorCode.INVALID_VALUE,
        errors=["Healer schema validation failed"],
    )

    with pytest.raises(PreProcessingError) as exc_info:
        refiner.refine_dialogue(sample_dialogue, dummy_npc_context, dummy_game_context)

    assert exc_info.value.code == ValidationErrorCode.INVALID_VALUE
    assert "Healer schema validation failed" in exc_info.value.errors[0]
    assert mock_judger.judge_dialogue.call_count == 1
    assert mock_healer.heal_dialogue.call_count == 1


# --- Quest Intent & Multiple Issues Payload Tests ---

def test_refine_dialogue_with_quest_intent_and_choice_options(
    refiner, mock_healer, mock_judger, dummy_game_context
):
    """Refinement works properly when the dialogue has a Quest intent and QuestChoiceSchema."""
    quest_intent = Quest(
        type="Quest",
        objective="Slay the dragon",
        name="Dragon Slayer",
        description="A fierce dragon threatens the village",
        has_choice=True,
    )
    npc_context = NPCContext(
        name="Elder Thomas",
        age="70",
        personality="Wise village elder",
        context="Village council hall",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Revered champion",
        intent=quest_intent,
    )
    quest_dialogue = ComposedDialogue(
        intent=quest_intent,
        dialogue="Please, help us slay the dragon!",
        player_options=QuestChoiceSchema(
            accept="I will slay the beast.",
            refuse="Find someone else.",
            dialogue_options=["Where was it last seen?"],
        ),
    )

    issue = JudgeIssue(category="Accept/Refuse", issue="Accept option is not heroic enough")
    healed_quest = ComposedDialogue(
        intent=quest_intent,
        dialogue="Please, help us slay the terrifying beast!",
        player_options=QuestChoiceSchema(
            accept="By my sword, the dragon shall fall!",
            refuse="Find someone else.",
            dialogue_options=["Where was it last seen?"],
        ),
    )

    mock_judger.judge_dialogue.side_effect = [[issue], []]
    mock_healer.heal_dialogue.return_value = healed_quest

    result = refiner.refine_dialogue(quest_dialogue, npc_context, dummy_game_context)

    assert result == healed_quest
    mock_healer.heal_dialogue.assert_called_once_with(
        quest_dialogue, dummy_game_context, npc_context, [issue]
    )
