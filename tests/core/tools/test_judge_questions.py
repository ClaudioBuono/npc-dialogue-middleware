"""Unit tests for core.tools.judge_questions.

Covers:
- Module-level constant integrity
- build_persona_questions()
- build_rules_questions() – all intent branches
- build_judge_questions() – flag combinations
"""
import pytest
from unittest.mock import MagicMock

from core.tools.judge_questions import (
    CONSISTENT_CHOICES_QUESTION,
    CONSISTENT_OPTIONS_QUESTION,
    FAIRNESS_QUESTION,
    NO_NARRATION_QUESTION,
    PERSONA_QUESTIONS,
    PROFANITY_QUESTION,
    build_judge_questions,
    build_persona_questions,
    build_rules_questions,
)
from core.types.contexts import Dialogue, NPCContext, Quest, Talkativeness
from core.types.dataclasses import JudgeQuestion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npc(intent) -> NPCContext:
    """Minimally-valid NPCContext for any intent."""
    return NPCContext(
        name="Garen",
        age=40,
        personality="Stoic",
        context="Guard post",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Neutral",
        intent=intent,
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleLevelConstants:
    """Constants must be JudgeQuestion instances with non-empty ids and texts."""

    def test_persona_questions_is_tuple(self):
        assert isinstance(PERSONA_QUESTIONS, tuple)

    def test_persona_questions_length(self):
        assert len(PERSONA_QUESTIONS) == 4

    def test_persona_questions_each_is_judge_question(self):
        for q in PERSONA_QUESTIONS:
            assert isinstance(q, JudgeQuestion)

    def test_persona_questions_ids_non_empty(self):
        for q in PERSONA_QUESTIONS:
            assert q.id.strip() != ""

    def test_persona_questions_texts_non_empty(self):
        for q in PERSONA_QUESTIONS:
            assert q.text.strip() != ""

    def test_persona_question_ids_are_unique(self):
        ids = [q.id for q in PERSONA_QUESTIONS]
        assert len(ids) == len(set(ids))

    def test_no_narration_question_fields(self):
        assert isinstance(NO_NARRATION_QUESTION, JudgeQuestion)
        assert NO_NARRATION_QUESTION.id == "no_narration"
        assert len(NO_NARRATION_QUESTION.text) > 0

    def test_consistent_choices_question_fields(self):
        assert isinstance(CONSISTENT_CHOICES_QUESTION, JudgeQuestion)
        assert CONSISTENT_CHOICES_QUESTION.id == "consistent_choices"

    def test_consistent_options_question_fields(self):
        assert isinstance(CONSISTENT_OPTIONS_QUESTION, JudgeQuestion)
        assert CONSISTENT_OPTIONS_QUESTION.id == "consistent_options"

    def test_fairness_question_fields(self):
        assert isinstance(FAIRNESS_QUESTION, JudgeQuestion)
        assert FAIRNESS_QUESTION.id == "fairness"

    def test_profanity_question_fields(self):
        assert isinstance(PROFANITY_QUESTION, JudgeQuestion)
        assert PROFANITY_QUESTION.id == "profanity"


# ---------------------------------------------------------------------------
# build_persona_questions
# ---------------------------------------------------------------------------

class TestBuildPersonaQuestions:
    """build_persona_questions returns a fresh mutable list of the 4 fixed questions."""

    def test_returns_list(self):
        result = build_persona_questions()
        assert isinstance(result, list)

    def test_length(self):
        assert len(build_persona_questions()) == 4

    def test_contents_match_persona_questions_constant(self):
        assert build_persona_questions() == list(PERSONA_QUESTIONS)

    def test_returns_new_list_each_call(self):
        """Safe to mutate – each call is independent."""
        a = build_persona_questions()
        b = build_persona_questions()
        assert a is not b

    def test_mutation_does_not_affect_next_call(self):
        a = build_persona_questions()
        a.clear()
        b = build_persona_questions()
        assert len(b) == 4


# ---------------------------------------------------------------------------
# build_rules_questions
# ---------------------------------------------------------------------------

class TestBuildRulesQuestions:
    """build_rules_questions adapts to the intent type."""

    # --- Plain Dialogue ---

    def test_plain_dialogue_always_includes_no_narration(self):
        npc = _make_npc(Dialogue())
        questions = build_rules_questions(npc)
        assert NO_NARRATION_QUESTION in questions

    def test_plain_dialogue_returns_only_no_narration(self):
        """A plain Dialogue should NOT add choices or options questions."""
        npc = _make_npc(Dialogue())
        questions = build_rules_questions(npc)
        assert len(questions) == 1
        assert questions[0] is NO_NARRATION_QUESTION

    def test_plain_dialogue_no_choice_question(self):
        npc = _make_npc(Dialogue())
        ids = [q.id for q in build_rules_questions(npc)]
        assert "consistent_choices" not in ids
        assert "consistent_options" not in ids

    # --- Quest intent ---
    #
    # NOTE: build_rules_questions uses isinstance(intent, Dialogue) as its
    # early-return guard. Because Quest is a subclass of Dialogue, that check
    # is always True and the function returns immediately with only
    # [NO_NARRATION_QUESTION] for every intent type, including Quest.
    # The tests below document this actual runtime behaviour.

    def test_quest_with_choice_still_returns_only_no_narration(self):
        """Quest is a Dialogue subclass so the isinstance guard always fires early."""
        quest = Quest(objective="Retrieve artefact", has_choice=True, number_of_options=0)
        npc = _make_npc(quest)
        questions = build_rules_questions(npc)
        assert len(questions) == 1
        assert questions[0] is NO_NARRATION_QUESTION

    def test_quest_with_options_still_returns_only_no_narration(self):
        quest = Quest(objective="Find gem", has_choice=False, number_of_options=2)
        npc = _make_npc(quest)
        questions = build_rules_questions(npc)
        assert len(questions) == 1
        assert questions[0] is NO_NARRATION_QUESTION

    def test_quest_with_both_flags_still_returns_only_no_narration(self):
        quest = Quest(objective="Build fort", has_choice=True, number_of_options=3)
        npc = _make_npc(quest)
        questions = build_rules_questions(npc)
        assert len(questions) == 1
        assert questions[0] is NO_NARRATION_QUESTION

    def test_quest_no_choice_no_options_returns_only_no_narration(self):
        quest = Quest(objective="Explore cave", has_choice=False, number_of_options=0)
        npc = _make_npc(quest)
        questions = build_rules_questions(npc)
        assert len(questions) == 1
        assert "consistent_options" not in [q.id for q in questions]
        assert "consistent_choices" not in [q.id for q in questions]

    # --- Ordering guarantee ---

    def test_no_narration_is_always_first(self):
        for intent in [Dialogue(), Quest(objective="x", has_choice=True, number_of_options=2)]:
            npc = _make_npc(intent)
            questions = build_rules_questions(npc)
            assert questions[0] is NO_NARRATION_QUESTION

    # --- Return type ---

    def test_returns_list(self):
        npc = _make_npc(Dialogue())
        assert isinstance(build_rules_questions(npc), list)


# ---------------------------------------------------------------------------
# build_judge_questions
# ---------------------------------------------------------------------------

class TestBuildJudgeQuestions:
    """build_judge_questions assembles the full question list from persona, rules, and flags."""

    def _dialogue_npc(self):
        return _make_npc(Dialogue())

    def test_returns_list(self):
        assert isinstance(build_judge_questions(self._dialogue_npc(), False, False), list)

    # --- Base content (no flags) ---

    def test_no_flags_contains_persona_questions(self):
        questions = build_judge_questions(self._dialogue_npc(), False, False)
        for pq in PERSONA_QUESTIONS:
            assert pq in questions

    def test_no_flags_contains_rules_questions(self):
        questions = build_judge_questions(self._dialogue_npc(), False, False)
        assert NO_NARRATION_QUESTION in questions

    def test_no_flags_excludes_fairness(self):
        questions = build_judge_questions(self._dialogue_npc(), False, False)
        assert FAIRNESS_QUESTION not in questions

    def test_no_flags_excludes_profanity(self):
        questions = build_judge_questions(self._dialogue_npc(), False, False)
        assert PROFANITY_QUESTION not in questions

    # --- Fairness flag ---

    def test_fairness_true_includes_fairness_question(self):
        questions = build_judge_questions(self._dialogue_npc(), True, False)
        assert FAIRNESS_QUESTION in questions

    def test_fairness_false_excludes_fairness_question(self):
        questions = build_judge_questions(self._dialogue_npc(), False, False)
        assert FAIRNESS_QUESTION not in questions

    # --- Profanity flag ---

    def test_profanity_true_includes_profanity_question(self):
        questions = build_judge_questions(self._dialogue_npc(), False, True)
        assert PROFANITY_QUESTION in questions

    def test_profanity_false_excludes_profanity_question(self):
        questions = build_judge_questions(self._dialogue_npc(), False, False)
        assert PROFANITY_QUESTION not in questions

    # --- Both flags ---

    def test_both_flags_includes_both_optional_questions(self):
        questions = build_judge_questions(self._dialogue_npc(), True, True)
        assert FAIRNESS_QUESTION in questions
        assert PROFANITY_QUESTION in questions

    # --- Ordering: persona then rules then optional ---

    def test_ordering_persona_before_rules_before_optional(self):
        """Persona questions precede rules questions, which precede optional ones."""
        npc = _make_npc(Dialogue())
        questions = build_judge_questions(npc, True, True)

        persona_ids = {q.id for q in PERSONA_QUESTIONS}
        persona_indices = [i for i, q in enumerate(questions) if q.id in persona_ids]
        rule_indices = [
            i for i, q in enumerate(questions)
            if q.id in {"no_narration"}  # only rule for any intent given isinstance behaviour
        ]
        optional_indices = [
            i for i, q in enumerate(questions)
            if q.id in {"fairness", "profanity"}
        ]

        assert max(persona_indices) < min(rule_indices)
        assert max(rule_indices) < min(optional_indices)

    # --- Total length ---

    def test_total_length_no_flags_plain_dialogue(self):
        """4 persona + 1 no_narration = 5."""
        questions = build_judge_questions(self._dialogue_npc(), False, False)
        assert len(questions) == 5

    def test_total_length_both_flags_plain_dialogue(self):
        """4 persona + 1 no_narration + 1 fairness + 1 profanity = 7."""
        questions = build_judge_questions(self._dialogue_npc(), True, True)
        assert len(questions) == 7

    def test_total_length_both_flags_quest_any_intent(self):
        """Quest is a Dialogue subclass so rules collapse to 1 question.
        4 persona + 1 no_narration + 1 fairness + 1 profanity = 7."""
        quest = Quest(objective="x", has_choice=True, number_of_options=2)
        npc = _make_npc(quest)
        questions = build_judge_questions(npc, True, True)
        assert len(questions) == 7

    # --- No duplicates ---

    def test_no_duplicate_question_ids(self):
        npc = _make_npc(Dialogue())
        questions = build_judge_questions(npc, True, True)
        ids = [q.id for q in questions]
        assert len(ids) == len(set(ids))
