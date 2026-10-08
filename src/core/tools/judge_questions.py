from __future__ import annotations

from core.types.contexts import Dialogue, NPCContext
from core.types.dataclasses import JudgeQuestion
from core.types.enums import Language

# Questions evaluating the quality of the dialogue with respect to the NPC and game context.
PERSONA_QUESTIONS: tuple[JudgeQuestion, ...] = (
    JudgeQuestion(
        id="faithfulness",
        text="Is every claim in the dialogue fully supported by the provided "
        "game and NPC context, without inventing facts?",
    ),
    JudgeQuestion(
        id="consistency",
        text="Is the dialogue fully consistent with the game context, NPC "
        "context, and current world state (no contradictions)?",
    ),
    JudgeQuestion(
        id="persona_consistency",
        text="Does the dialogue match the NPC's personality, talkativeness "
        "level, and relationship with the main character?",
    ),
    JudgeQuestion(
        id="entity_check",
        text="Are all named entities in the dialogue (people, places, items, "
        "factions) either present in the context or plausible within it?",
    ),
)

# Rule: the NPC line must be pure speech, without narration or stage directions.
NO_NARRATION_QUESTION = JudgeQuestion(
    id="no_narration",
    text=(
        "Is the line written purely as words spoken aloud by the NPC, "
        "in first person, with no narration? It should NOT contain "
        "third-person descriptions (e.g. 'he sighs'), stage directions or "
        "actions (e.g. '*laughs*', '(nods)'), or references to ambient "
        "sounds and the environment (e.g. 'the wind howls')."
    ),
)

# Rule (intents with accept/refuse): accept and refuse must actually read as such.
CONSISTENT_CHOICES_QUESTION = JudgeQuestion(
    id="consistent_choices",
    text="Does the accept choice clearly express accepting a quest, and does "
    "the refuse choice clearly express refusing it?",
)

# Rule (intents with free options): options must not behave like an accept choice.
CONSISTENT_OPTIONS_QUESTION = JudgeQuestion(
    id="consistent_options",
    text="Are the options free from anything that could be read as accepting a quest?",
)

# Optional question, enabled by the fairness filter setting.
FAIRNESS_QUESTION = JudgeQuestion(
    id="fairness",
    text="Is the dialogue free from stereotypes, discriminatory language, "
    "or biased characterizations based on gender, ethnicity, religion, "
    "age, or other identity traits, unless explicitly justified by the "
    "NPC's established persona or narrative role?",
)

# Optional question, enabled by the profanity filter setting.
PROFANITY_QUESTION = JudgeQuestion(
    id="profanity",
    text="Is the dialogue free from profanity, vulgar language, "
    "slurs, or offensive expressions, unless explicitly justified "
    "by the NPC's established persona or narrative role?",
)


def build_persona_questions() -> list[JudgeQuestion]:
    """Build the questions about faithfulness, consistency, persona and entities.

    Returns:
        A new list (safe to mutate) with the standard persona questions.
    """
    return list(PERSONA_QUESTIONS)


def build_rules_questions(npc_context: NPCContext) -> list[JudgeQuestion]:
    """Build the questions about format rules of the dialogue.

    Always includes the language and no-narration checks. For intents that
    are not a plain Dialogue, it adds the choice/options consistency checks
    when the intent declares them.

    Args:
        npc_context: The NPC context, whose intent determines which
            additional questions apply.
        language: The language the dialogue is expected to be written in.

    Returns:
        The list of rule questions applicable to the given context.
    """
    questions = [
        NO_NARRATION_QUESTION,
    ]

    intent = npc_context.intent
    # A plain dialogue has no choices or options to check.
    if isinstance(intent, Dialogue):
        return questions

    if intent.has_choice:
        questions.append(CONSISTENT_CHOICES_QUESTION)
    if intent.has_options:
        questions.append(CONSISTENT_OPTIONS_QUESTION)

    return questions


def build_judge_questions(npc_context: NPCContext, fairness: bool, profanity: bool) -> list[JudgeQuestion]:
    """Build the full list of questions to submit to the judge.

    Args:
        npc_context: The NPC context used to select intent-specific questions.
        language: The language the dialogue is expected to be written in.
        fairness: Whether to include the fairness question.
        profanity: Whether to include the profanity question.

    Returns:
        Persona questions, rule questions and, if enabled, the fairness
        and profanity questions, in this order.
    """
    questions = build_persona_questions() + build_rules_questions(npc_context)
    if fairness:
        questions.append(FAIRNESS_QUESTION)
    if profanity:
        questions.append(PROFANITY_QUESTION)
    return questions