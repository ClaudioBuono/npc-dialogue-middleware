from __future__ import annotations
import logging
import time 
from pydantic import ValidationError
from api.schemas import ComposedDialogue
from core.configuration.settings import Settings
from core.generation.contract_builder import ContractBuilder
from core.helpers.formatters import to_json_format
from core.llm.openai_client import OpenAICompatibleClient
from core.pipeline.guardrail import Guardrail
from core.tools.errors import MiddlewareError, MiddlewareErrorCode, PreProcessingError, ValidationErrorCode
from core.tools.judge_questions import build_judge_questions
from core.types.contexts import NPCContext
from core.types.dataclasses import JudgeIssue, JudgeOutput, JudgeProblem, JudgeQuestion

logger = logging.getLogger(__name__)

MIN_REASON_LENGTH = 15
MAX_JUDGE_ATTEMPTS = 3

class Judger:
    """
    Handles dialogue evaluation and compliance checking against defined game rules and context.

    Uses an LLM client to execute evaluation contracts and generate structured judgments on NPC dialogues.
    """
    def __init__(self, contract_builder: ContractBuilder, guardrail: Guardrail) -> None:
        """Initialize the generator with an optional LLM client.

        Args:
            client: The OpenAI-compatible client used to generate dialogue.
                May be None if no client is configured.
        """
        self.contract_builder = contract_builder
        self.guardrail = guardrail


    def set_client(self, client: OpenAICompatibleClient) -> None:
        self._client = client

    def judge_dialogue(self, composed_dialogue, npc_context, game_context) -> list[JudgeIssue]:
        """Evaluate a dialogue and report the issues found.

        Combines two kinds of checks:
        - LLM-based: a set of yes/no questions (built from the NPC context
          and the language/fairness/profanity settings) is sent to the LLM;
          every question answered False becomes an issue. If a False answer
          has a reason shorter than MIN_REASON_LENGTH, the judge is called
          again (up to MAX_JUDGE_ATTEMPTS) with feedback on the bad reasons.
        - Static: deterministic format checks run directly on the dialogue.

        Args:
            composed_dialogue: The dialogue to be judged.
            npc_context: Contextual information about the NPC speaking the dialogue.
            game_context: Contextual information about the game/world state.

        Returns:
            The list of JudgeIssue found (LLM issues first, then static
            ones). An empty list means the dialogue passed all checks.

        Raises:
            PreProcessingError: If the judge output does not match the
                expected schema (ValidationError during parsing).
            MiddlewareError: If the judge answers do not match the questions
                asked (missing or unexpected question ids), or if the reasons
                are still invalid after MAX_JUDGE_ATTEMPTS attempts.
            Exception: Any exception raised by the LLM client during
                generation is logged and re-raised unchanged.
        """
        judge_questions = build_judge_questions(
            npc_context=npc_context,
            language=Settings().language,
            fairness=Settings().fairness_filter,
            profanity=Settings().profanity_filter,
        )
        logger.info(
            "Judging dialogue: %d questions (%s), language=%s",
            len(judge_questions), ", ".join(q.id for q in judge_questions), Settings().language.name,
        )

        judge_output: JudgeOutput | None = None
        feedback: str | None = None

        for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
            judge_contract = self.contract_builder.build_judge_contract(
                composed_dialogue, game_context, npc_context, judge_questions, feedback,
            )

            try:
                judge_output_raw = self._client.generate(judge_contract, Settings().llm.judger_temperature)  # TODO: Test higher temperatures
            except Exception:
                logger.exception("Judge LLM call failed")
                raise

            try:
                candidate = JudgeOutput.model_validate_json(judge_output_raw)
            except ValidationError as e:
                logger.error(
                    "Judge output does not match schema (%d validation errors). Raw output: %s",
                    e.error_count(), judge_output_raw,
                )
                raise PreProcessingError(
                    code=ValidationErrorCode.INVALID_VALUE,
                    errors=[f"Judge output does not match expected schema: {e}", f"Raw output: {judge_output_raw}"],
                )

            if not self._check_valid_response(candidate, judge_questions):
                expected = {q.id for q in judge_questions}
                received = {a.id for a in candidate.answers}
                logger.error(
                    "Judge response mismatch: missing=%s, unexpected=%s, expected_count=%d, received_count=%d",
                    sorted(expected - received), sorted(received - expected),
                    len(judge_questions), len(candidate.answers),
                )
                raise MiddlewareError(
                    code=MiddlewareErrorCode.INVALID_RESPONSE,
                    errors=["There was a problem during the judging process."],
                )

            bad_reasons = self._check_reasons(candidate)
            if bad_reasons:
                logger.warning(
                    "Judge attempt %d/%d: %d invalid reason(s): %s",
                    attempt, MAX_JUDGE_ATTEMPTS, len(bad_reasons),
                    ", ".join(f"{a.id}={a.reason!r}" for a in bad_reasons),
                )
                feedback = self._build_reason_feedback(bad_reasons)
                continue

            judge_output = candidate
            break

        if judge_output is None:
            logger.error("Judge produced invalid reasons for %d consecutive attempts", MAX_JUDGE_ATTEMPTS)
            raise MiddlewareError(
                code=MiddlewareErrorCode.INVALID_RESPONSE,
                errors=["There was a problem during the judging process."],
            )

        # Extract problems (where the answer is False)
        judge_problems: list[JudgeProblem] = [p for p in judge_output.answers if not p.answer]
        logger.info(
            "LLM judge: %d/%d checks failed%s",
            len(judge_problems), len(judge_output.answers),
            f" ({', '.join(p.id for p in judge_problems)})" if judge_problems else "",
        )

        # Build issues array
        llm_issues = self._map_judge_issues(judge_problems)
        static_issues = self._judge_static_format(composed_dialogue, npc_context)
        issues = llm_issues + static_issues

        if issues:
            logger.info(
                "Judgment: %d issue(s) (llm=%d, static=%d): %s",
                len(issues), len(llm_issues), len(static_issues),
                ", ".join(i.category for i in issues),
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Issue details: %s", to_json_format(issues))
        else:
            logger.info("Judgment: no issues")
        return issues

    # Helper methods --------------------------------------------------------------------------------

    def _check_valid_response(self, response: list[JudgeOutput], questions: list[JudgeQuestion]) -> bool:
        """Check that the judge's response covers exactly the expected questions.

        Args:
            response: The list of answer entries returned by the judge, each
                expected to contain at least an "id" key.
            questions: The list of questions that were asked, used as the
                source of truth for expected ids.

        Returns:
            True if response contains exactly one entry per expected
            question id (regardless of order), False otherwise.
        """
        if len(response.answers) != len(questions):
            return False

        response_ids = {item.id for item in response.answers}
        expected_ids = {q.id for q in questions}

        return response_ids == expected_ids

    def _map_judge_issues(self, judge_problems: list[JudgeProblem]) -> list[JudgeIssue]:
        """Convert failed judge answers into JudgeIssue instances.

        Args:
            judge_problems: The judge's answers where the condition did not
                hold (answer == False).

        Returns:
            A list of JudgeIssue instances, one per problem, using the
            question id as category and the judge's reason as the issue text.
        """
        issues: list[JudgeIssue] = []

        for problem in judge_problems:
            issues.append(JudgeIssue(category=problem.id, issue=problem.reason))

        return issues

    def _judge_static_format(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext) -> list[JudgeIssue]:
        """Run deterministic, non-LLM checks on the dialogue's structure.

        Verifies formatting rules that don't require semantic judgment: use
        of the NPC's mandatory expression, the expected number of player
        dialogue options, and the presence of accept/refuse options when
        the NPC's intent requires a choice.

        Args:
            composed_dialogue: The generated dialogue to check.
            npc_context: Contextual information about the NPC, including
                its intent and required expression.

        Returns:
            A list of JudgeIssue instances for each static rule violated.
            Empty if no violations are found.
        """
        checks = (
            self._check_mandatory_expression(composed_dialogue, npc_context),
            self._check_option_count(composed_dialogue),
            self._check_accept_refuse(composed_dialogue, npc_context),
            self._check_banned_words(composed_dialogue),
        )
        issues = [issue for issue in checks if issue is not None]
        logger.debug("Static checks: %d/%d violated", len(issues), len(checks))
        return issues

    def _check_mandatory_expression(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext) -> JudgeIssue | None:
        """
        Check if the NPC's mandatory expression is used in the dialogue.
        """
        expression = npc_context.intent.must_use_expression
        if expression and expression not in composed_dialogue.dialogue:
            logger.debug("Mandatory expression %r not found in dialogue", expression)
            return JudgeIssue(category="Must use expression", issue="Expression is not used in dialogue")
        return None

    def _check_option_count(self, composed_dialogue: ComposedDialogue) -> JudgeIssue | None:
        """
        Check if the number of player options in the dialogue matches the expected number.
        """
        options = composed_dialogue.player_options
        if not options or not options.dialogue_options:
            return None
        expected = Settings().number_of_options
        actual = len(options.dialogue_options)
        if actual != expected:
            logger.debug("Option count mismatch: expected=%d, actual=%d", expected, actual)
            return JudgeIssue(category="Number of options", issue=f"Incorrect number of options: expected {expected}, got {actual}")
        return None

    def _check_accept_refuse(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext) -> JudgeIssue | None:
        """
        Check if the dialogue includes Accept/Refuse options when required by the NPC's intent.
        """
        if not getattr(npc_context.intent, "has_choice", False):
            return None
        options = composed_dialogue.player_options
        if not options or not (options.accept and options.refuse):
            logger.debug(
                "Accept/Refuse missing: accept=%s, refuse=%s",
                bool(options and options.accept), bool(options and options.refuse),
            )
            return JudgeIssue(category="Accept/Refuse", issue="Missing Accept/Refuse options")
        return None

    def _check_banned_words(self, composed_dialogue) -> JudgeIssue | None:
        """
        Check if the dialogue includes Banned words from the hurtlex lexicon.
        """
        if not Settings().profanity_filter:
            logger.debug("Banned words check skipped (profanity_filter=%s)", Settings().profanity_filter)
            return None
        
        banned_words = self.guardrail.retrieve_banned_words_in_composed_dialogue(composed_dialogue)
        if banned_words:
            joined = ", ".join(banned_words)
            logger.debug("Banned words found: %s", joined)
            return JudgeIssue(category="Banned words", issue=f"Banned words used in the dialogue: {joined}")
        logger.debug("Banned words check: clean")
        return None

    def _check_reasons(self, judge_output: JudgeOutput) -> list[JudgeProblem]:
        """Return the False answers whose reason is shorter than MIN_REASON_LENGTH.

        Only False answers are checked, since they are the ones that become
        JudgeIssue.

        Returns:
            The offending answers (empty list if all reasons are valid).
        """
        return [
            a for a in judge_output.answers
            if len(a.reason.strip()) < MIN_REASON_LENGTH
        ]

    def _build_reason_feedback(self, bad_reasons: list[JudgeProblem]) -> str:
        """Build the feedback message to send to the judge on retry."""
        details = "\n".join(f'- id="{a.id}": reason={a.reason!r}' for a in bad_reasons)
        return details