from __future__ import annotations
from pprint import pprint 
from pydantic import ValidationError
from api.schemas import ComposedDialogue
from core.configuration.settings import Settings
from core.generation.contract_builder import ContractBuilder
from core.helpers.formatters import to_json_format
from core.llm.openai_client import OpenAICompatibleClient
from core.pipeline.guardrail import Guardrail
from core.tools.errors import MiddlewareError, MiddlewareErrorCode, PreProcessingError, ValidationErrorCode
from core.types.contexts import Dialogue, GameContext, NPCContext
from core.types.dataclasses import JudgeIssue, JudgeOutput, JudgeProblem, JudgeQuestion
from core.types.enums import Language, ProfanityMode


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

    def judge_dialogue(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext, game_context: GameContext) -> list[JudgeIssue]:
        # Build contract
        persona_questions = self._build_persona_questions()
        option_questions = self._build_rules_questions(npc_context, Settings().language)
        fairness_questions = self._build_fairness_questions(Settings().prompt_fairness_filter)
        judge_questions = persona_questions + option_questions + fairness_questions
        judge_contract = self.contract_builder.build_judge_contract(composed_dialogue, game_context, npc_context, judge_questions)
        print("JUDGE INPUT: ",to_json_format(judge_contract))
        
        # Parse response and check its validity 
        judge_output_raw = self._client.generate(judge_contract, temperature = 0.3) # TODO: Test higher temperatures
        try:
            judge_output = JudgeOutput.model_validate_json(judge_output_raw)
            print("JUDGE RESULT: ",to_json_format(judge_output))
        except ValidationError as e:
            raise PreProcessingError(code=ValidationErrorCode.INVALID_VALUE, errors=[f"Judge output does not match expected schema: {e}", f"Raw output: {judge_output_raw}"])
        
        if not self._check_valid_response(judge_output, judge_questions):
            raise MiddlewareError(code=MiddlewareErrorCode.INVALID_RESPONSE, errors=["There was a problem during the judging process."])

        # Extract problems (where the answer is False)
        judge_problems: list[JudgeProblem] = [p for p in judge_output.answers if not p.answer]

        # Build issues array
        issues = self._map_judge_issues(judge_problems)
        static_issues = self._judge_static_format(composed_dialogue, npc_context)
        return issues + static_issues

    # Helper methods --------------------------------------------------------------------------------
    def _build_persona_questions(self) -> list[JudgeQuestion]:
        """Build the standard set of judge questions used to evaluate dialogue.

        Each question targets a specific quality dimension of the generated
        dialogue: faithfulness to context, consistency with the game/NPC/world
        state, adherence to the NPC's persona, correctness of named entities,
        and language compliance.

        Args:
            default_language: The language the dialogue is expected to be
                written in, used to build the language-check question.

        Returns:
            A list of JudgeQuestion instances covering faithfulness,
            consistency, persona consistency, entity check, and language.
        """
        # TODO: Integrate in refiner?
                        
        return [
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
        ]
    
    # options non devono essere acept/refuse
    # accept e refuse effettivamente facciano intedere accept/refuse
    def _build_rules_questions(self, npc_context: NPCContext, current_language: Language) -> list[JudgeQuestion]:
        language_output: str = current_language.name
        rules_questions: list[JudgeQuestion] = [
            JudgeQuestion(
                id="language",
                text=f"Is the entire dialogue written in {language_output}, with no other language mixed in?",
            ),
            JudgeQuestion(
                id="no_narration",
                text=(
                    "Is the line written purely as words spoken aloud by the NPC, "
                    "in first person, with no narration? It should NOT contain "
                    "third-person descriptions (e.g. 'he sighs'), stage directions or "
                    "actions (e.g. '*laughs*', '(nods)'), or references to ambient "
                    "sounds and the environment (e.g. 'the wind howls')."
                )
            )
        ]

        if isinstance(npc_context.intent, Dialogue):
            return rules_questions

        if npc_context.intent.has_choice:
            choice_question = JudgeQuestion(
                id="consistent_choices",
                text="Is accept choice actually refering to something that could intend accepting a quest? Is the same happening for refuse?"
            )
            rules_questions.append(choice_question)

        if npc_context.intent.has_options:
            options_questions = JudgeQuestion(
                id="consistent_options",
                text="Are options not refering to something that could intend accepting a quest?"
            )
            rules_questions.append(options_questions)

        return rules_questions

    # TODO: Improve fairness detection
    def _build_fairness_questions(self, filter: bool) -> list[JudgeQuestion]:
        if filter:
            return [
                JudgeQuestion(
                    id="fairness",
                    text="Is the dialogue free from stereotypes, discriminatory language, "
                    "or biased characterizations based on gender, ethnicity, religion, "
                    "age, or other identity traits, unless explicitly justified by the "
                    "NPC's established persona or narrative role?",
                )
            ]

        return []


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
            self._check_banned_words(composed_dialogue)
        )
        return [issue for issue in checks if issue is not None]

    def _check_mandatory_expression(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext) -> JudgeIssue | None:
        """
        Check if the NPC's mandatory expression is used in the dialogue.
        """
        expression = npc_context.intent.must_use_expression
        if expression and expression not in composed_dialogue.dialogue:
            return JudgeIssue(category="Must use expression", issue="Expression is not used in dialogue")
        return None

    def _check_option_count(self, composed_dialogue: ComposedDialogue) -> JudgeIssue | None:
        """
        Check if the number of player options in the dialogue matches the expected number.
        """
        options = composed_dialogue.player_options
        if not options or not options.dialogue_options:
            return None
        if len(options.dialogue_options) != Settings().number_of_options:
            return JudgeIssue(category="Number of options", issue="Incorrect number of options")
        return None

    def _check_accept_refuse(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext) -> JudgeIssue | None:
        """
        Check if the dialogue includes Accept/Refuse options when required by the NPC's intent.
        """
        if not getattr(npc_context.intent, "has_choice", False):
            return None
        options = composed_dialogue.player_options
        if not options or not (options.accept and options.refuse):
            return JudgeIssue(category="Accept/Refuse", issue="Missing Accept/Refuse options")
        return None

    def _check_banned_words(self, composed_dialogue) -> JudgeIssue | None:
        """
        Check if the dialogue includes Banned words from the hurtlex lexicon.
        """
        if Settings().profanity_mode != ProfanityMode.DISABLED:
            banned_words = self.guardrail.retrieve_banned_words_in_composed_dialogue(composed_dialogue)
            if len(banned_words) > 0:
                return JudgeIssue(category="Banned words", issue=f"Banned words used in the dialogue: {", ".join(banned_words)}")
        return None
