from __future__ import annotations
from pprint import pprint 
from pydantic import ValidationError
from api.schemas import ComposedDialogue
from core.config.settings import Settings
from core.contract_builder import ContractBuilder
from core.helpers.formatters import to_json_format
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import MiddlewareError, MiddlewareErrorCode, PreProcessingError, ValidationErrorCode
from core.types.contexts import GameContext, NPCContext
from core.types.dataclasses import JudgeIssue, JudgeOutput, JudgeProblem, JudgeQuestion


class Judger:
    """
    Handles dialogue evaluation and compliance checking against defined game rules and context.

    Uses an LLM client to execute evaluation contracts and generate structured judgments on NPC dialogues.
    """
    def __init__(self, contract_builder: ContractBuilder) -> None:
        """Initialize the generator with an optional LLM client.

        Args:
            client: The OpenAI-compatible client used to generate dialogue.
                May be None if no client is configured.
        """
        self.contract_builder = contract_builder


    def set_client(self, client: OpenAICompatibleClient) -> None:
        self._client = client

    def judge_dialogue(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext, game_context: GameContext) -> list[JudgeIssue]:
        # Build contract
        judge_questions = self._build_questions(Settings().language.name)
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
    def _build_questions(self, default_language: str) -> list[JudgeQuestion]:
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
            JudgeQuestion(
                id="language",
                text=f"Is the entire dialogue written in {default_language}, with no other language mixed in?",
            ),
            JudgeQuestion(
                id="fairness",
                text="Is the dialogue free from stereotypes, discriminatory language, "
                "or biased characterizations based on gender, ethnicity, religion, "
                "age, or other identity traits, unless explicitly justified by the "
                "NPC's established persona or narrative role?",
            ),
        ]

    #TODO: refactor to reduce complexity
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
        issues: list[JudgeIssue] = []

        if npc_context.intent.must_use_expression and ( not npc_context.intent.must_use_expression in composed_dialogue.dialogue):
            issues.append(JudgeIssue(category="Must use expression", issue="Expression is not used in dialogue"))

        if composed_dialogue.player_options and composed_dialogue.player_options.dialogue_options:
            if len(composed_dialogue.player_options.dialogue_options) != Settings().number_of_options:
                issues.append(JudgeIssue(category="Number of options", issue="Incorrect number of options"))

        if npc_context.intent.has_choice:
            if composed_dialogue.player_options and composed_dialogue.player_options.accept and composed_dialogue.player_options.refuse:
                if not(composed_dialogue.player_options.accept and composed_dialogue.player_options.refuse):
                    issues.append(JudgeIssue(category="Accept/Refuse", issue="Missing Accept/Refuse options"))

        return issues
    
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