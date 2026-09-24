from __future__ import annotations
from api.schemas import ComposedDialogue
from core.config.settings import Settings
from core.contract_builder import ContractBuilder
from core.llm.openai_client import OpenAICompatibleClient
from core.types.contexts import GameContext, NPCContext
from core.types.dataclasses import Contract, JudgeQuestion


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
        self.score = 1


    def set_client(self, client: OpenAICompatibleClient) -> None:
        self._client = client


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
                "faithfulness",
                "Is every claim in the dialogue fully supported by the provided "
                "game and NPC context, without inventing facts?",
            ),
            JudgeQuestion(
                "consistency",
                "Is the dialogue fully consistent with the game context, NPC "
                "context, and current world state (no contradictions)?",
            ),
            JudgeQuestion(
                "persona_consistency",
                "Does the dialogue match the NPC's personality, talkativeness "
                "level, and relationship with the main character?",
            ),
            JudgeQuestion(
                "entity_check",
                "Are all named entities in the dialogue (people, places, items, "
                "factions) either present in the context or plausible within it?",
            ),
            JudgeQuestion(
                "language",
                f"Is the dialogue written in {default_language} or in a language "
                "consistent with the NPC's allowed languages?",
            ),
            JudgeQuestion(
                "fairness",
                "Is the dialogue free from stereotypes, discriminatory language, "
                "or biased characterizations based on gender, ethnicity, religion, "
                "age, or other identity traits, unless explicitly justified by the "
                "NPC's established persona or narrative role?",
            ),
        ]

    # TODO: implement static checks and evaluations for the healing

    def judge_dialogue(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext, game_context: GameContext):
        judge_questions = self._build_questions(Settings().language.name)
        judge_contract = self.contract_builder.build_judge_contract(composed_dialogue, game_context, npc_context, judge_questions)

        # TODO: extract issues list, to test higher temperatures
        judge_output_raw = self._client.generate(judge_contract, temperature = 0.0)

        issues = []

        #TODO:
        # score calculator function based on issues judged and weights
        #judged_issues = calculate_weigthed_score()

        static_static = self.judge_static_format(composed_dialogue, npc_context)
        #issues = [judged_issues, static_static]
        
        return [self.score, issues]


    def judge_static_format(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext):

        issues = []

        if not npc_context.intent.must_use_expression in composed_dialogue.dialogue:
            issues.append("- Missing must use expression")

        if len(composed_dialogue.player_options.dialogue_options) != Settings().number_of_options:
            issues.append("- Incorrect number of options")

        if npc_context.intent.has_choice:
            if not(composed_dialogue.player_options.accept and composed_dialogue.player_options.refuse):
                issues.append("- Missing Accept/Refuse")
                

        self.score = 0 if issues else self.score

        return issues
            

        