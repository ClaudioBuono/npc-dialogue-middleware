from __future__ import annotations
from core.config.settings import Settings
from core.llm.openai_client import OpenAICompatibleClient
from core.types.dataclasses import Contract, JudgeQuestion


class Judger:
    """
    Handles dialogue evaluation and compliance checking against defined game rules and context.

    Uses an LLM client to execute evaluation contracts and generate structured judgments on NPC dialogues.
    """

    def __init__(self, client: OpenAICompatibleClient | None) -> None:
        self._client = client

    def generate(self, contract: Contract) -> str:
        return self._client.generate(contract, temperature=Settings().llm.temperature)

    @staticmethod
    def build_questions(default_language: str) -> list[JudgeQuestion]:
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
        ]

    # TODO: implement static checks and evaluations for the healing