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
        """Initialize the generator with an optional LLM client.

        Args:
            client: The OpenAI-compatible client used to generate dialogue.
                May be None if no client is configured.
        """
        self._client = client

    def generate(self, contract: Contract) -> str:
        """Generate dialogue text for the given contract.

        Uses the configured client to produce a response, applying the
        temperature defined in the application settings.

        Args:
            contract: The contract describing the generation request
                (e.g. game context, NPC context, world state).

        Returns:
            The generated dialogue as a string.
        """
        return self._client.generate(contract, temperature=Settings().llm.temperature)

    @staticmethod
    def build_questions(default_language: str) -> list[JudgeQuestion]:
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
        ]

    # TODO: implement static checks and evaluations for the healing