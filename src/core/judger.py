from __future__ import annotations
from core.config.settings import Settings
from core.llm.openai_client import OpenAICompatibleClient
from core.types.dataclasses import Contract


class Judger:
    """
    Handles dialogue evaluation and compliance checking against defined game rules and context.

    Uses an LLM client to execute evaluation contracts and generate structured judgments on NPC dialogues.
    """

    def __init__(self, client: OpenAICompatibleClient | None) -> None:
        self._client = client

    def generate(self, contract: Contract) -> str:
        return self._client.generate(contract, temperature=Settings().llm.temperature)

    # TODO: implement static checks and evaluations for the healing