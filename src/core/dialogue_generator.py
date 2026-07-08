# core/dialogue_generator.py
import logging
from typing import Any

from core.llm.openai_client import OpenAICompatibleClient
from core.types.contract import Contract


class DialogueGenerator:
    """
    Takes a ready-made Contract and generates the NPC dialogue line
    by calling the LLM client.
    """

    def __init__(self, client: OpenAICompatibleClient) -> None:
        self._client = client

    def generate(self, contract: Contract) -> str:
        return self._client.generate(contract, temperature=0.7)