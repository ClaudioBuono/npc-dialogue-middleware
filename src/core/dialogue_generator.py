import logging
from typing import Iterator
from core.llm.openai_client import OpenAICompatibleClient
from core.types.dataclasses import Contract
logger = logging.getLogger(__name__)


class DialogueGenerator:
    """
    Takes a Contract and generates the NPC dialogue line by calling the LLM client.
    """
    
    def __init__(self) -> None:
        self._client = None

    #TODO: Manage exceptions
    def generate(self, contract: Contract) -> str:
        return self._client.generate(contract, temperature=0.3)
    
    def set_client(self, client: OpenAICompatibleClient) -> None:
        self._client = client

    def generate_stream(self, contract: Contract) -> Iterator[str]:
        return self._client.generate_streaming(contract, temperature=0.3)
