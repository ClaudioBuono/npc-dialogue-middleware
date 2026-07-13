from abc import ABC, abstractmethod
from typing import Iterator
from core.types.contract import Contract


class BaseLLMClient(ABC):
    """
    Abstract interface for LLM clients.
    """

    @abstractmethod
    def generate(self, contract: Contract, temperature: float) -> str:
        """
        Performs a call to the model and returns the raw response string.

        Args:
            prompt:      Full prompt string to send to the model.
            temperature: Temperature value for generation (0.0–1.0).

        Returns:
            JSON string or raw text returned by the model.
        """
        ...

    @abstractmethod
    def generate_streaming(self, contract: Contract, temperature: float) -> Iterator[str]:
        """Performs a streaming call to the model, yielding text chunks
        as they arrive.

        Used primarily for profiling (time to first token, throughput),
        where measuring incremental arrival matters. Regular dialogue
        generation should keep using `generate()`.

        Args:
            contract:    Structured payload describing the request.
            temperature: Temperature value for generation (0.0-1.0).

        Yields:
            Text chunks as they are produced by the model.

        Raises:
            LLMClientError: If the call fails (connection, auth, rate
                limit, timeout).
        """
        ...

    @abstractmethod
    def _format_request(self, contract: Contract, temperature: float) -> dict | None:
        """
        Formats a contract into a prompt payload for the request.

        Args:
            contract (Contract): The contract object to be formatted.
            temperature (float): The sampling temperature to use for generation.

        Returns:
            dict | None: A dictionary containing the formatted request data,
                         or `None` if the formatting fails.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""
        ...
