from abc import ABC, abstractmethod
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
