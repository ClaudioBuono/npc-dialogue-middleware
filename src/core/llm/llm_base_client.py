from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """
    Abstract interface for LLM clients.
    """

    @abstractmethod
    def generate(self, prompt: str, temperature: float) -> str:
        """
        Performs a call to the model and returns the raw response string.

        Args:
            prompt:      Full prompt string to send to the model.
            temperature: Temperature value for generation (0.0–1.0).

        Returns:
            JSON string or raw text returned by the model.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""
        ...
