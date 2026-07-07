from openai import (
    OpenAI,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIStatusError,
)
from core.llm.llm_base_client import BaseLLMClient
from tools.errors import LLMClientError, LLMClientErrorCode


class OpenAICompatibleClient(BaseLLMClient):
    """Generic client for any provider exposing an OpenAI-compatible API.

    Covers Ollama, vLLM, LM Studio, llama.cpp server, LocalAI, and most
    cloud providers (Groq, Together AI, Mistral, DeepSeek, etc.), since
    they all implement the same request/response schema as OpenAI's API.
    """

    def __init__(self, endpoint: str, api_key: str | None, model_identifier: str):
        self._client = OpenAI(base_url=endpoint, api_key=api_key or "not-required")
        self._model_identifier = model_identifier

    def generate(self, prompt: str, temperature: float) -> str:
        """Performs a call to the model and returns the raw response string.

        Args:
            prompt:      Full prompt string to send to the model.
            temperature: Temperature value for generation (0.0-1.0).

        Returns:
            The generated text content.

        Raises:
            LLMClientError: If the call fails (connection, auth, rate limit,
                timeout) or if the model returns an empty/non-text response.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model_identifier,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        except AuthenticationError as e:
            raise LLMClientError(
                code=LLMClientErrorCode.AUTHENTICATION_ERROR,
                message=f"Authentication failed for model '{self._model_identifier}': {e}",
            ) from e
        except RateLimitError as e:
            raise LLMClientError(
                code=LLMClientErrorCode.RATE_LIMIT_ERROR,
                message=f"Rate limit exceeded for model '{self._model_identifier}': {e}",
            ) from e
        except APITimeoutError as e:
            raise LLMClientError(
                code=LLMClientErrorCode.TIMEOUT_ERROR,
                message=f"Request timed out for model '{self._model_identifier}': {e}",
            ) from e
        except APIConnectionError as e:
            raise LLMClientError(
                code=LLMClientErrorCode.CONNECTION_ERROR,
                message=f"Could not connect to endpoint for model '{self._model_identifier}': {e}",
            ) from e
        except APIStatusError as e:
            raise LLMClientError(
                code=LLMClientErrorCode.UNKNOWN_ERROR,
                message=f"Provider returned an error (status {e.status_code}) for model '{self._model_identifier}': {e}",
            ) from e

        content = response.choices[0].message.content

        if content is None:
            raise LLMClientError(
                code=LLMClientErrorCode.EMPTY_RESPONSE,
                message=f"Model '{self._model_identifier}' returned an empty response "
                        f"(possibly a tool call or content filter block).",
            )

        return content

    @property
    def model_name(self) -> str:
        return self._model_identifier
