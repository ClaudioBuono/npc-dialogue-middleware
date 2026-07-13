from typing import Iterator

from openai import (
    OpenAI,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIStatusError,
)
from core.types.contract import Contract
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

    def generate(self, contract: Contract, temperature: float) -> str:
        """Performs a call to the model and returns the raw response string.

        Args:
            contract:    Structured payload describing the request
                (system prompt, user prompt, and expected output schema).
            temperature: Temperature value for generation (0.0-1.0).

        Returns:
            The generated text content.

        Raises:
            LLMClientError: If the call fails (connection, auth, rate limit,
                timeout) or if the model returns an empty/non-text response.
        """
        request = self._format_request(contract, temperature)

        try:
            response = self._client.chat.completions.create(**request)
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

    def generate_streaming(self, contract: Contract, temperature: float) -> Iterator[str]:
            """
            Streams text chunks as they arrive, used for profiling (time
            to first token, throughput). See BaseLLMClient for full contract.
            """
            request_kwargs = self._format_request(contract, temperature)
            request_kwargs["stream"] = True

            try:
                stream = self._client.chat.completions.create(**request_kwargs, timeout=3.0)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            except AuthenticationError as e:
                raise LLMClientError(code=LLMClientErrorCode.AUTHENTICATION_ERROR, message=f"Authentication failed for model '{self._model_identifier}': {e}") from e
            except RateLimitError as e:
                raise LLMClientError(code=LLMClientErrorCode.RATE_LIMIT_ERROR, message=f"Rate limit exceeded for model '{self._model_identifier}': {e}") from e
            except APITimeoutError as e:
                raise LLMClientError(code=LLMClientErrorCode.TIMEOUT_ERROR, message=f"Request timed out for model '{self._model_identifier}': {e}") from e
            except APIConnectionError as e:
                raise LLMClientError(code=LLMClientErrorCode.CONNECTION_ERROR, message=f"Could not connect to endpoint for model '{self._model_identifier}': {e}") from e
            except APIStatusError as e:
                raise LLMClientError(code=LLMClientErrorCode.UNKNOWN_ERROR, message=f"Provider returned an error (status {e.status_code}) for model '{self._model_identifier}': {e}") from e

    @property
    def model_name(self) -> str:
        return self._model_identifier

    def _format_request(self, contract: Contract, temperature: float) -> dict:
        """Builds the full keyword-argument dict for the OpenAI chat completion call.

        Translates the provider-agnostic Contract (system/user prompt,
        optional output schema) into the request shape expected by the
        OpenAI-compatible `chat.completions.create` endpoint.

        Args:
            contract:    Structured payload describing the request.
            temperature: Temperature value for generation (0.0-1.0).

        Returns:
            A dict of keyword arguments ready to be unpacked into
            `self._client.chat.completions.create(**request_kwargs)`.
        """
        request_kwargs = {
            "model": self._model_identifier,
            "messages": self._build_messages(contract),
            "temperature": temperature,
        }

        response_format = self._build_response_format(contract)
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        return request_kwargs

    @staticmethod
    def _build_messages(contract: Contract) -> list[dict[str, str]]:
        """Translates the provider-agnostic Contract into OpenAI's messages format."""
        return [
            {"role": "system", "content": contract.system_prompt},
            {"role": "user", "content": contract.user_prompt},
        ]

    @staticmethod
    def _build_response_format(contract: Contract) -> dict | None:
        """Translates Contract.output_schema into OpenAI's response_format, if present."""
        if not contract.output_schema:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "contract_output",
                "schema": contract.output_schema,
                "strict": True,
            },
        }