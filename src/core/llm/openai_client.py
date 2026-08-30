import logging
from typing import Iterator
import httpx
import pynvml
from core.logger import to_json_format
from openai import (
    OpenAI,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIStatusError,
)
from core.telemetry import TelemetryRecorder
from core.types.dataclasses import Contract
from core.llm.llm_base_client import BaseLLMClient
from tools.errors import LLMClientError, LLMClientErrorCode
logger = logging.getLogger(__name__)


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
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
        logger.debug(f"Sending request to OpenAI client:\n{to_json_format(request)}")

        recorder = TelemetryRecorder(self._model_identifier)
        chunks: list[str] = []

        with recorder:
            try:
                stream = self._client.chat.completions.create(**request)
                for chunk in stream:
                    if chunk.usage is not None:
                        recorder.set_usage(chunk.usage.completion_tokens)

                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        chunks.append(delta)
                        recorder.record_chunk()

            except AuthenticationError as e:
                logger.error(f"Authentication failed for model '{self._model_identifier}': {e}")
                raise LLMClientError(
                    code=LLMClientErrorCode.AUTHENTICATION_ERROR,
                    message=f"Authentication failed for model '{self._model_identifier}': {e}",
                ) from e
            except RateLimitError as e:
                logger.error(f"Rate limit exceeded for model '{self._model_identifier}': {e}")
                raise LLMClientError(
                    code=LLMClientErrorCode.RATE_LIMIT_ERROR,
                    message=f"Rate limit exceeded for model '{self._model_identifier}': {e}",
                ) from e
            except APITimeoutError as e:
                logger.error(f"Request timed out for model '{self._model_identifier}': {e}")
                raise LLMClientError(
                    code=LLMClientErrorCode.TIMEOUT_ERROR,
                    message=f"Request timed out for model '{self._model_identifier}': {e}",
                ) from e
            except APIConnectionError as e:
                logger.error(f"Connection error for model '{self._model_identifier}': {e}")
                raise LLMClientError(
                    code=LLMClientErrorCode.CONNECTION_ERROR,
                    message=f"Could not connect to endpoint for model '{self._model_identifier}': {e}",
                ) from e
            except APIStatusError as e:
                logger.error(f"API status error for model '{self._model_identifier}': {e}")
                raise LLMClientError(
                    code=LLMClientErrorCode.UNKNOWN_ERROR,
                    message=f"Provider returned an error (status {e.status_code}) for model '{self._model_identifier}': {e}",
                ) from e

            recorder.set_vram(*self._probe_vram())

        content = "".join(chunks)

        if not content:
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

    def _probe_vram(self) -> tuple[float | None, str]:
        """Tries the Ollama /api/ps endpoint first (more precise, per-model);
        falls back to NVML (total GPU VRAM) if it's not available."""
        vram, source = self._probe_vram_ollama()
        if vram is not None:
            return vram, source
        return self._probe_vram_nvml()

    def _probe_vram_nvml(self) -> tuple[float | None, str]:
        """Reads total VRAM currently in use on GPU 0 via NVML.

        Generic fallback for any backend, since it queries the GPU driver
        directly rather than the inference server. Returns (None, "unavailable")
        if NVML is not installed, no NVIDIA GPU is present, or the call fails
        for any other reason.
        """
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return round(info.used / (1024**2), 2), "nvml"
        except Exception:
            return None, "unavailable"

    def _probe_vram_ollama(self) -> tuple[float | None, str]:
        """Reads the VRAM used by this specific model from Ollama's /api/ps
        endpoint, which lists currently loaded models along with their
        per-model VRAM footprint (size_vram).

        Returns (None, "unavailable") if the endpoint is unreachable (e.g. the
        backend isn't Ollama), the request fails, or this model isn't currently
        loaded.
        """
        try:
            resp = httpx.get(f"{self._client.base_url}/api/ps", timeout=1.0)
            models = resp.json().get("models", [])
            match = next((m for m in models if m["name"] == self._model_identifier), None)
            if match:
                return round(match["size_vram"] / (1024**2), 2), "ollama_ps"
        except Exception:
            pass
        return None, "unavailable"