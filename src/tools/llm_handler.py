import logging
from abc import ABC, abstractmethod
import requests

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Base client interface (extensibility hook for real LLM providers)      #
# ======================================================================= #

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

# ======================================================================= #
#  LLMHandler — orchestrates the single LLM call                          #
# ======================================================================= #

class OllamaClient(BaseLLMClient):
    """
    Client for models served through Ollama.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        system_prompt: str | None = None,
        extra_options: dict | None = None,
    ):
        """
        Args:
            model:         Ollama model name.
            base_url:      Base URL of the Ollama server.
            timeout:       Timeout in seconds for the HTTP request.
            system_prompt: Optional system prompt applied to every call.
            extra_options: Additional options to pass to the Ollama API 'options' parameter.
        """
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.extra_options = extra_options or {}

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, temperature: float) -> str:
        """
        Performs a call to Ollama (/api/generate endpoint) in non-streaming mode
        and returns the generated text from the model.
        """
        url = f"{self.base_url}/api/generate"

        options = {"temperature": temperature, **self.extra_options}

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if self.system_prompt:
            payload["system"] = self.system_prompt

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Unable to connect to Ollama at {self.base_url}. "
                f"Check that the service is running ('ollama serve')."
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"Timeout ({self.timeout}s) calling model '{self._model}'."
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Ollama HTTP error ({response.status_code}): {response.text}"
            ) from e

        data = response.json()

        if "response" not in data:
            raise RuntimeError(
                f"Unexpected response from Ollama, missing 'response' field: {data}"
            )

        return data["response"]


class LLMCallError(Exception):
    """Exception raised when the LLM call fails irrecoverably."""
    pass


class LLMHandler:
    """
    Manages the single LLM call that generates NPC Dialogue and Player Options simultaneously.

    Builds the structured prompt from the context, invokes the LLM client,
    and deserializes the raw response into an LLMOutput object.

    Args:
        client: Instance of BaseLLMClient.
    """

    # System prompt template
    _SYSTEM_PROMPT = (
        "You are an expert game master for interactive narrative. "
        "You must generate ONLY valid JSON with the following structure:\n"
        '{{"dialogue": "<NPC dialogue>", "options": [{{"option_id": "...", '
        '"text": "...", "tone": "...", "consequence": "..."}}]}}\n'
        "Do not add text outside the JSON. "
        "Use the language specified in the context. "
        "If there are RETRY_FEEDBACK instructions, apply them strictly. "
        "Below is the game and NPC context in JSON format."
    )

    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client


    def call(self, npc_context_data, game_context_data) -> dict[str, any]:
        """
        Performs the single LLM call to generate NPC dialogue and player options.

        Args:
            npc_context_data: Context data for the NPC.
            game_context_data: Context data for the game.

        Returns:
            LLMOutput object with parsed dialogue and options.

        Raises:
            LLMCallError: If the response cannot be parsed after all attempts.
        """
        data = {"NPC_CONTEXT": npc_context_data, "GAME_CONTEXT": game_context_data}
        prompt = self._build_prompt(data)

        raw_response = self.client.generate(
            prompt=prompt,
            temperature=0.3,
        )

        return raw_response

    # ------------------------------------------------------------------ #
    #  Private helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_prompt(self, data) -> str:
        """Builds the full prompt (system + game + npc) to send to the LLM."""
        return (
            f"[SYSTEM]\n{self._SYSTEM_PROMPT}\n\n"
            f"[CONTEXT]\n{data}"
        )
