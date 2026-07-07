import requests

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
