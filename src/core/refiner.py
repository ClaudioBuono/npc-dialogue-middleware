from api.schemas import ComposedDialogue
from core.config.settings import Settings
from core.healer import Healer
from core.judger import Judger
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import MiddlewareError, MiddlewareErrorCode
from core.types.contexts import GameContext, NPCContext
from core.types.dataclasses import JudgeIssue


class Refiner:
    """Orchestrates the iterative refinement of a composed dialogue.

    The Refiner coordinates a `Judger`, which evaluates a dialogue and
    reports any issues, and a `Healer`, which attempts to fix a dialogue
    given a list of issues. It repeatedly alternates judging and healing,
    up to a configurable maximum number of iterations, until the dialogue
    is judged issue-free or the iteration budget is exhausted.
    """

    def __init__(self, healer: Healer, judger: Judger) -> None:
        """Initialize the Refiner.

        Args:
            healer: The Healer instance used to fix issues found in a dialogue.
            judger: The Judger instance used to evaluate a dialogue and detect issues.
        """
        self.healer = healer
        self.judger = judger
        self.max_iterations = Settings().refiner_max_iterations

    def set_client(self, client: OpenAICompatibleClient):
        """Propagate the LLM client to the underlying healer and judger.

        Args:
            client: The OpenAI-compatible client to use for LLM calls.
        """
        self.healer.set_client(client)
        self.judger.set_client(client)

    def refine_dialogue(self, composed_dialogue: ComposedDialogue, npc_context: NPCContext, game_context: GameContext) -> ComposedDialogue:
        """Iteratively judge and heal a dialogue until it passes validation.

        On each iteration, the dialogue is evaluated by the Judger. If no
        issues are found, the current (valid) dialogue is returned
        immediately. Otherwise, the Healer attempts to produce a corrected
        version of the original dialogue based on the reported issues, and
        the process repeats with the healed dialogue.

        Args:
            composed_dialogue: The initial dialogue to refine.
            npc_context: Contextual information about the NPC speaking the dialogue.
            game_context: Contextual information about the game/world state.

        Returns:
            The refined ComposedDialogue once the Judger reports no issues.

        Raises:
            MiddlewareError: If the dialogue still has unresolved issues after
                `self.max_iterations` refinement attempts.
        """
        current_dialogue: ComposedDialogue = composed_dialogue

        for i in range(self.max_iterations):
            issues: list[JudgeIssue] = self.judger.judge_dialogue(current_dialogue, npc_context, game_context)

            if len(issues) == 0:
                return current_dialogue

            current_dialogue = self.healer.heal_dialogue(composed_dialogue, npc_context, game_context, issues)

        raise MiddlewareError(code=MiddlewareErrorCode.REFUSED, errors=["Could not refine the dialogue."])