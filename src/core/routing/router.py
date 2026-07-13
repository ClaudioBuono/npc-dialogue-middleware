from core.routing.registry import ModelRegistry, IntendedTier
from core.routing.profiler import RankedModel
from core.llm.llm_base_client import BaseLLMClient
from tools.errors import LLMClientError, LLMClientErrorCode


class LLMRouter:
    """
    Selects which registered model should handle a given request,
    based on the request's complexity tier.
    """

	# TODO: Replace with actual complexity evaluation
    def select_model(self, complexity: IntendedTier) -> BaseLLMClient:
        """
        Picks the best available model client for the given complexity.
        """
        ranked_model: RankedModel | None = ModelRegistry.get_best_for_tier(complexity)

        if ranked_model is None:
            raise LLMClientError(
                code=LLMClientErrorCode.UNKNOWN_ERROR,
                message="No models are currently registered. Call /api/set_models first.",
            )

        return ranked_model.client