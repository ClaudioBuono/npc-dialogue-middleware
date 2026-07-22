import logging
from typing import List
from core.config import thresholds
from core.config.thresholds import HIGH_THRESHOLD, LOW_THRESHOLD
from core.routing.complexity_analyzer import ComplexityAnalyzer, ComplexityScore
from core.routing.registry import ModelRegistry
from core.routing.profiler import RankedModel
from core.llm.llm_base_client import BaseLLMClient
from core.types.contexts import GameContext, NPCContext
from core.types.enums import ComplexityTier
from tools.errors import LLMClientError, LLMClientErrorCode
logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Selects which registered model should handle a given request,
    based on the request's complexity tier.
    """

    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer.from_thresholds(thresholds)
    
    def select_model(self, game_context: GameContext, npc_context: NPCContext) -> BaseLLMClient:
        """
        Picks the best available model client for the given complexity.
        """
        selected_model: RankedModel
        ranked_models: List[RankedModel] = ModelRegistry().get_ranked_models()
        
        if not ranked_models: # Empty list -> raise exception
            raise LLMClientError(
                code=LLMClientErrorCode.UNKNOWN_ERROR,
                message="No models are currently registered. Call /api/set_models first.",
            )
        
        elif len(ranked_models) == 1: # Only one element -> return the only client
            selected_model = ranked_models[0]
        
        else: # More than one model -> handle complexity based on tier
            
            complexity = self.complexity_analyzer.analyze(game_context, npc_context)
            logger.debug(f"Computed complexity score: {complexity.value}, tier: {complexity.tier}")

            match complexity.tier:
                case ComplexityTier.LOW: selected_model = self._handle_low_complexity()  
                case ComplexityTier.MEDIUM: selected_model = self._handle_medium_complexity(complexity.value) 
                case ComplexityTier.HIGH: selected_model = self._handle_high_complexity() 

        return selected_model.client
    
    @staticmethod
    def _handle_low_complexity() -> RankedModel:
        """
        Pick the most adequate model for a task with LOW complexity. 
        It follows an "escalation" strategy:
            1. Pick the best LOW model
            2. If there are no LOW models, pick the best MEDIUM model
            3. If there are no MEDIUM models, pick the best HIGH model
        """
        selected_model: RankedModel | None = ModelRegistry().get_best_for_tier(ComplexityTier.LOW)
        if selected_model:
            return selected_model
        else:
            selected_model = ModelRegistry().get_best_for_tier(ComplexityTier.MEDIUM)
            if selected_model:
                return selected_model
        
        return ModelRegistry().get_best_for_tier(ComplexityTier.HIGH) 

    @staticmethod
    def _handle_medium_complexity(score: float) -> RankedModel:
        """
        Pick the most adequate model for a task with MEDIUM complexity. 
        It follows the following strategy:
            1. Pick the best medium models
            2. See if the complexity `score` is closer to LOW or HIGH, then pick the closest complexity tier
            3. If there are no models in the closest complexity tier, pick the remaining tier
        """
        selected_model: RankedModel | None = ModelRegistry().get_best_for_tier(ComplexityTier.MEDIUM)
        if selected_model:
            return selected_model
        
        else:
            low_diff = score - LOW_THRESHOLD
            high_diff = HIGH_THRESHOLD - score

            if high_diff < low_diff: # Closer to HIGH
                selected_model = ModelRegistry().get_best_for_tier(ComplexityTier.HIGH)
                if selected_model:
                    return selected_model
                return ModelRegistry().get_best_for_tier(ComplexityTier.LOW) 
            
            else: # Closer to LOW
                selected_model = ModelRegistry().get_best_for_tier(ComplexityTier.LOW)
                if selected_model:
                    return selected_model
                return ModelRegistry().get_best_for_tier(ComplexityTier.HIGH) 


    @staticmethod
    def _handle_high_complexity() -> RankedModel:
        """
        Pick the most adequate model for a task with MEDIUM complexity. 
        It follows an "de-escalation" strategy:
            1. Pick the best HIGH model
            2. If there are no HIGH models, pick the best MEDIUM model
            3. If there are no MEDIUM models, pick the best LOW model
        """
        selected_model: RankedModel | None = ModelRegistry().get_best_for_tier(ComplexityTier.HIGH)
        if selected_model:
            return selected_model
        
        else:
            selected_model = ModelRegistry().get_best_for_tier(ComplexityTier.MEDIUM)
            if selected_model:
                return selected_model
            
        return ModelRegistry().get_best_for_tier(ComplexityTier.LOW) 