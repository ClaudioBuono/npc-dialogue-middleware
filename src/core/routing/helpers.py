from core.config.thresholds import HIGH_THRESHOLD, LOW_THRESHOLD
from core.types.enums import ComplexityTier


def classify_score_to_complexity_tier(score: float) -> ComplexityTier:
    """Converts a score (0.0 - 1.0) to ComplexityTier"""
    
    if score < LOW_THRESHOLD:
        return ComplexityTier.LOW
    if score < HIGH_THRESHOLD:
        return ComplexityTier.MEDIUM
    return ComplexityTier.HIGH