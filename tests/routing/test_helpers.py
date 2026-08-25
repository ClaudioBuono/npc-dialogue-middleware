import pytest
from core.routing.helpers import classify_score_to_complexity_tier
from core.types.enums import ComplexityTier
from core.config.thresholds import LOW_THRESHOLD, HIGH_THRESHOLD

def test_classify_score_to_complexity_tier():
    # LOW threshold is 0.35, HIGH threshold is 0.7
    # Under LOW_THRESHOLD -> LOW
    assert classify_score_to_complexity_tier(0.0) == ComplexityTier.LOW
    assert classify_score_to_complexity_tier(LOW_THRESHOLD - 0.01) == ComplexityTier.LOW
    
    # At or above LOW_THRESHOLD, below HIGH_THRESHOLD -> MEDIUM
    assert classify_score_to_complexity_tier(LOW_THRESHOLD) == ComplexityTier.MEDIUM
    assert classify_score_to_complexity_tier(HIGH_THRESHOLD - 0.01) == ComplexityTier.MEDIUM
    
    # At or above HIGH_THRESHOLD -> HIGH
    assert classify_score_to_complexity_tier(HIGH_THRESHOLD) == ComplexityTier.HIGH
    assert classify_score_to_complexity_tier(1.0) == ComplexityTier.HIGH
