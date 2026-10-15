import pytest
from core.routing.helpers import classify_score_to_complexity_tier
from core.types.enums import ComplexityTier
from core.configuration.thresholds import LOW_THRESHOLD, HIGH_THRESHOLD


@pytest.mark.parametrize(
    "score, expected_tier",
    [
        # Negative scores and zero boundary -> LOW
        (-1.0, ComplexityTier.LOW),
        (-0.001, ComplexityTier.LOW),
        (0.0, ComplexityTier.LOW),
        (0.1, ComplexityTier.LOW),
        (LOW_THRESHOLD - 1e-6, ComplexityTier.LOW),
        (LOW_THRESHOLD - 0.01, ComplexityTier.LOW),

        # Exact LOW_THRESHOLD and values between LOW and HIGH -> MEDIUM
        (LOW_THRESHOLD, ComplexityTier.MEDIUM),
        (LOW_THRESHOLD + 1e-6, ComplexityTier.MEDIUM),
        ((LOW_THRESHOLD + HIGH_THRESHOLD) / 2.0, ComplexityTier.MEDIUM),
        (HIGH_THRESHOLD - 0.01, ComplexityTier.MEDIUM),
        (HIGH_THRESHOLD - 1e-6, ComplexityTier.MEDIUM),

        # Exact HIGH_THRESHOLD and values above -> HIGH
        (HIGH_THRESHOLD, ComplexityTier.HIGH),
        (HIGH_THRESHOLD + 1e-6, ComplexityTier.HIGH),
        (0.85, ComplexityTier.HIGH),
        (1.0, ComplexityTier.HIGH),
        (1.5, ComplexityTier.HIGH),
        (100.0, ComplexityTier.HIGH),
    ],
)
def test_classify_score_to_complexity_tier_boundaries(score: float, expected_tier: ComplexityTier):
    """Verifies that scores map correctly to complexity tiers across boundaries, edge cases, and extreme values."""
    assert classify_score_to_complexity_tier(score) == expected_tier

