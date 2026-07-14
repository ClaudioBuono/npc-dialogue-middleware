from enum import Enum


class ComplexityTier(str, Enum):
    """
    Complexity categories used to route LLM requests.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"