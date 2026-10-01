from enum import Enum


class ComplexityTier(str, Enum):
    """
    Complexity categories used to route LLM requests.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Language(str, Enum):

    ENGLISH = "en"
    ITALIAN = "it"

class MiddlewareState(str, Enum):
    """Possible states of the middleware pipeline."""

    IDLE = "idle" # Doing nothing, can accept new requests
    STARTING = "starting" # Initializing the middleware
    SETTING_CONTEXT = "setting_context" # Setting the game context
    GENERATING = "generating" # Generating the dialogue