from enum import Enum

# Error codes ------------------------------------------------------------
class ValidationErrorCode(Enum):
    """Semantic error codes for preprocessing validation failures."""

    EMPTY_FIELD = "empty_field"
    FIELD_TOO_LONG = "field_too_long"
    FIELD_TOO_SHORT = "field_too_short"
    INVALID_VALUE = "invalid_value"
    
class LLMClientErrorCode(Enum):
    """Semantic error codes for LLM client failures."""

    CONNECTION_ERROR = "connection_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TIMEOUT_ERROR = "timeout_error"
    EMPTY_RESPONSE = "empty_response"
    UNKNOWN_ERROR = "unknown_error"


# Error parsing ----------------------------------------------------------
class PreProcessingError(Exception):
    """Raised when semantic validation of an incoming model fails.

    Carries a machine-readable error code and a list of human-readable
    field-level error messages, so the API layer can translate this
    into an appropriate HTTP response without needing domain knowledge.
    """

    def __init__(self, code: ValidationErrorCode, errors: list[str]):
        self.code = code
        self.errors = errors
        super().__init__(f"[{code.value}] " + "; ".join(errors))

class LLMClientError(Exception):
    """Raised when an LLM client fails to produce a valid response.

    Carries a machine-readable error code so the orchestrator/API layer
    can decide how to react (retry, fallback model, surface to client, etc.)
    without needing to inspect provider-specific exception types.
    """

    def __init__(self, code: LLMClientErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code.value}] {message}")