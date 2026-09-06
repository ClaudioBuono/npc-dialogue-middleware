from pydantic import BaseModel, Field
from typing import Any, List, Dict

from api.handlers import LLM_ERROR_STATUS_MAP, MIDDLEWARE_ERROR_STATUS_MAP

class ErrorMessageResponse(BaseModel):
    """Generic error response model containing an error code and a message."""
    error_code: int = Field(..., description="Internal or specific error code.")
    message: str = Field(..., description="Human-readable error message.")

class ErrorValidationResponse(BaseModel):
    """Validation error response model containing an error code and a list of errors."""
    error_code: int = Field(..., description="Internal or specific error code.")
    errors: List[Dict[str, Any]] = Field(..., description="List of validation errors.")

# --------------------------------------------------------------------------- #
# Error responses grouped by originating exception type
# --------------------------------------------------------------------------- #

# RoutingConfigError -> always 400
ROUTING_CONFIG_ERROR_RESPONSES: dict[int, dict] = {
    400: {
        "model": ErrorValidationResponse,
        "description": "Bad Request - Inconsistent routing configuration or similar bad inputs.",
    },
}

# PreProcessingError -> always 422
PREPROCESSING_ERROR_RESPONSES: dict[int, dict] = {
    422: {
        "model": ErrorValidationResponse,
        "description": "Unprocessable Entity - Semantic validation errors on the incoming payload.",
    },
}

# LLMClientError -> status codes derived from LLM_ERROR_STATUS_MAP values
_LLM_STATUS_DESCRIPTIONS: dict[int, dict] = {
    401: {
        "model": ErrorMessageResponse,
        "description": "Unauthorized - Authentication error with the LLM provider.",
    },
    429: {
        "model": ErrorMessageResponse,
        "description": "Too Many Requests - Rate limit exceeded.",
    },
    500: {
        "model": ErrorMessageResponse,
        "description": "Internal Server Error - Unknown or unhandled error.",
    },
    502: {
        "model": ErrorMessageResponse,
        "description": "Bad Gateway - Connection error or empty response from the LLM provider.",
    },
    504: {
        "model": ErrorMessageResponse,
        "description": "Gateway Timeout - Timeout error when connecting to the LLM provider.",
    },
}
LLM_CLIENT_ERROR_RESPONSES: dict[int, dict] = {
    code: _LLM_STATUS_DESCRIPTIONS[code] for code in set(LLM_ERROR_STATUS_MAP.values())
}

# MiddlewareError -> status codes derived from MIDDLEWARE_ERROR_STATUS_MAP values
_MIDDLEWARE_STATUS_DESCRIPTIONS: dict[int, dict] = {
    403: {
        "model": ErrorValidationResponse,
        "description": "Forbidden - The request was rejected by safety filters.",
    },
    409: {
        "model": ErrorValidationResponse,
        "description": "Conflict - The middleware is not in a valid state for this request (busy generating or game context not set).",
    },
    503: {
        "model": ErrorValidationResponse,
        "description": "Service Unavailable - Middleware is starting up or setting context, not ready to serve requests.",
    },
}
MIDDLEWARE_ERROR_RESPONSES: dict[int, dict] = {
    code: _MIDDLEWARE_STATUS_DESCRIPTIONS[code] for code in set(MIDDLEWARE_ERROR_STATUS_MAP.values())
}

# All error-response blocks combined, for endpoints that can raise any of them
ALL_ERROR_RESPONSES: dict[int, dict] = {
    **ROUTING_CONFIG_ERROR_RESPONSES,
    **PREPROCESSING_ERROR_RESPONSES,
    **LLM_CLIENT_ERROR_RESPONSES,
    **MIDDLEWARE_ERROR_RESPONSES,
}

# --------------------------------------------------------------------------- #
# Helper to combine error-response blocks in an endpoint's `responses` dict
# --------------------------------------------------------------------------- #
def error_responses(*blocks: dict[int, dict]) -> dict[int, dict]:
    """
    Merge one or more error-response blocks (e.g. LLM_CLIENT_ERROR_RESPONSES,
    MIDDLEWARE_ERROR_RESPONSES) into a single dict suitable for FastAPI's
    `responses` parameter.

    Example:
        responses={
            **error_responses(ROUTING_CONFIG_ERROR_RESPONSES, LLM_CLIENT_ERROR_RESPONSES),
            200: {"description": "Stream of dialogue text."},
        }
    """
    merged: dict[int, dict] = {}
    for block in blocks:
        merged.update(block)
    return merged