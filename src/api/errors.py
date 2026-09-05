from pydantic import BaseModel, Field
from typing import Any, List, Dict

class ErrorMessageResponse(BaseModel):
    """Generic error response model containing an error code and a message."""
    error_code: int = Field(..., description="Internal or specific error code.")
    message: str = Field(..., description="Human-readable error message.")

class ErrorValidationResponse(BaseModel):
    """Validation error response model containing an error code and a list of errors."""
    error_code: int = Field(..., description="Internal or specific error code.")
    errors: List[Dict[str, Any]] = Field(..., description="List of validation errors.")

GLOBAL_ERROR_RESPONSES = {
    400: {
        "model": ErrorValidationResponse,
        "description": "Bad Request - Inconsistent routing configuration or similar bad inputs.",
    },
    401: {
        "model": ErrorMessageResponse,
        "description": "Unauthorized - Authentication error with the LLM provider.",
    },
    409: {
        "model": ErrorValidationResponse,
        "description": "Conflict - Middleware is busy generating a response and cannot process the request.",
    },
    422: {
        "model": ErrorValidationResponse,
        "description": "Unprocessable Entity - Semantic validation errors on the incoming payload.",
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
    503: {
        "model": ErrorValidationResponse,
        "description": "Service Unavailable - Middleware is starting up or setting context, not ready to serve requests.",
    },
    504: {
        "model": ErrorMessageResponse,
        "description": "Gateway Timeout - Timeout error when connecting to the LLM provider.",
    }
}