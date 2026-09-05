from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from tools.errors import (  
    PreProcessingError,
    LLMClientError,
    LLMClientErrorCode,
    RoutingConfigError,
    MiddlewareError,
    MiddlewareErrorCode,
)


# --------------------------------------------------------------------------- #
# Error code -> HTTP status mapping for LLMClientError
# --------------------------------------------------------------------------- #
_LLM_ERROR_STATUS_MAP: dict[LLMClientErrorCode, int] = {
    LLMClientErrorCode.CONNECTION_ERROR: status.HTTP_502_BAD_GATEWAY,
    LLMClientErrorCode.AUTHENTICATION_ERROR: status.HTTP_401_UNAUTHORIZED,
    LLMClientErrorCode.RATE_LIMIT_ERROR: status.HTTP_429_TOO_MANY_REQUESTS,
    LLMClientErrorCode.TIMEOUT_ERROR: status.HTTP_504_GATEWAY_TIMEOUT,
    LLMClientErrorCode.EMPTY_RESPONSE: status.HTTP_502_BAD_GATEWAY,
    LLMClientErrorCode.UNKNOWN_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


# --------------------------------------------------------------------------- #
# Error code -> HTTP status mapping for MiddlewareError
# --------------------------------------------------------------------------- #
_MIDDLEWARE_ERROR_STATUS_MAP: dict[MiddlewareErrorCode, int] = {
    MiddlewareErrorCode.STARTING: status.HTTP_503_SERVICE_UNAVAILABLE,
    MiddlewareErrorCode.SETTING_CONTEXT: status.HTTP_503_SERVICE_UNAVAILABLE,
    MiddlewareErrorCode.GENERATING: status.HTTP_409_CONFLICT,
}


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
async def preprocessing_error_handler(request: Request, exc: PreProcessingError) -> JSONResponse:
    """422 - semantic validation errors on the incoming payload."""
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": exc.code.value,
            "errors": exc.errors,
        },
    )


async def llm_client_error_handler(request: Request, exc: LLMClientError) -> JSONResponse:
    """Status code varies depending on the type of LLM client failure."""
    
    http_status = _LLM_ERROR_STATUS_MAP.get(
        exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return JSONResponse(
        status_code=http_status,
        content={
            "error_code": exc.code.value,
            "message": exc.message,
        },
    )


async def routing_config_error_handler(request: Request, exc: RoutingConfigError) -> JSONResponse:
    """400 - inconsistent model routing configuration."""
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": exc.code.value,
            "errors": exc.errors,
        },
    )


async def middleware_error_handler(request: Request, exc: MiddlewareError) -> JSONResponse:
    """Status code varies depending on the type of middleware failure."""

    http_status = _MIDDLEWARE_ERROR_STATUS_MAP.get(
        exc.code, status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return JSONResponse(
        status_code=http_status,
        content={
            "error_code": exc.code.value,
            "errors": exc.errors,
        },
    )


# --------------------------------------------------------------------------- #
# Centralized registration
# --------------------------------------------------------------------------- #
def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(PreProcessingError, preprocessing_error_handler)
    app.add_exception_handler(LLMClientError, llm_client_error_handler)
    app.add_exception_handler(RoutingConfigError, routing_config_error_handler)
    app.add_exception_handler(MiddlewareError, middleware_error_handler)
    