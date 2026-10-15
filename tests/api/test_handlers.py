import logging
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from api.handlers import (
    register_exception_handlers,
    preprocessing_error_handler,
    llm_client_error_handler,
    routing_config_error_handler,
    middleware_error_handler,
    _short,
    LLM_ERROR_STATUS_MAP,
    MIDDLEWARE_ERROR_STATUS_MAP,
)
from core.tools.errors import (
    PreProcessingError,
    LLMClientError,
    LLMClientErrorCode,
    RoutingConfigError,
    RoutingConfigErrorCode,
    MiddlewareError,
    MiddlewareErrorCode,
    ValidationErrorCode,
)

app = FastAPI()
register_exception_handlers(app)


@app.get("/test-preprocessing-error")
async def raise_preprocessing_error():
    raise PreProcessingError(code=ValidationErrorCode.INVALID_VALUE, errors=["invalid field value"])


@app.get("/test-llm-error/{error_type}")
async def raise_llm_error(error_type: str):
    code = LLMClientErrorCode(error_type)
    raise LLMClientError(code=code, message="LLM error occurred")


@app.get("/test-routing-error")
async def raise_routing_error():
    raise RoutingConfigError(
        code=RoutingConfigErrorCode.MISSING_INTENDED_TIER, errors=["no default model"]
    )


@app.get("/test-middleware-error/{error_type}")
async def raise_middleware_error(error_type: str):
    code = MiddlewareErrorCode(error_type)
    raise MiddlewareError(code=code, errors=["Middleware error occurred"])


client = TestClient(app)


def test_preprocessing_error_handler():
    response = client.get("/test-preprocessing-error")
    assert response.status_code == 422
    assert response.json() == {
        "error_code": ValidationErrorCode.INVALID_VALUE.value,
        "errors": ["invalid field value"],
    }


@pytest.mark.parametrize(
    "error_code, expected_status",
    [
        (LLMClientErrorCode.CONNECTION_ERROR.value, 502),
        (LLMClientErrorCode.AUTHENTICATION_ERROR.value, 401),
        (LLMClientErrorCode.RATE_LIMIT_ERROR.value, 429),
        (LLMClientErrorCode.TIMEOUT_ERROR.value, 504),
        (LLMClientErrorCode.EMPTY_RESPONSE.value, 502),
        (LLMClientErrorCode.UNKNOWN_ERROR.value, 500),
    ],
)
def test_llm_client_error_handler(error_code, expected_status):
    response = client.get(f"/test-llm-error/{error_code}")
    assert response.status_code == expected_status
    assert response.json() == {
        "error_code": error_code,
        "message": "LLM error occurred",
    }


def test_llm_client_error_handler_unmapped_code():
    """Unmapped LLM error code should default to 500 Internal Server Error."""
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url.path = "/test"

    dummy_code = MagicMock()
    dummy_code.value = "custom_unknown_code"
    exc = LLMClientError(code=dummy_code, message="Unexpected LLM glitch")

    import asyncio

    response = asyncio.run(llm_client_error_handler(request, exc))
    assert response.status_code == 500
    assert response.body == b'{"error_code":"custom_unknown_code","message":"Unexpected LLM glitch"}'


def test_routing_config_error_handler():
    response = client.get("/test-routing-error")
    assert response.status_code == 400
    assert response.json() == {
        "error_code": RoutingConfigErrorCode.MISSING_INTENDED_TIER.value,
        "errors": ["no default model"],
    }


@pytest.mark.parametrize(
    "error_code, expected_status",
    [
        (MiddlewareErrorCode.STARTING.value, 503),
        (MiddlewareErrorCode.SETTING_CONTEXT.value, 503),
        (MiddlewareErrorCode.CONTEXT_NOT_SET.value, 409),
        (MiddlewareErrorCode.REFUSED.value, 403),
        (MiddlewareErrorCode.GENERATING.value, 409),
        (MiddlewareErrorCode.INVALID_RESPONSE.value, 502),
    ],
)
def test_middleware_error_handler(error_code, expected_status):
    response = client.get(f"/test-middleware-error/{error_code}")
    assert response.status_code == expected_status
    assert response.json() == {
        "error_code": error_code,
        "errors": ["Middleware error occurred"],
    }


def test_middleware_error_handler_unmapped_code():
    """Unmapped middleware error code should default to 500 Internal Server Error."""
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url.path = "/status"

    dummy_code = MagicMock()
    dummy_code.value = "unknown_state_error"
    exc = MiddlewareError(code=dummy_code, errors=["Unknown failure"])

    import asyncio

    response = asyncio.run(middleware_error_handler(request, exc))
    assert response.status_code == 500
    assert response.body == b'{"error_code":"unknown_state_error","errors":["Unknown failure"]}'


@pytest.mark.parametrize(
    "code, expected_log_level",
    [
        (MiddlewareErrorCode.STARTING, logging.ERROR),  # 503 >= 500
        (MiddlewareErrorCode.INVALID_RESPONSE, logging.ERROR),  # 502 >= 500
        (MiddlewareErrorCode.REFUSED, logging.WARNING),  # 403 < 500
        (MiddlewareErrorCode.CONTEXT_NOT_SET, logging.WARNING),  # 409 < 500
    ],
)
def test_middleware_error_logging_level(code, expected_log_level):
    request = MagicMock(spec=Request)
    request.method = "POST"
    request.url.path = "/generate"

    exc = MiddlewareError(code=code, errors=["test error"])

    import asyncio

    with patch("api.handlers.logger.log") as mock_log:
        asyncio.run(middleware_error_handler(request, exc))
        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == expected_log_level


def test_register_exception_handlers():
    custom_app = FastAPI()
    register_exception_handlers(custom_app)

    assert PreProcessingError in custom_app.exception_handlers
    assert LLMClientError in custom_app.exception_handlers
    assert RoutingConfigError in custom_app.exception_handlers
    assert MiddlewareError in custom_app.exception_handlers


def test_short_helper_string():
    # String within limit
    short_str = "Short message"
    assert _short(short_str, limit=50) == "Short message"

    # String exceeding limit
    long_str = "A" * 350
    result = _short(long_str, limit=300)
    assert result == f"{'A' * 300}… [350 chars]"


def test_short_helper_list():
    # List within limit
    items = ["item 1", "item 2", "item 3"]
    assert _short(items, limit=50) == "item 1; item 2; item 3"

    # List exceeding limit
    long_items = ["Long item " + str(i) for i in range(50)]
    joined = "; ".join(long_items)
    result = _short(long_items, limit=100)
    assert result == f"{joined[:100]}… [{len(joined)} chars]"

    # Non-string list items
    mixed = [123, 456, True]
    assert _short(mixed, limit=50) == "123; 456; True"
