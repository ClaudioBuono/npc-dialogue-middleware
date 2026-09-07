import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.handlers import register_exception_handlers
from core.tools.errors import (
    PreProcessingError,
    LLMClientError,
    LLMClientErrorCode,
    RoutingConfigError,
    RoutingConfigErrorCode,
    MiddlewareError,
    MiddlewareErrorCode,
    ValidationErrorCode
)

app = FastAPI()
register_exception_handlers(app)

@app.get("/test-preprocessing-error")
async def raise_preprocessing_error():
    raise PreProcessingError(code=ValidationErrorCode.INVALID_VALUE, errors={"field": "value"})

@app.get("/test-llm-error/{error_type}")
async def raise_llm_error(error_type: str):
    code = LLMClientErrorCode(error_type)
    raise LLMClientError(code=code, message="LLM error occurred")

@app.get("/test-routing-error")
async def raise_routing_error():
    raise RoutingConfigError(code=RoutingConfigErrorCode.MISSING_INTENDED_TIER, errors=["no default model"])

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
        "errors": {"field": "value"}
    }

@pytest.mark.parametrize("error_code, expected_status", [
    (LLMClientErrorCode.CONNECTION_ERROR.value, 502),
    (LLMClientErrorCode.AUTHENTICATION_ERROR.value, 401),
    (LLMClientErrorCode.RATE_LIMIT_ERROR.value, 429),
    (LLMClientErrorCode.TIMEOUT_ERROR.value, 504),
    (LLMClientErrorCode.EMPTY_RESPONSE.value, 502),
    (LLMClientErrorCode.UNKNOWN_ERROR.value, 500),
])
def test_llm_client_error_handler(error_code, expected_status):
    response = client.get(f"/test-llm-error/{error_code}")
    assert response.status_code == expected_status
    assert response.json() == {
        "error_code": error_code,
        "message": "LLM error occurred"
    }

def test_routing_config_error_handler():
    response = client.get("/test-routing-error")
    assert response.status_code == 400
    assert response.json() == {
        "error_code": RoutingConfigErrorCode.MISSING_INTENDED_TIER.value,
        "errors": ["no default model"]
    }

@pytest.mark.parametrize("error_code, expected_status", [
    (MiddlewareErrorCode.STARTING.value, 503),
    (MiddlewareErrorCode.SETTING_CONTEXT.value, 503),
    (MiddlewareErrorCode.CONTEXT_NOT_SET.value, 409),
    (MiddlewareErrorCode.REFUSED.value, 403),
    (MiddlewareErrorCode.GENERATING.value, 409),
])
def test_middleware_error_handler(error_code, expected_status):
    response = client.get(f"/test-middleware-error/{error_code}")
    assert response.status_code == expected_status
    assert response.json() == {
        "error_code": error_code,
        "errors": ["Middleware error occurred"]
    }
