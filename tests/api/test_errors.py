import pytest
from pydantic import ValidationError

from api.errors import (
    ErrorMessageResponse,
    ErrorValidationResponse,
    ROUTING_CONFIG_ERROR_RESPONSES,
    PREPROCESSING_ERROR_RESPONSES,
    LLM_CLIENT_ERROR_RESPONSES,
    MIDDLEWARE_ERROR_RESPONSES,
    ALL_ERROR_RESPONSES,
    error_responses,
)
from api.handlers import LLM_ERROR_STATUS_MAP, MIDDLEWARE_ERROR_STATUS_MAP


# --------------------------------------------------------------------------- #
# Error Response Models
# --------------------------------------------------------------------------- #
def test_error_message_response_valid():
    model = ErrorMessageResponse(error_code=401, message="Unauthorized access")
    assert model.error_code == 401
    assert model.message == "Unauthorized access"


def test_error_message_response_invalid():
    with pytest.raises(ValidationError):
        ErrorMessageResponse(error_code=401)  # missing message

    with pytest.raises(ValidationError):
        ErrorMessageResponse(message="No code")  # missing error_code


def test_error_validation_response_valid():
    model = ErrorValidationResponse(
        error_code=422,
        errors=[{"field": "name", "reason": "empty"}],
    )
    assert model.error_code == 422
    assert model.errors == [{"field": "name", "reason": "empty"}]


def test_error_validation_response_invalid():
    with pytest.raises(ValidationError):
        ErrorValidationResponse(error_code=422)  # missing errors

    with pytest.raises(ValidationError):
        ErrorValidationResponse(errors=[{"field": "name"}])  # missing error_code


# --------------------------------------------------------------------------- #
# Error Response Dictionaries
# --------------------------------------------------------------------------- #
def test_routing_config_error_responses():
    assert 400 in ROUTING_CONFIG_ERROR_RESPONSES
    assert ROUTING_CONFIG_ERROR_RESPONSES[400]["model"] is ErrorValidationResponse
    assert "Bad Request" in ROUTING_CONFIG_ERROR_RESPONSES[400]["description"]


def test_preprocessing_error_responses():
    assert 422 in PREPROCESSING_ERROR_RESPONSES
    assert PREPROCESSING_ERROR_RESPONSES[422]["model"] is ErrorValidationResponse
    assert "Unprocessable Entity" in PREPROCESSING_ERROR_RESPONSES[422]["description"]


def test_llm_client_error_responses_coverage():
    expected_status_codes = set(LLM_ERROR_STATUS_MAP.values())
    assert set(LLM_CLIENT_ERROR_RESPONSES.keys()) == expected_status_codes

    for code, spec in LLM_CLIENT_ERROR_RESPONSES.items():
        assert spec["model"] is ErrorMessageResponse
        assert isinstance(spec["description"], str)
        assert len(spec["description"]) > 0


def test_middleware_error_responses_coverage():
    expected_status_codes = set(MIDDLEWARE_ERROR_STATUS_MAP.values())
    assert set(MIDDLEWARE_ERROR_RESPONSES.keys()) == expected_status_codes

    for code, spec in MIDDLEWARE_ERROR_RESPONSES.items():
        assert spec["model"] is ErrorValidationResponse
        assert isinstance(spec["description"], str)
        assert len(spec["description"]) > 0


def test_all_error_responses_combination():
    expected_keys = (
        set(ROUTING_CONFIG_ERROR_RESPONSES.keys())
        | set(PREPROCESSING_ERROR_RESPONSES.keys())
        | set(LLM_CLIENT_ERROR_RESPONSES.keys())
        | set(MIDDLEWARE_ERROR_RESPONSES.keys())
    )
    assert set(ALL_ERROR_RESPONSES.keys()) == expected_keys


# --------------------------------------------------------------------------- #
# error_responses helper function
# --------------------------------------------------------------------------- #
def test_error_responses_empty():
    assert error_responses() == {}


def test_error_responses_single_block():
    result = error_responses(ROUTING_CONFIG_ERROR_RESPONSES)
    assert result == ROUTING_CONFIG_ERROR_RESPONSES


def test_error_responses_multiple_blocks():
    result = error_responses(
        ROUTING_CONFIG_ERROR_RESPONSES,
        PREPROCESSING_ERROR_RESPONSES,
    )
    assert 400 in result
    assert 422 in result
    assert result[400] == ROUTING_CONFIG_ERROR_RESPONSES[400]
    assert result[422] == PREPROCESSING_ERROR_RESPONSES[422]


def test_error_responses_override():
    block1 = {400: {"desc": "first"}, 500: {"desc": "server"}}
    block2 = {400: {"desc": "overridden"}, 404: {"desc": "not found"}}

    result = error_responses(block1, block2)
    assert result[400]["desc"] == "overridden"
    assert result[500]["desc"] == "server"
    assert result[404]["desc"] == "not found"
