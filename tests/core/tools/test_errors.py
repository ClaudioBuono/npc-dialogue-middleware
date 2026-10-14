"""Unit tests for core.tools.errors – error codes and exception classes."""
import pytest

from core.tools.errors import (
    LLMClientError,
    LLMClientErrorCode,
    MiddlewareError,
    MiddlewareErrorCode,
    PreProcessingError,
    RoutingConfigError,
    RoutingConfigErrorCode,
    ValidationErrorCode,
)


# ---------------------------------------------------------------------------
# ValidationErrorCode enum
# ---------------------------------------------------------------------------

def test_validation_error_code_members():
    """All expected members are present and have the correct values."""
    assert ValidationErrorCode.EMPTY_FIELD.value == "empty_field"
    assert ValidationErrorCode.FIELD_TOO_LONG.value == "field_too_long"
    assert ValidationErrorCode.FIELD_TOO_SHORT.value == "field_too_short"
    assert ValidationErrorCode.INVALID_VALUE.value == "invalid_value"


def test_validation_error_code_count():
    """Enum has exactly 4 members – catches accidental additions/removals."""
    assert len(ValidationErrorCode) == 4


# ---------------------------------------------------------------------------
# LLMClientErrorCode enum
# ---------------------------------------------------------------------------

def test_llm_client_error_code_members():
    assert LLMClientErrorCode.CONNECTION_ERROR.value == "connection_error"
    assert LLMClientErrorCode.AUTHENTICATION_ERROR.value == "authentication_error"
    assert LLMClientErrorCode.RATE_LIMIT_ERROR.value == "rate_limit_error"
    assert LLMClientErrorCode.TIMEOUT_ERROR.value == "timeout_error"
    assert LLMClientErrorCode.EMPTY_RESPONSE.value == "empty_response"
    assert LLMClientErrorCode.UNKNOWN_ERROR.value == "unknown_error"


def test_llm_client_error_code_count():
    assert len(LLMClientErrorCode) == 6


# ---------------------------------------------------------------------------
# RoutingConfigErrorCode enum
# ---------------------------------------------------------------------------

def test_routing_config_error_code_members():
    assert RoutingConfigErrorCode.MISSING_INTENDED_TIER.value == "missing_intended_tier"


def test_routing_config_error_code_count():
    assert len(RoutingConfigErrorCode) == 1


# ---------------------------------------------------------------------------
# MiddlewareErrorCode enum
# ---------------------------------------------------------------------------

def test_middleware_error_code_members():
    assert MiddlewareErrorCode.STARTING.value == "starting"
    assert MiddlewareErrorCode.SETTING_CONTEXT.value == "setting_context"
    assert MiddlewareErrorCode.CONTEXT_NOT_SET.value == "context_not_set"
    assert MiddlewareErrorCode.REFUSED.value == "refused"
    assert MiddlewareErrorCode.GENERATING.value == "generating"
    assert MiddlewareErrorCode.INVALID_RESPONSE.value == "invalid_response"


def test_middleware_error_code_count():
    assert len(MiddlewareErrorCode) == 6


# ---------------------------------------------------------------------------
# PreProcessingError
# ---------------------------------------------------------------------------

class TestPreProcessingError:
    """Tests for PreProcessingError construction and message formatting."""

    def _make(self, code=ValidationErrorCode.EMPTY_FIELD, errors=None):
        return PreProcessingError(code, errors or ["field 'name' is empty"])

    def test_is_exception_subclass(self):
        assert issubclass(PreProcessingError, Exception)

    def test_stores_code_attribute(self):
        err = self._make(ValidationErrorCode.FIELD_TOO_LONG, ["too long"])
        assert err.code is ValidationErrorCode.FIELD_TOO_LONG

    def test_stores_errors_list(self):
        errors = ["first error", "second error"]
        err = self._make(errors=errors)
        assert err.errors == errors

    def test_str_contains_code_value(self):
        err = self._make(ValidationErrorCode.INVALID_VALUE, ["bad value"])
        assert "invalid_value" in str(err)

    def test_str_contains_all_error_messages(self):
        errors = ["msg one", "msg two"]
        err = self._make(errors=errors)
        assert "msg one" in str(err)
        assert "msg two" in str(err)

    def test_single_error_message(self):
        err = PreProcessingError(ValidationErrorCode.EMPTY_FIELD, ["only message"])
        assert str(err) == "[empty_field] only message"

    def test_multiple_error_messages_joined_by_semicolon(self):
        err = PreProcessingError(ValidationErrorCode.EMPTY_FIELD, ["a", "b", "c"])
        assert str(err) == "[empty_field] a; b; c"

    def test_can_be_raised_and_caught(self):
        with pytest.raises(PreProcessingError) as exc_info:
            raise PreProcessingError(ValidationErrorCode.EMPTY_FIELD, ["raised"])
        assert exc_info.value.code is ValidationErrorCode.EMPTY_FIELD

    def test_empty_errors_list(self):
        """Edge case: empty list produces a valid (though useless) error string."""
        err = PreProcessingError(ValidationErrorCode.EMPTY_FIELD, [])
        assert str(err) == "[empty_field] "  # join of empty list

    def test_errors_list_is_mutable(self):
        """The stored list is the same object passed in (no defensive copy needed)."""
        errors = ["initial"]
        err = PreProcessingError(ValidationErrorCode.EMPTY_FIELD, errors)
        assert err.errors is errors


# ---------------------------------------------------------------------------
# LLMClientError
# ---------------------------------------------------------------------------

class TestLLMClientError:
    """Tests for LLMClientError construction and message formatting."""

    def _make(self, code=LLMClientErrorCode.UNKNOWN_ERROR, message="something went wrong"):
        return LLMClientError(code, message)

    def test_is_exception_subclass(self):
        assert issubclass(LLMClientError, Exception)

    def test_stores_code_attribute(self):
        err = self._make(LLMClientErrorCode.TIMEOUT_ERROR)
        assert err.code is LLMClientErrorCode.TIMEOUT_ERROR

    def test_stores_message_attribute(self):
        err = self._make(message="request timed out")
        assert err.message == "request timed out"

    def test_str_format(self):
        err = LLMClientError(LLMClientErrorCode.CONNECTION_ERROR, "refused")
        assert str(err) == "[connection_error] refused"

    def test_str_contains_code_value(self):
        err = self._make(LLMClientErrorCode.RATE_LIMIT_ERROR, "too many requests")
        assert "rate_limit_error" in str(err)

    def test_str_contains_message(self):
        err = self._make(message="my custom message")
        assert "my custom message" in str(err)

    def test_can_be_raised_and_caught_as_exception(self):
        with pytest.raises(LLMClientError):
            raise LLMClientError(LLMClientErrorCode.EMPTY_RESPONSE, "nothing")

    def test_empty_message(self):
        err = LLMClientError(LLMClientErrorCode.UNKNOWN_ERROR, "")
        assert str(err) == "[unknown_error] "

    def test_all_codes_constructible(self):
        """Each error code can be used without raising during construction."""
        for code in LLMClientErrorCode:
            err = LLMClientError(code, "test")
            assert err.code is code


# ---------------------------------------------------------------------------
# RoutingConfigError
# ---------------------------------------------------------------------------

class TestRoutingConfigError:
    """Tests for RoutingConfigError construction and message formatting."""

    def test_is_exception_subclass(self):
        assert issubclass(RoutingConfigError, Exception)

    def test_stores_code(self):
        err = RoutingConfigError(RoutingConfigErrorCode.MISSING_INTENDED_TIER, ["missing tier"])
        assert err.code is RoutingConfigErrorCode.MISSING_INTENDED_TIER

    def test_stores_errors(self):
        errors = ["model A has no tier", "model B has no tier"]
        err = RoutingConfigError(RoutingConfigErrorCode.MISSING_INTENDED_TIER, errors)
        assert err.errors == errors

    def test_str_format_single_error(self):
        err = RoutingConfigError(RoutingConfigErrorCode.MISSING_INTENDED_TIER, ["one"])
        assert str(err) == "[missing_intended_tier] one"

    def test_str_format_multiple_errors(self):
        err = RoutingConfigError(
            RoutingConfigErrorCode.MISSING_INTENDED_TIER, ["alpha", "beta"]
        )
        assert str(err) == "[missing_intended_tier] alpha; beta"

    def test_can_be_raised(self):
        with pytest.raises(RoutingConfigError):
            raise RoutingConfigError(RoutingConfigErrorCode.MISSING_INTENDED_TIER, ["x"])


# ---------------------------------------------------------------------------
# MiddlewareError
# ---------------------------------------------------------------------------

class TestMiddlewareError:
    """Tests for MiddlewareError construction and message formatting."""

    def test_is_exception_subclass(self):
        assert issubclass(MiddlewareError, Exception)

    def test_stores_code(self):
        err = MiddlewareError(MiddlewareErrorCode.GENERATING, ["LLM call failed"])
        assert err.code is MiddlewareErrorCode.GENERATING

    def test_stores_errors(self):
        errors = ["issue one", "issue two"]
        err = MiddlewareError(MiddlewareErrorCode.INVALID_RESPONSE, errors)
        assert err.errors == errors

    def test_str_format(self):
        err = MiddlewareError(MiddlewareErrorCode.STARTING, ["boot failure"])
        assert str(err) == "[starting] boot failure"

    def test_str_multiple_errors(self):
        err = MiddlewareError(MiddlewareErrorCode.CONTEXT_NOT_SET, ["a", "b"])
        assert str(err) == "[context_not_set] a; b"

    def test_all_codes_constructible(self):
        for code in MiddlewareErrorCode:
            err = MiddlewareError(code, ["msg"])
            assert err.code is code

    def test_can_be_raised_and_caught(self):
        with pytest.raises(MiddlewareError) as exc_info:
            raise MiddlewareError(MiddlewareErrorCode.REFUSED, ["declined"])
        assert exc_info.value.code is MiddlewareErrorCode.REFUSED
