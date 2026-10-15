import pytest
from unittest.mock import MagicMock
from fastapi import Request
from api.dependencies import get_generate_service
from core.generation.generate_service import GenerateService


def test_get_generate_service_success():
    """Verify get_generate_service retrieves dialogue_service from request.app.state."""
    mock_service = MagicMock(spec=GenerateService)
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.dialogue_service = mock_service

    result = get_generate_service(mock_request)
    assert result is mock_service


def test_get_generate_service_missing_attribute():
    """Verify AttributeError is raised when dialogue_service is not configured on app.state."""
    mock_request = MagicMock(spec=Request)
    del mock_request.app.state.dialogue_service

    with pytest.raises(AttributeError):
        get_generate_service(mock_request)
