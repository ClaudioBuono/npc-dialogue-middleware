import pytest
from unittest.mock import MagicMock, patch
from openai import AuthenticationError, RateLimitError, APITimeoutError
from core.llm.openai_client import OpenAICompatibleClient
from core.types.dataclasses import Contract
from core.tools.errors import LLMClientError, LLMClientErrorCode

@pytest.fixture
def dummy_contract():
    return Contract(
        system_prompt="sys",
        user_prompt="user",
        output_schema={"type": "object", "properties": {"msg": {"type": "string"}}}
    )

@pytest.fixture
def mock_openai_client():
    with patch("core.llm.openai_client.OpenAI") as mock_openai:
        mock_instance = mock_openai.return_value
        yield mock_instance

@pytest.fixture
def client(mock_openai_client):
    return OpenAICompatibleClient(endpoint="http://localhost:1234", api_key="test", model_identifier="test-model")

@patch("core.llm.openai_client.TelemetryRecorder")
def test_generate_success(mock_telemetry, client, mock_openai_client, dummy_contract):
    # Setup mock response
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = '{"msg": "hello"}'
    mock_chunk.usage = None
    
    mock_openai_client.chat.completions.create.return_value = [mock_chunk]
    
    result = client.generate(dummy_contract, temperature=0.7)
    
    assert result == '{"msg": "hello"}'
    mock_openai_client.chat.completions.create.assert_called_once()
    
    kwargs = mock_openai_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0.7
    assert kwargs["stream"] is True
    assert "response_format" in kwargs

@patch("core.llm.openai_client.TelemetryRecorder")
def test_generate_auth_error(mock_telemetry, client, mock_openai_client, dummy_contract):
    mock_openai_client.chat.completions.create.side_effect = AuthenticationError(
        message="Auth failed", response=MagicMock(), body=None
    )
    
    with pytest.raises(LLMClientError) as exc_info:
        client.generate(dummy_contract, temperature=0.7)
        
    assert exc_info.value.code == LLMClientErrorCode.AUTHENTICATION_ERROR

@patch("core.llm.openai_client.TelemetryRecorder")
def test_generate_rate_limit(mock_telemetry, client, mock_openai_client, dummy_contract):
    mock_openai_client.chat.completions.create.side_effect = RateLimitError(
        message="Too many requests", response=MagicMock(), body=None
    )
    
    with pytest.raises(LLMClientError) as exc_info:
        client.generate(dummy_contract, temperature=0.7)
        
    assert exc_info.value.code == LLMClientErrorCode.RATE_LIMIT_ERROR

def test_generate_streaming_success(client, mock_openai_client, dummy_contract):
    # Setup mock response chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = '{"msg": '
    
    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = '"hi"}'
    
    mock_openai_client.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])
    
    stream = client.generate_streaming(dummy_contract, temperature=0.5)
    result = list(stream)
    
    assert result == ['{"msg": ', '"hi"}']
    
def test_empty_response(client, mock_openai_client, dummy_contract):
    mock_openai_client.chat.completions.create.return_value = [] # Empty stream
    
    with pytest.raises(LLMClientError) as exc_info:
        client.generate(dummy_contract, temperature=0.7)
        
    assert exc_info.value.code == LLMClientErrorCode.EMPTY_RESPONSE
