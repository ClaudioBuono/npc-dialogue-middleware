import pytest
from unittest.mock import MagicMock
from core.configuration.settings import Settings
from core.generation.dialogue_generator import DialogueGenerator
from core.llm.openai_client import OpenAICompatibleClient
from core.tools.errors import LLMClientError, LLMClientErrorCode
from core.types.dataclasses import Contract


@pytest.fixture
def mock_client():
    client = MagicMock(spec=OpenAICompatibleClient)
    client.generate.return_value = '{"dialogue": "Hello!"}'
    client.generate_streaming.return_value = iter(['{"dialogue"', ': "Hello!"}'])
    return client


@pytest.fixture
def generator(mock_client):
    gen = DialogueGenerator()
    gen.set_client(mock_client)
    return gen


@pytest.fixture
def dummy_contract():
    return Contract(
        system_prompt="You are an NPC",
        user_prompt="Say hi",
        output_schema={"type": "object"},
    )


def test_initial_state_has_no_client():
    gen = DialogueGenerator()
    assert gen._client is None


def test_generate_happy_path(generator, mock_client, dummy_contract):
    result = generator.generate(dummy_contract)
    assert result == '{"dialogue": "Hello!"}'
    expected_temp = Settings().llm.dialogue_generator_temperature
    mock_client.generate.assert_called_once_with(dummy_contract, temperature=expected_temp)


def test_generate_stream_happy_path(generator, mock_client, dummy_contract):
    stream = generator.generate_stream(dummy_contract)
    result = list(stream)
    assert result == ['{"dialogue"', ': "Hello!"}']
    expected_temp = Settings().llm.dialogue_generator_temperature
    mock_client.generate_streaming.assert_called_once_with(dummy_contract, temperature=expected_temp)


def test_set_client_replaces_client(generator, dummy_contract):
    new_client = MagicMock(spec=OpenAICompatibleClient)
    new_client.generate.return_value = '{"dialogue": "New client response"}'

    generator.set_client(new_client)
    result = generator.generate(dummy_contract)

    assert result == '{"dialogue": "New client response"}'
    new_client.generate.assert_called_once()


def test_generate_without_client_raises_attribute_error(dummy_contract):
    gen = DialogueGenerator()
    with pytest.raises(AttributeError):
        gen.generate(dummy_contract)


def test_generate_stream_without_client_raises_attribute_error(dummy_contract):
    gen = DialogueGenerator()
    with pytest.raises(AttributeError):
        gen.generate_stream(dummy_contract)


def test_generate_propagates_client_exception(generator, mock_client, dummy_contract):
    mock_client.generate.side_effect = LLMClientError(
        code=LLMClientErrorCode.CONNECTION_ERROR,
        message="Connection failed",
    )
    with pytest.raises(LLMClientError, match="Connection failed"):
        generator.generate(dummy_contract)


def test_generate_stream_propagates_client_exception(generator, mock_client, dummy_contract):
    mock_client.generate_streaming.side_effect = LLMClientError(
        code=LLMClientErrorCode.TIMEOUT_ERROR,
        message="Request timed out",
    )
    with pytest.raises(LLMClientError, match="Request timed out"):
        generator.generate_stream(dummy_contract)


def test_generate_stream_empty_iterator(generator, mock_client, dummy_contract):
    mock_client.generate_streaming.return_value = iter([])
    stream = generator.generate_stream(dummy_contract)
    assert list(stream) == []
