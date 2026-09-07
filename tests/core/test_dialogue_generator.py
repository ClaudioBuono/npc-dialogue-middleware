import pytest
from unittest.mock import MagicMock
from core.config.settings import Settings
from core.dialogue_generator import DialogueGenerator
from core.types.dataclasses import Contract
from core.llm.openai_client import OpenAICompatibleClient

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
        output_schema={}
    )

def test_generate(generator, mock_client, dummy_contract):
    result = generator.generate(dummy_contract)
    assert result == '{"dialogue": "Hello!"}'
    mock_client.generate.assert_called_once_with(dummy_contract, temperature=Settings().llm.default_temperature)

def test_generate_stream(generator, mock_client, dummy_contract):
    stream = generator.generate_stream(dummy_contract)
    result = list(stream)
    assert result == ['{"dialogue"', ': "Hello!"}']
    mock_client.generate_streaming.assert_called_once_with(dummy_contract, temperature=0.3)
