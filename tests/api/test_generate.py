import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.generate import router as generate_router
from api.dependencies import get_generate_service
from api.handlers import register_exception_handlers
from api.schemas import ComposedDialogue
from core.generation.generate_service import GenerateService
from core.tools.errors import (
    MiddlewareError,
    MiddlewareErrorCode,
    PreProcessingError,
    LLMClientError,
    LLMClientErrorCode,
    ValidationErrorCode,
)
from core.types.contexts import Dialogue, GameContext, NPCContext, Talkativeness
from core.types.dataclasses import DialogueStream
from core.types.enums import MiddlewareState


@pytest.fixture
def mock_generate_service():
    return MagicMock(spec=GenerateService)


@pytest.fixture
def app(mock_generate_service):
    app = FastAPI()
    app.include_router(generate_router)
    register_exception_handlers(app)
    app.dependency_overrides[get_generate_service] = lambda: mock_generate_service
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def sample_npc_context_dict():
    return {
        "name": "Eldrin",
        "age": 120,
        "personality": "Wise and patient",
        "context": "Inside the ancient library",
        "talkativeness": "Balanced",
        "main_character_relation": "Respected scholar",
        "intent": {
            "type": "Dialogue",
            "must_use_expression": "The archives hold many secrets.",
            "number_of_options": 2,
        },
    }


@pytest.fixture
def sample_game_context_dict():
    return {
        "epoch": "High Fantasy",
        "environment": "Tower of knowledge, stormy night",
        "world_state": "The Great Calamity has passed",
        "main_character_description": "A traveling archivist",
    }


# --------------------------------------------------------------------------- #
# GET /health
# --------------------------------------------------------------------------- #
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# GET /status
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "middleware_state, expected_status, expected_error_code, expected_message",
    [
        (MiddlewareState.IDLE, 200, None, None),
        (MiddlewareState.STARTING, 503, "starting", "The middleware is starting."),
        (
            MiddlewareState.SETTING_CONTEXT,
            503,
            "setting_context",
            "The middleware is setting the game context.",
        ),
        (
            MiddlewareState.GENERATING,
            409,
            "generating",
            "The middleware is generating the dialogue.",
        ),
    ],
)
def test_middleware_status_known_states(
    client, middleware_state, expected_status, expected_error_code, expected_message
):
    with patch("api.generate.StateManager") as mock_sm_cls:
        mock_sm_instance = MagicMock()
        mock_sm_instance.state = middleware_state
        mock_sm_cls.return_value = mock_sm_instance

        response = client.get("/status")
        assert response.status_code == expected_status
        data = response.json()
        assert data["state"] == middleware_state.value
        if expected_status == 200:
            assert "error_code" not in data
            assert "message" not in data
        else:
            assert data["error_code"] == expected_error_code
            assert data["message"] == expected_message


def test_middleware_status_unknown_state(client):
    with patch("api.generate.StateManager") as mock_sm_cls:
        mock_sm_instance = MagicMock()
        mock_unknown_state = MagicMock()
        mock_unknown_state.value = "CUSTOM_UNKNOWN"
        mock_sm_instance.state = mock_unknown_state
        mock_sm_cls.return_value = mock_sm_instance

        response = client.get("/status")
        assert response.status_code == 503
        data = response.json()
        assert data["state"] == "CUSTOM_UNKNOWN"
        assert data["error_code"] is None
        assert data["message"] == "Unknown middleware state: CUSTOM_UNKNOWN"


# --------------------------------------------------------------------------- #
# POST /set-game-context
# --------------------------------------------------------------------------- #
def test_set_game_context_success(client, mock_generate_service, sample_game_context_dict):
    response = client.post("/set-game-context", json=sample_game_context_dict)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_generate_service.set_game_context.assert_called_once()
    called_arg = mock_generate_service.set_game_context.call_args[0][0]
    assert isinstance(called_arg, GameContext)
    assert called_arg.epoch == sample_game_context_dict["epoch"]


def test_set_game_context_invalid_payload(client):
    # Missing required fields like epoch, environment, world_state
    response = client.post("/set-game-context", json={"epoch": "Medieval"})
    assert response.status_code == 422


def test_set_game_context_middleware_error(client, mock_generate_service, sample_game_context_dict):
    mock_generate_service.set_game_context.side_effect = MiddlewareError(
        code=MiddlewareErrorCode.REFUSED,
        errors=["The middleware refused the game context."],
    )

    response = client.post("/set-game-context", json=sample_game_context_dict)
    assert response.status_code == 403
    assert response.json() == {
        "error_code": MiddlewareErrorCode.REFUSED.value,
        "errors": ["The middleware refused the game context."],
    }


def test_set_game_context_preprocessing_error(client, mock_generate_service, sample_game_context_dict):
    mock_generate_service.set_game_context.side_effect = PreProcessingError(
        code=ValidationErrorCode.INVALID_VALUE,
        errors=["Game context validation failed."],
    )

    response = client.post("/set-game-context", json=sample_game_context_dict)
    assert response.status_code == 422
    assert response.json() == {
        "error_code": ValidationErrorCode.INVALID_VALUE.value,
        "errors": ["Game context validation failed."],
    }


# --------------------------------------------------------------------------- #
# POST /generate-dialogue (Non-Streaming)
# --------------------------------------------------------------------------- #
def test_generate_dialogue_non_streaming_success(
    client, mock_generate_service, sample_npc_context_dict
):
    mock_dialogue = ComposedDialogue(
        intent=Dialogue(),
        dialogue="Welcome to the ancient library.",
    )
    mock_generate_service.generate_dialogue.return_value = mock_dialogue

    payload = {
        "npc_context": sample_npc_context_dict,
        "last_player_choice": "Hello there",
        "stream": False,
    }

    response = client.post("/generate-dialogue", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dialogue"] == "Welcome to the ancient library."
    assert data["intent"]["type"] == "Dialogue"

    mock_generate_service.generate_dialogue.assert_called_once()
    npc_arg, choice_arg = mock_generate_service.generate_dialogue.call_args[0]
    assert isinstance(npc_arg, NPCContext)
    assert npc_arg.name == sample_npc_context_dict["name"]
    assert choice_arg == "Hello there"


def test_generate_dialogue_returns_none_raises_refused(
    client, mock_generate_service, sample_npc_context_dict
):
    """When service.generate_dialogue returns None, endpoint raises MiddlewareError(REFUSED)."""
    mock_generate_service.generate_dialogue.return_value = None

    payload = {
        "npc_context": sample_npc_context_dict,
        "stream": False,
    }

    response = client.post("/generate-dialogue", json=payload)
    assert response.status_code == 403
    assert response.json() == {
        "error_code": MiddlewareErrorCode.REFUSED.value,
        "errors": ["Dialogue generation was refused."],
    }


def test_generate_dialogue_context_not_set_error(
    client, mock_generate_service, sample_npc_context_dict
):
    mock_generate_service.generate_dialogue.side_effect = MiddlewareError(
        code=MiddlewareErrorCode.CONTEXT_NOT_SET,
        errors=["Game context is not set."],
    )

    payload = {
        "npc_context": sample_npc_context_dict,
        "stream": False,
    }

    response = client.post("/generate-dialogue", json=payload)
    assert response.status_code == 409
    assert response.json() == {
        "error_code": MiddlewareErrorCode.CONTEXT_NOT_SET.value,
        "errors": ["Game context is not set."],
    }


def test_generate_dialogue_llm_client_error(
    client, mock_generate_service, sample_npc_context_dict
):
    mock_generate_service.generate_dialogue.side_effect = LLMClientError(
        code=LLMClientErrorCode.TIMEOUT_ERROR,
        message="Request to LLM timed out.",
    )

    payload = {
        "npc_context": sample_npc_context_dict,
        "stream": False,
    }

    response = client.post("/generate-dialogue", json=payload)
    assert response.status_code == 504
    assert response.json() == {
        "error_code": LLMClientErrorCode.TIMEOUT_ERROR.value,
        "message": "Request to LLM timed out.",
    }


def test_generate_dialogue_invalid_request_payload(client):
    response = client.post("/generate-dialogue", json={"stream": False})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# POST /generate-dialogue (Streaming)
# --------------------------------------------------------------------------- #
def test_generate_dialogue_streaming_success(
    client, mock_generate_service, sample_npc_context_dict
):
    chunks = ["The stars ", "align tonight, ", "traveler."]

    def chunk_generator():
        yield from chunks

    mock_generate_service.generate_dialogue_stream.return_value = DialogueStream(
        chunks=chunk_generator()
    )

    payload = {
        "npc_context": sample_npc_context_dict,
        "last_player_choice": None,
        "stream": True,
    }

    response = client.post("/generate-dialogue", json=payload)
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert response.text == "The stars align tonight, traveler."

    mock_generate_service.generate_dialogue_stream.assert_called_once()
    npc_arg, choice_arg = mock_generate_service.generate_dialogue_stream.call_args[0]
    assert isinstance(npc_arg, NPCContext)
    assert npc_arg.name == sample_npc_context_dict["name"]
    assert choice_arg is None


def test_generate_dialogue_streaming_with_stream_error(
    client, mock_generate_service, sample_npc_context_dict
):
    chunks = ["Beginning line...", "\n[STREAM_ERROR]\n"]

    mock_generate_service.generate_dialogue_stream.return_value = DialogueStream(
        chunks=iter(chunks)
    )

    payload = {
        "npc_context": sample_npc_context_dict,
        "stream": True,
    }

    response = client.post("/generate-dialogue", json=payload)
    assert response.status_code == 200
    assert response.text == "Beginning line...\n[STREAM_ERROR]\n"
