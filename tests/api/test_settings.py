import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.settings import router as settings_router
from api.handlers import register_exception_handlers
from core.configuration.settings import AppSettings, LLMSettings, Settings
from core.tools.errors import MiddlewareError, MiddlewareErrorCode
from core.types.enums import Language


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(settings_router)
    register_exception_handlers(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# --------------------------------------------------------------------------- #
# GET /settings
# --------------------------------------------------------------------------- #
def test_get_settings(client):
    mock_app_settings = AppSettings(
        language=Language.ENGLISH,
        fairness_filter=True,
        profanity_filter=True,
        censor_word="[CENSORED]",
        refine_dialogue=True,
        refiner_max_iterations=3,
        llm=LLMSettings(
            dialogue_generator_temperature=0.7,
            judger_temperature=0.2,
            healer_temperature=0.4,
        ),
    )

    with patch.object(Settings, "get_current", return_value=mock_app_settings):
        response = client.get("/settings")
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "en"
        assert data["fairness_filter"] is True
        assert data["censor_word"] == "[CENSORED]"
        assert data["refiner_max_iterations"] == 3
        assert data["llm"]["dialogue_generator_temperature"] == 0.7


# --------------------------------------------------------------------------- #
# POST /settings/language
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lang_str, expected_enum",
    [
        ("en", Language.ENGLISH),
        ("it", Language.ITALIAN),
    ],
)
def test_change_language_success(client, lang_str, expected_enum):
    with patch.object(Settings, "change_language") as mock_change:
        response = client.post("/settings/language", json={"language": lang_str})
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_change.assert_called_once_with(expected_enum)


def test_change_language_invalid_payload(client):
    response = client.post("/settings/language", json={"language": "NonExistentLanguage"})
    assert response.status_code == 422


def test_change_language_missing_body(client):
    response = client.post("/settings/language", json={})
    assert response.status_code == 422


def test_change_language_middleware_error(client):
    """When the middleware is not idle, Settings().change_language raises MiddlewareError."""
    with patch.object(
        Settings,
        "change_language",
        side_effect=MiddlewareError(
            code=MiddlewareErrorCode.GENERATING,
            errors=["Language cannot be changed while the middleware is 'Generating'."],
        ),
    ):
        response = client.post("/settings/language", json={"language": "it"})
        assert response.status_code == 409
        assert response.json() == {
            "error_code": MiddlewareErrorCode.GENERATING.value,
            "errors": ["Language cannot be changed while the middleware is 'Generating'."],
        }


# --------------------------------------------------------------------------- #
# POST /settings/fairness-filter
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", [True, False])
def test_toggle_fairness_filter(client, flag):
    with patch.object(Settings, "toggle_fairness_filter") as mock_toggle:
        response = client.post("/settings/fairness-filter", json={"enabled": flag})
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_toggle.assert_called_once_with(flag)


def test_toggle_fairness_filter_invalid(client):
    response = client.post("/settings/fairness-filter", json={})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# POST /settings/llm
# --------------------------------------------------------------------------- #
def test_update_llm_settings_success(client):
    payload = {
        "dialogue_generator_temperature": 0.85,
        "judger_temperature": 0.15,
        "healer_temperature": 0.35,
    }
    with patch.object(Settings, "update_llm_settings") as mock_update:
        response = client.post("/settings/llm", json=payload)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_update.assert_called_once()
        called_arg = mock_update.call_args[0][0]
        assert isinstance(called_arg, LLMSettings)
        assert called_arg.dialogue_generator_temperature == 0.85
        assert called_arg.judger_temperature == 0.15
        assert called_arg.healer_temperature == 0.35


@pytest.mark.parametrize("invalid_temp", [-0.1, 2.1])
def test_update_llm_settings_out_of_bounds(client, invalid_temp):
    payload = {
        "dialogue_generator_temperature": invalid_temp,
        "judger_temperature": 0.2,
        "healer_temperature": 0.4,
    }
    response = client.post("/settings/llm", json=payload)
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# POST /settings/profanity-mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", [True, False])
def test_update_profanity_filter_settings(client, flag):
    with patch.object(Settings, "update_profanity_filter_settings") as mock_update:
        response = client.post("/settings/profanity-filter", json={"enabled": flag})
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_update.assert_called_once_with(flag)


def test_update_profanity_filter_settings_invalid(client):
    response = client.post("/settings/profanity-filter", json={"enabled": "not_a_bool"})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# POST /settings/censor-word
# --------------------------------------------------------------------------- #
def test_update_censor_word_success(client):
    with patch.object(Settings, "update_censor_word") as mock_update:
        response = client.post("/settings/censor-word", json={"word": "[BLEEP]"})
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_update.assert_called_once_with("[BLEEP]")


@pytest.mark.parametrize("word", ["*", "A" * 32])
def test_update_censor_word_boundary(client, word):
    with patch.object(Settings, "update_censor_word") as mock_update:
        response = client.post("/settings/censor-word", json={"word": word})
        assert response.status_code == 200
        mock_update.assert_called_once_with(word)


@pytest.mark.parametrize("invalid_word", ["", "A" * 33])
def test_update_censor_word_invalid(client, invalid_word):
    response = client.post("/settings/censor-word", json={"word": invalid_word})
    assert response.status_code == 422


def test_update_censor_word_missing_payload(client):
    response = client.post("/settings/censor-word", json={})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# POST /settings/refiner-max-iterations
# --------------------------------------------------------------------------- #
def test_update_refiner_max_iterations_success(client):
    with patch.object(Settings, "update_refiner_max_iterations") as mock_update:
        response = client.post("/settings/refiner-max-iterations", json={"value": 5})
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_update.assert_called_once_with(5)


def test_update_refiner_max_iterations_boundary(client):
    with patch.object(Settings, "update_refiner_max_iterations") as mock_update:
        response = client.post("/settings/refiner-max-iterations", json={"value": 1})
        assert response.status_code == 200
        mock_update.assert_called_once_with(1)


@pytest.mark.parametrize("invalid_val", [0, -1, -10])
def test_update_refiner_max_iterations_invalid(client, invalid_val):
    response = client.post("/settings/refiner-max-iterations", json={"value": invalid_val})
    assert response.status_code == 422


def test_update_refiner_max_iterations_missing_payload(client):
    response = client.post("/settings/refiner-max-iterations", json={})
    assert response.status_code == 422
