from fastapi import APIRouter
from api.errors import MIDDLEWARE_ERROR_RESPONSES
from core.config.settings import Settings, AppSettings, LLMSettings
from api.schemas import CensorWordRequest, LanguageRequest, ProfanityModeRequest, ToggleRequest, NumberOfOptionsRequest, SettingsUpdatedResponse
from core.orchestrator import Orchestrator
from core.tools.errors import MiddlewareError, MiddlewareErrorCode

router = APIRouter(prefix="/settings", tags=["settings"])

@router.get(
    "",
    response_model=AppSettings,
    summary="Get Current Settings",
    description="Returns the currently loaded application settings.",
    responses={200: {"description": "Current settings retrieved successfully."}},
)
def get_settings():
    return Settings.get_current()

@router.post(
    "/language",
    response_model=SettingsUpdatedResponse,
    summary="Change Language",
    description="Updates the active application language.",
    responses={200: {"description": "Language updated successfully."}},
)
def change_language(request: LanguageRequest):
    Settings().change_language(request.language)
    return {"status": "ok"}

@router.post(
    "/prompt-fairness-filter",
    response_model=SettingsUpdatedResponse,
    summary="Toggle Prompt Fairness Filter",
    description="Enables or disables the prompt fairness filter applied to incoming requests.",
    responses={200: {"description": "Prompt fairness filter updated successfully."}},
)
def toggle_prompt_fairness_filter(request: ToggleRequest):
    Settings().toggle_prompt_fairness_filter(request.enabled)
    return {"status": "ok"}

@router.post(
    "/number-of-options",
    response_model=SettingsUpdatedResponse,
    summary="Set Number of Dialogue Options",
    description="Sets how many player response options are generated per dialogue turn.",
    responses={200: {"description": "Number of options updated successfully."}},
)
def set_number_of_options(request: NumberOfOptionsRequest):
    Settings().set_number_of_options(request.value)
    return {"status": "ok"}


@router.post(
    "/llm",
    response_model=SettingsUpdatedResponse,
    summary="Update LLM Settings",
    description="Updates the default temperature and max tokens used for dialogue generation.",
    responses={200: {"description": "LLM settings updated successfully."}},
)
def update_llm_settings(request: LLMSettings):
    Settings().update_llm_settings(request)
    return {"status": "ok"}

@router.post(
    "/profanity-mode",
    response_model=SettingsUpdatedResponse,
    summary="Update profanity mode",
    description="Updates the default profanity mode used for dialogue generation.",
    responses={200: {"description": "Profanity mode updated successfully."}},
)
def update_profanity_mode_settings(request: ProfanityModeRequest):
    Settings().update_profanity_mode_settings(request.profanity_mode)
    return {"status": "ok"}

@router.post(
    "/censor-word",
    response_model=SettingsUpdatedResponse,
    summary="Update Censor Word",
    description="Updates the word used to censor filtered content.",
    responses={
        403: MIDDLEWARE_ERROR_RESPONSES[403],
        200: {"description": "Censor word updated successfully."},
    },
)
def update_censor_word(request: CensorWordRequest):
    censor_word: str = request.word
    if not Orchestrator().guardrail.validate_censor_word(censor_word): # TODO: find a better way to do it
        raise MiddlewareError(
            code=MiddlewareErrorCode.REFUSED,
            errors=["The provided censor word is itself a banned word."],
        )
    
    Settings().update_censor_word(censor_word)
    return {"status": "ok"}