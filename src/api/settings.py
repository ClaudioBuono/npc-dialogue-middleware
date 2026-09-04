from fastapi import APIRouter
from api.schemas import LanguageRequest
from core.config.settings import Settings

router = APIRouter(prefix="/settings", tags=["settings"])

@router.post("/language") 
def change_language(language: LanguageRequest):
	Settings().change_language(language.language)

@router.post("/profanity-filter")
def toggle_profanity_filter(toggle: bool):
	Settings().toggle_profanity_filter(toggle)
