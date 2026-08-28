from fastapi import APIRouter
from core.config.settings import Settings
from core.types.enums import Language

router = APIRouter(prefix="/settings", tags=["settings"])

@router.post("/language") 
def change_language(language: Language):
	Settings().change_language(language)

@router.post("/profanity-filter")
def toggle_profanity_filter(toggle: bool):
	Settings().toggle_profanity_filter(toggle)
