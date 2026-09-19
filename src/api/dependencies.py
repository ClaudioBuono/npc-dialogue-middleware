from fastapi import Request
from core.dialogue_service import DialogueService
from core.orchestrator import Orchestrator


def get_dialogue_service(request: Request) -> DialogueService:
    return request.app.state.dialogue_service


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator