from fastapi import Request
from core.generate_service import GenerateService
from core.orchestrator import Orchestrator


def get_generate_service(request: Request) -> GenerateService:
    """Retrieves the `GenerateService` instance from the application state."""
    return request.app.state.dialogue_service
