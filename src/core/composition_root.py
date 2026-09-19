from core.guardrail import Guardrail
from core.tools.history import DialogueHistory
from core.orchestrator import Orchestrator
from core.generate_service import GenerateService


def build_orchestrator() -> Orchestrator:
    """
    Builds the orchestrator instantiting the shared components.
    """
    guardrail = Guardrail()
    dialogue_history = DialogueHistory()

    orchestrator = Orchestrator(
        guardrail=guardrail,
        dialogue_history=dialogue_history,
    )
    return orchestrator


def build_generate_service(orchestrator: Orchestrator) -> GenerateService:
    """
    Builds the DialogueService using the orchestrator instance.
    """
    return GenerateService(
        orchestrator=orchestrator,
        guardrail=orchestrator.guardrail,
        dialogue_history=orchestrator.dialogue_history,
    )