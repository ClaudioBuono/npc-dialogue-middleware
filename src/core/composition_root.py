from core.contract_builder import ContractBuilder
from core.dialogue_generator import DialogueGenerator
from core.guardrail import Guardrail
from core.judger import Judger
from core.routing.router import LLMRouter
from core.tools.history import DialogueHistory
from core.orchestrator import Orchestrator
from core.generate_service import GenerateService
from core.tools.output_composer import DialogueOutputComposer


def build_orchestrator() -> Orchestrator:
    """
    Builds the orchestrator instantiting the shared components.
    """
    contract_builder = ContractBuilder()
    llm_router = LLMRouter()
    dialogue_generator = DialogueGenerator()
    dialogue_composer = DialogueOutputComposer()
    dialogue_history = DialogueHistory()
    guardrail = Guardrail()
    judger = Judger(contract_builder)

    orchestrator = Orchestrator(
        contract_builder = contract_builder,
        llm_router = llm_router,
        dialogue_generator = dialogue_generator,
        dialogue_composer = dialogue_composer,
        dialogue_history = dialogue_history,
        guardrail = guardrail,
        judger = judger
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