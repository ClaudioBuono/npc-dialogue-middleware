from core.generation.contract_builder import ContractBuilder
from core.generation.dialogue_generator import DialogueGenerator
from core.pipeline.guardrail import Guardrail
from core.pipeline.healer import Healer
from core.pipeline.judger import Judger
from core.pipeline.refiner import Refiner
from core.routing.router import LLMRouter
from core.generation.history import DialogueHistory
from core.pipeline.orchestrator import Orchestrator
from core.generation.generate_service import GenerateService
from core.generation.output_composer import DialogueOutputComposer


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
    refiner = build_refiner(contract_builder=contract_builder, dialogue_composer=dialogue_composer)


    orchestrator = Orchestrator(
        contract_builder = contract_builder,
        llm_router = llm_router,
        dialogue_generator = dialogue_generator,
        dialogue_composer = dialogue_composer,
        dialogue_history = dialogue_history,
        guardrail = guardrail,
        refiner = refiner
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

def build_refiner(contract_builder: ContractBuilder, dialogue_composer: DialogueOutputComposer) -> Refiner:
    """
    Builds the Refiner.
    """

    healer: Healer = Healer(contract_builder, dialogue_composer)
    judger: Judger = Judger(contract_builder)


    return Refiner(
        healer=healer,
        judger=judger
    )