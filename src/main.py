from core.orchestrator import Orchestrator
from tools.llm_handler import LLMHandler, OllamaClient


def main():
    orchestrator = Orchestrator(llm_handler=LLMHandler(client=OllamaClient(model="llama3.2")))
    result = orchestrator.generate_dialogue(name="John", intent="greeting", description="A friendly villager")
    print(result)

if __name__ == "__main__":
    main()
