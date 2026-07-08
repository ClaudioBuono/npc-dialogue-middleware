from core.orchestrator import Orchestrator
from core.types.contexts import Talkativeness


def main():
    orchestrator = Orchestrator()

    orchestrator.set_game_context(environment="Dark", epoch="mediaeval age", plot="An age of darkness with monsters and heroes")

    result = orchestrator.generate_dialogue(
        name="John",
        age=32,
        personality="A friendly villager",
        context="Inside his tavern",
        talkativeness=Talkativeness.VERYLOW,
        main_character_relation="Acquaintance",
        intent=None,
    )
    print(result)

if __name__ == "__main__":
    main()
