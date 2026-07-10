from core.orchestrator import Orchestrator
from core.types.contexts import Quest, Talkativeness


def main():
    orchestrator = Orchestrator()

    orchestrator.set_game_context(environment="Dark", epoch="mediaeval age", world_state="An age of darkness with monsters and heroes")

    result = orchestrator.generate_dialogue(
        name="John",
        age=32,
        personality="A friendly villager",
        context="Near a tavern",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Acquaintance",
        intent= Quest(name="Necklace",description="John's mother has lost her necklace near the lake. John offers the main character to help him finding the necklace.",objective="Find the lost necklace near the lake", has_choice=True, has_options=True, more_info="The necklace has a lot of blue gems"),
    )
    print(result)

if __name__ == "__main__":
    main()
