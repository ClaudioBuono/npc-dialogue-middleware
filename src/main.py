from core.orchestrator import Orchestrator
from core.types.contexts import Dialogue, Quest, Talkativeness
from core.logger import setup_logging, to_json_format
import logging

# Standard quest
quest1 = Quest(
    name="Necklace",
    description="John's mother has lost her necklace near the lake. John offers the main character to help him finding the necklace.",
    objective="Find the lost necklace near the lake", 
    has_choice=True, 
    has_options=False, 
)

quest2 = Quest(
    name="The Goblin Menace",
    description="Mayor Richard is deeply worried about a pack of goblins raiding supply wagons on the northern trade route. He needs a capable adventurer to thin their ranks.",
    objective="Defeat 5 goblins near the northern crossroads",
    location="Northern Crossroads",
    reward="50 Gold pieces and a Steel Dagger",
    has_choice=True,
    has_options=True,
    more_info="The goblins are rumored to be led by a larger scout wearing a distinctive red cap.",
    must_use_expression="For the safety of our town"
)

quest3 = Quest(
    name="Iron Shortage",
    description="The local blacksmith, Grunthor, is running low on raw materials due to the recent blockade and needs iron ore to keep supplying the town guard.",
    objective="Gather 5 chunks of Iron Ore from the Echoing Caves",
    location="Echoing Caves",
    reward="Custom armor reinforcement (+2 Defense)",
    has_choice=False, # Quest obbligatoria per sbloccare i servizi del fabbro
    has_options=True,
    more_info="The caves are damp and heavily infested with giant cave spiders.",
    must_use_expression="No metal, no masterpieces"
)


def main():
    setup_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("Starting middleware")

    orchestrator = Orchestrator()

    orchestrator.set_game_context(environment="Dark", epoch="mediaeval age", world_state="An age of darkness with monsters and heroes")

    result = orchestrator.generate_dialogue(
        name="John",
        age=32,
        personality="A friendly villager",
        context="Near a tavern",
        talkativeness=Talkativeness.VERY_LOW,
        main_character_relation="Acquaintance",
        intent= quest2,
        last_player_choice=None,
    )
    logger.info("Dialogue generation complete.")
    logger.info(f"Result:\n{to_json_format(result)}")

    result = orchestrator.generate_dialogue(
        name="John",
        age=32,
        personality="A friendly villager",
        context="Near a tavern",
        talkativeness=Talkativeness.VERY_LOW,
        main_character_relation="Acquaintance",
        intent= Dialogue(more_info="Thank John for taking the risk"),
        last_player_choice=result.accept,
    )
    logger.info("Dialogue generation complete.")
    logger.info(f"Result:\n{to_json_format(result)}")


if __name__ == "__main__":
    main()
