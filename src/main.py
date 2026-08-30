import argparse
import logging
from pathlib import Path
from fastapi import FastAPI
import uvicorn
from api import generate, settings
from api.handlers import register_exception_handlers
from core.logger import setup_logging, to_json_format
from core.config.settings import Settings
from core.orchestrator import Orchestrator
from core.paths import get_base_path
from core.routing.registry import ModelRegistry
from core.telemetry import TelemetryStore
from core.types.contexts import *

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8321

def _setup_telemetry_store():
    """Sets-up the telemetry store path with all necessary information."""
    registry = ModelRegistry()  
    model_ids = [m.identifier for m in registry.get_ranked_models()]  

    telemetry_path = get_base_path() / "logs" / "telemetry_snapshot.json"
    TelemetryStore.instance().configure_persistence(telemetry_path)
    TelemetryStore.instance().initialize_models(model_ids)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    app = FastAPI(title="NPC Middleware")
    app.include_router(generate.router)
    app.include_router(settings.router)
    register_exception_handlers(app)
    return app


app = create_app()

def run_debug():
    """Gets executed in place on the main() method when the --debug flag is active."""
    orchestrator = Orchestrator()

    _setup_telemetry_store()

    orchestrator.set_game_context(environment="Dark", epoch="mediaeval age", world_state="An age of darkness with monsters and heroes", main_character_description=None)

    result = orchestrator.generate_dialogue(
        name="Evil John",
        age=32,
        personality="A very rude villager, often drunk. Insults everyone on sight with racial and homophobic slurs.",
        context="Near a tavern",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Stranger",
        intent= Dialogue(),
        last_player_choice=None,
    )

    result = orchestrator.generate_dialogue(
        name="Evil John",
        age=32,
        personality="A very rude villager, often drunk. Insults everyone on sight with racial and homophobic slurs.",
        context="Near a tavern",
        talkativeness=Talkativeness.AVERAGE,
        main_character_relation="Stranger",
        intent= Quest(
            objective="Talk some sense into Evil John",
            description="Evil John has been doin this for far too long. It's time to put an end to his behaviour.",
            reward="An hug from Evil John",  
        ),
        last_player_choice=None,
    )

    if (not result):
        logger.info("Dialogue generation failed.")
    else:
        logger.info("Dialogue generation complete.")
        logger.info(f"Result:\n{to_json_format(result)}")
        print(to_json_format(TelemetryStore.instance().snapshot()))


def main():
    """Parse CLI arguments, initialize configuration/logging, and run the server.

    Command-line arguments:
        --config-dir: Optional path to the folder containing settings.yaml.
        --port: TCP port to bind the server to. Defaults to DEFAULT_PORT.
        --port-file: Optional path to a file where the chosen port is
            written as JSON (e.g. for the game client to read on startup).
        --debug: Executes debug code instead of starting the application.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--port-file", default=None)
    parser.add_argument("--debug", action="store_true", help="Executes debug code instead of starting the application")
    args = parser.parse_args()

    config_dir = Path(args.config_dir) if args.config_dir else get_base_path() / "config"
    Settings.configure(config_dir)
    Settings()  # force loading now, to fail fast on invalid/missing config

    setup_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("Starting middleware")

    _setup_telemetry_store()

    # Run debug code if debug flag is active
    if args.debug:
        run_debug()
        return

    port = args.port

    if args.port_file:
        Path(args.port_file).write_text(f'{{"port": {port}}}')

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()