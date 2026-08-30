import argparse
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
import uvicorn

from api import generate, settings
from api.handlers import register_exception_handlers
from core.logger import setup_logging
from core.config.settings import Settings
from core.paths import get_base_path


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance.

    Registers all API routers (dialogue generation, settings) and the
    global exception handlers.

    Returns:
        FastAPI: The fully configured application, ready to be served.
    """
    app = FastAPI(title="NPC Middleware")
    app.include_router(generate.router)
    app.include_router(settings.router)
    register_exception_handlers(app)
    return app


app = create_app()


DEFAULT_PORT = 8321


def main():
    """Parse CLI arguments, initialize configuration/logging, and run the server.

    Command-line arguments:
        --config-dir: Optional path to the folder containing settings.yaml.
            Defaults to a "config" folder next to the running executable
            (or main.py in development).
        --port: TCP port to bind the server to. Defaults to DEFAULT_PORT.
        --port-file: Optional path to a file where the chosen port is
            written as JSON (e.g. for the game client to read on startup).

    The configuration is loaded and validated before logging is set up,
    and before Uvicorn starts serving requests, so that invalid config or
    missing files fail fast with a clear error instead of failing later
    at an unpredictable point during request handling.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--port-file", default=None)
    args = parser.parse_args()

    config_dir = Path(args.config_dir) if args.config_dir else get_base_path() / "config"
    Settings.configure(config_dir)
    Settings()  # force loading now, to fail fast on invalid/missing config

    setup_logging(logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("Starting middleware")

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