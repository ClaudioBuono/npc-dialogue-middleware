import logging
import sys
import json
import dataclasses
from pydantic import BaseModel

def setup_logging(level=logging.INFO):
    """
    Configure logging for the entire project.
    Sets up both a stream handler for stdout and a file handler for a log file.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Prevent adding handlers multiple times if called more than once
    if logger.handlers:
        return

    # Date format: dd-mm-yyyy HH:MM:SS
    date_format = "%d-%m-%Y %H:%M:%S"
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt=date_format
    )

    # Console Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Suppress noisy third-party debug logs
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)

    # File Handler
    # TODO: replace with the actual telemetry store once implemented
    file_handler = logging.FileHandler("npc_middleware.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def to_json_format(obj) -> str:
    """Pretty prints an object (dict, dataclass, pydantic model) as a JSON string."""
    def custom_encoder(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, BaseModel):
            return o.model_dump()
        return str(o)
    return json.dumps(obj, default=custom_encoder, indent=2)
