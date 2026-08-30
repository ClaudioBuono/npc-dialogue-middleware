import logging
import sys
import json
import dataclasses
from pathlib import Path
from pydantic import BaseModel

from core.paths import get_base_path

LOG_FILENAME = "npc_middleware.log"


def setup_logging(level=logging.INFO):
    """Configure logging for the entire project.

    Sets up both a stream handler (stdout) and a file handler pointing to
    a fixed log file location, relative to the executable's directory
    (``<exe_dir>/logs/npc_middleware.log``).

    Args:
        level: The minimum logging level for the root logger
            (e.g. logging.DEBUG, logging.INFO).
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
    # Fixed path: <exe_dir>/logs/npc_middleware.log
    telemetry_path = get_base_path() / "logs" / LOG_FILENAME
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)  # create the folder if missing

    file_handler = logging.FileHandler(telemetry_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def to_json_format(obj) -> str:
    """Pretty-print an object (dict, dataclass, or Pydantic model) as a JSON string.

    Args:
        obj: The object to serialize. Can be a dict, list, dataclass,
            Pydantic model, or any combination thereof.

    Returns:
        str: An indented (2-space) JSON string representation of ``obj``.
    """
    def custom_encoder(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, BaseModel):
            return o.model_dump()
        return str(o)
    return json.dumps(obj, default=custom_encoder, indent=2)