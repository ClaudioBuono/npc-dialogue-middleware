import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def get_base_path() -> Path:
    """Return the folder containing the running entry point"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(sys.modules["__main__"].__file__).resolve().parent

def resource_path(relative_path: str) -> Path:
    """Resolve bundled resource path"""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(sys.modules["__main__"].__file__).resolve().parent / "assets"
    return base / relative_path

def resolve_config_file(config_dir: Path, filename: str) -> Path:
    """Return the real config file, or fall back to its .example twin."""
    real = config_dir / filename
    if real.exists():
        return real

    stem, _, ext = filename.rpartition(".")
    example = config_dir / f"{stem}.example.{ext}"
    if example.exists():
        logger.warning(
            f"{filename} not found in {config_dir}: using {example.name}. "
            f"Copy it to {filename} to customize your configuration."
        )
        return example

    raise FileNotFoundError(f"Neither {real} nor {example} found")