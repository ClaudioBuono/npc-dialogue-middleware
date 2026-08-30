import sys
from pathlib import Path


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
        base = Path(__file__).parent.parent / "tools"
    return base / relative_path