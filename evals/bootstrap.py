from __future__ import annotations
import sys
from pathlib import Path

_EVALS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVALS_DIR.parent

# `core` (and `api`) live under src/ (src-layout). They're only importable
# without extra setup when pytest is configured to add src/ to sys.path
# (e.g. pyproject.toml's [tool.pytest.ini_options] pythonpath, or an
# editable install used only in that context). Running this module
# directly with `python -m evals...` doesn't get that for free, so fall
# back to inserting src/ by hand if it exists and isn't already there.
_SRC_DIR = _PROJECT_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import core.helpers.paths
from core.config.settings import Settings
from core.routing.registry import ModelRegistry


def configure_environment() -> None:
    """Configure Settings, patch resource_path, and populate ModelRegistry.

    Mirrors tests/conftest.py's setup_global_settings fixture, plus the
    ModelRegistry().set_models() call that the main unit test suite never
    needs (Judger/Healer are always exercised there with a mocked client,
    so LLMRouter.select_model — and therefore ModelRegistry — is never
    actually invoked). Idempotent: safe to call more than once, since
    Settings and ModelRegistry are singletons and set_models() simply
    re-registers the same configured models.
    """
    config_dir = _PROJECT_ROOT / "config"
    if not config_dir.exists():
        config_dir = _PROJECT_ROOT / "src" / "config"
    Settings.configure(config_dir)
    Settings()

    assets_dir = _PROJECT_ROOT / "assets"
    if not assets_dir.exists():
        assets_dir = _PROJECT_ROOT / "src" / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(
            f"Cartella assets non trovata. Tentati: {_PROJECT_ROOT / 'assets'} e {_PROJECT_ROOT / 'src' / 'assets'}"
        )

    def mock_resource_path(relative_path: str) -> Path:
        return assets_dir / relative_path

    core.helpers.paths.resource_path = mock_resource_path

    # See ModelRegistry.set_models(): with no args it uses whatever is
    # already loaded from modelconfigs.json, profiler=True by default —
    # the same thing calling /api/set_models with an empty body would do.
    ModelRegistry().set_models()