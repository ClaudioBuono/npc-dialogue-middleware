import hashlib
from pathlib import Path
import shutil
import pytest
from core.configuration.settings import Settings
import core.helpers.paths

def _hash(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

@pytest.fixture(scope="session", autouse=True)
def setup_global_settings(tmp_path_factory):
    project_root = Path(__file__).resolve().parent.parent
    
    # Settings() mandatory configuration
    src_config = project_root / "config"
    if not src_config.exists():
        src_config = project_root / "src" / "config"

    config_dir = tmp_path_factory.mktemp("config")
    shutil.copytree(src_config, config_dir, dirs_exist_ok=True)
    
    Settings.configure(config_dir)
    Settings()

    # Check if real settings file has been modified
    real_file = src_config / Settings().SETTINGS_FILENAME
    before = _hash(real_file)

    # Find the asset folder 
    assets_dir = project_root / "assets"
    if not assets_dir.exists():
        assets_dir = project_root / "src" / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(
            f"Assets folder not found. Tried: {project_root / 'assets'} and {project_root / 'src' / 'assets'}"
        )

	# Replace wth the found path
    def mock_resource_path(relative_path: str) -> Path:
        return assets_dir / relative_path

    core.helpers.paths.resource_path = mock_resource_path

    yield

    assert _hash(real_file) == before, "Tests have modified real settings file"