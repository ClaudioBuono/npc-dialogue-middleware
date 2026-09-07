from pathlib import Path
import pytest
from core.config.settings import Settings
import core.helpers.paths

@pytest.fixture(scope="session", autouse=True)
def setup_global_settings():
    project_root = Path(__file__).resolve().parent.parent
    
    # Settings() mandatory configuration
    config_dir = project_root / "config"
    if not config_dir.exists():
        config_dir = project_root / "src" / "config"
    
    Settings.configure(config_dir)
    Settings()

    # Find the asset folder 
    assets_dir = project_root / "assets"
    if not assets_dir.exists():
        assets_dir = project_root / "src" / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(
            f"Cartella assets non trovata. Tentati: {project_root / 'assets'} e {project_root / 'src' / 'assets'}"
        )

	# Replace wth the found path
    def mock_resource_path(relative_path: str) -> Path:
        return assets_dir / relative_path

    core.helpers.paths.resource_path = mock_resource_path