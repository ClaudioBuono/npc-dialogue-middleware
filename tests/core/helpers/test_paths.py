"""Unit tests for core.helpers.paths – base path, resource resolution, and config fallback."""
import importlib
import logging
import sys
import types
from pathlib import Path
import pytest

import core.helpers.paths
from core.helpers.paths import (
    get_base_path,
    resolve_config_file,
)


@pytest.fixture
def original_resource_path():
    """Fixture that retrieves the original unpatched resource_path function.

    tests/conftest.py monkeypatches core.helpers.paths.resource_path at session
    startup. This fixture reloads the module to access the authentic implementation,
    and restores conftest's mock at teardown so other test suites are unaffected.
    """
    conftest_mock = core.helpers.paths.resource_path
    importlib.reload(core.helpers.paths)
    real_func = core.helpers.paths.resource_path
    yield real_func
    core.helpers.paths.resource_path = conftest_mock


class TestGetBasePath:
    """Tests for get_base_path function in both frozen and unfrozen execution modes."""

    def test_frozen_mode_returns_executable_parent(self, monkeypatch, tmp_path):
        """When running in a PyInstaller bundle (sys.frozen=True), returns sys.executable parent."""
        fake_exe = tmp_path / "bin" / "npc_middleware.exe"
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(fake_exe))

        base_path = get_base_path()
        assert base_path == fake_exe.parent

    def test_unfrozen_mode_returns_main_module_file_parent(self, monkeypatch, tmp_path):
        """When running in standard Python, returns parent folder of __main__.__file__."""
        main_script = tmp_path / "project" / "src" / "main.py"
        fake_main = types.ModuleType("__main__")
        fake_main.__file__ = str(main_script)

        monkeypatch.setattr(sys, "frozen", False, raising=False)
        monkeypatch.setitem(sys.modules, "__main__", fake_main)

        base_path = get_base_path()
        assert base_path == main_script.resolve().parent


class TestResourcePath:
    """Tests for resource_path function in both frozen and unfrozen execution modes."""

    def test_frozen_mode_uses_meipass(self, monkeypatch, tmp_path, original_resource_path):
        """In frozen mode, resolves relative to sys._MEIPASS."""
        mei_dir = tmp_path / "_MEI123456"
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(mei_dir), raising=False)

        resolved = original_resource_path("prompts/judge.yaml")
        assert resolved == mei_dir / "prompts" / "judge.yaml"

    def test_unfrozen_mode_uses_assets_directory(self, monkeypatch, tmp_path, original_resource_path):
        """In unfrozen mode, resolves relative to <main_parent>/assets."""
        main_script = tmp_path / "app" / "main.py"
        fake_main = types.ModuleType("__main__")
        fake_main.__file__ = str(main_script)

        monkeypatch.setattr(sys, "frozen", False, raising=False)
        monkeypatch.setitem(sys.modules, "__main__", fake_main)

        resolved = original_resource_path("lexicons/blacklist.txt")
        expected = main_script.resolve().parent / "assets" / "lexicons" / "blacklist.txt"
        assert resolved == expected

    def test_nested_and_subpath_resolution(self, monkeypatch, tmp_path, original_resource_path):
        """Nested relative paths and simple file names are properly joined."""
        mei_dir = tmp_path / "bundle"
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(mei_dir), raising=False)

        assert original_resource_path("single.json") == mei_dir / "single.json"
        assert original_resource_path("deep/nested/dir/asset.png") == mei_dir / "deep" / "nested" / "dir" / "asset.png"


class TestResolveConfigFile:
    """Tests for resolve_config_file and fallback to .example configuration."""

    def test_real_file_exists(self, tmp_path, caplog):
        """When the real config file exists, it is returned without warnings."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("setting: 1", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            result = resolve_config_file(tmp_path, "settings.yaml")

        assert result == config_file
        assert "not found in" not in caplog.text

    def test_fallback_to_example_file(self, tmp_path, caplog):
        """When real file is missing but .example twin exists, returns example and logs warning."""
        example_file = tmp_path / "settings.example.yaml"
        example_file.write_text("setting: default", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            result = resolve_config_file(tmp_path, "settings.yaml")

        assert result == example_file
        assert "settings.yaml not found in" in caplog.text
        assert "using settings.example.yaml" in caplog.text
        assert "Copy it to settings.yaml to customize" in caplog.text

    def test_real_file_preferred_when_both_exist(self, tmp_path, caplog):
        """When both real file and .example file exist, real file takes priority."""
        real_file = tmp_path / "routing.json"
        real_file.write_text("{}", encoding="utf-8")
        example_file = tmp_path / "routing.example.json"
        example_file.write_text("{}", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            result = resolve_config_file(tmp_path, "routing.json")

        assert result == real_file
        assert "not found in" not in caplog.text

    def test_neither_file_exists_raises_file_not_found(self, tmp_path):
        """When neither real file nor example file exists, raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_config_file(tmp_path, "database.yaml")

        assert "Neither" in str(exc_info.value)
        assert "database.yaml" in str(exc_info.value)
        assert "database.example.yaml" in str(exc_info.value)

    def test_filename_with_multiple_dots(self, tmp_path, caplog):
        """Filename with multiple dots splits on the rightmost dot for extension."""
        example_file = tmp_path / "models.config.example.json"
        example_file.write_text("{}", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            result = resolve_config_file(tmp_path, "models.config.json")

        assert result == example_file
        assert "models.config.json not found in" in caplog.text
        assert "using models.config.example.json" in caplog.text

    def test_filename_without_extension(self, tmp_path):
        """Filename without an extension uses .example.<filename> pattern and raises if missing."""
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_config_file(tmp_path, "settings")

        assert "Neither" in str(exc_info.value)
        assert ".example.settings" in str(exc_info.value)

    def test_non_existent_config_dir(self, tmp_path):
        """Pointing to a non-existent directory raises FileNotFoundError."""
        non_existent_dir = tmp_path / "missing_dir"
        with pytest.raises(FileNotFoundError):
            resolve_config_file(non_existent_dir, "app.cfg")
