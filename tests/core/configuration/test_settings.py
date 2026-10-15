"""Comprehensive unit test suite for core.configuration.settings.

Covers:
- Helper functions: _format_setting_value
- Data Models: LLMSettings and AppSettings (defaults, constraints, boundaries, immutability)
- Settings Singleton Lifecycle: __new__, configure, reload, thread-safe lazy loading
- Settings Loading: YAML parsing, fallback resolution, directory/missing file errors, malformed syntax
- Attribute Access & Snapshots: __getattr__, get_current, snapshot context manager (nesting, isolation, error reset)
- Persistence: atomic save, directory creation, rollback on failure
- Updating & State Management: _update (no-op, disk persistence, disk error tolerance), setters
- StateManager integration & Language change signaling: idle check, MiddlewareError handling, Signal emission
"""

import os
from pathlib import Path
import threading
from unittest.mock import MagicMock, patch
import pytest
import yaml
from pydantic import ValidationError

from core.configuration.settings import (
    AppSettings,
    LLMSettings,
    Settings,
    _format_setting_value,
    _snapshot,
)
from core.infrastructure.state_manager import StateManager
from core.tools.errors import MiddlewareError, MiddlewareErrorCode
from core.types.enums import Language, MiddlewareState


# --------------------------------------------------------------------------- #
# Fixtures & Test Isolation
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def preserve_settings_state():
    """Preserve Settings singleton state across tests to ensure complete test isolation.

    Restores _instance, _config_dir, _settings, and resets the ContextVar _snapshot
    so tests never interfere with each other or tests/conftest.py session checks.
    """
    orig_instance = Settings._instance
    orig_config_dir = Settings._config_dir
    orig_settings = Settings._settings

    orig_settings_copy = orig_settings.model_copy(deep=True) if orig_settings is not None else None
    token = _snapshot.set(None)

    yield

    _snapshot.reset(token)
    with Settings._lock:
        Settings._instance = orig_instance
        Settings._config_dir = orig_config_dir
        Settings._settings = orig_settings_copy


@pytest.fixture(autouse=True)
def reset_state_manager():
    """Ensure StateManager is in IDLE state before and after each test."""
    state_mgr = StateManager()
    state_mgr.transition_to(MiddlewareState.IDLE)
    yield
    state_mgr.transition_to(MiddlewareState.IDLE)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create an isolated temporary configuration directory with a valid settings.yaml."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    settings_file = cfg_dir / "settings.yaml"
    initial_content = {
        "llm": {
            "dialogue_generator_temperature": 0.7,
            "judger_temperature": 0.2,
            "healer_temperature": 0.4,
        },
        "language": "en",
        "profiling": False,
        "fairness_filter": True,
        "profanity_filter": True,
        "censor_word": "[CENSORED]",
        "refine_dialogue": True,
        "refiner_max_iterations": 3,
    }
    with open(settings_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(initial_content, f)
    return cfg_dir


@pytest.fixture
def signal_receiver():
    """Helper to connect and automatically disconnect a mock receiver from language_changed."""
    mock_receiver = MagicMock()
    Settings.language_changed.connect(mock_receiver)
    yield mock_receiver
    Settings.language_changed.disconnect(mock_receiver)


# --------------------------------------------------------------------------- #
# 1. Helper function: _format_setting_value
# --------------------------------------------------------------------------- #

class TestFormatSettingValue:
    """Tests for _format_setting_value representation formatting."""

    def test_format_enum_returns_name(self):
        """Enum values should be represented by their name."""
        assert _format_setting_value(Language.ENGLISH) == "ENGLISH"
        assert _format_setting_value(Language.ITALIAN) == "ITALIAN"
        assert _format_setting_value(MiddlewareState.IDLE) == "IDLE"

    def test_format_pydantic_model_returns_key_value_pairs(self):
        """Pydantic models are formatted as comma-separated key=value pairs."""
        llm = LLMSettings(
            dialogue_generator_temperature=0.8,
            judger_temperature=0.1,
            healer_temperature=0.3,
        )
        formatted = _format_setting_value(llm)
        expected = "dialogue_generator_temperature=0.8, judger_temperature=0.1, healer_temperature=0.3"
        assert formatted == expected

    def test_format_string_returns_repr(self):
        """String values should be formatted using repr() (quoted)."""
        assert _format_setting_value("[CENSORED]") == "'[CENSORED]'"
        assert _format_setting_value("") == "''"
        assert _format_setting_value("custom_word") == "'custom_word'"

    def test_format_primitive_types_returns_str(self):
        """Non-string primitives (bool, int, float) should be formatted using str()."""
        assert _format_setting_value(True) == "True"
        assert _format_setting_value(False) == "False"
        assert _format_setting_value(42) == "42"
        assert _format_setting_value(0.75) == "0.75"


# --------------------------------------------------------------------------- #
# 2. Model: LLMSettings
# --------------------------------------------------------------------------- #

class TestLLMSettingsModel:
    """Tests for LLMSettings model defaults, bounds validation, and immutability."""

    def test_default_values(self):
        """Verify default temperatures for LLMSettings."""
        settings = LLMSettings()
        assert settings.dialogue_generator_temperature == 0.7
        assert settings.judger_temperature == 0.2
        assert settings.healer_temperature == 0.4

    def test_valid_custom_values(self):
        """Verify initialization with custom valid temperature values."""
        settings = LLMSettings(
            dialogue_generator_temperature=1.2,
            judger_temperature=0.0,
            healer_temperature=1.9,
        )
        assert settings.dialogue_generator_temperature == 1.2
        assert settings.judger_temperature == 0.0
        assert settings.healer_temperature == 1.9

    @pytest.mark.parametrize("val", [0.0, 2.0])
    def test_temperature_boundary_values(self, val):
        """0.0 (minimum) and 2.0 (maximum) are valid boundaries for all temperature fields."""
        settings = LLMSettings(
            dialogue_generator_temperature=val,
            judger_temperature=val,
            healer_temperature=val,
        )
        assert settings.dialogue_generator_temperature == val
        assert settings.judger_temperature == val
        assert settings.healer_temperature == val

    @pytest.mark.parametrize("field_name", ["dialogue_generator_temperature", "judger_temperature", "healer_temperature"])
    @pytest.mark.parametrize("invalid_val", [-0.01, -1.0, 2.01, 3.0])
    def test_temperature_out_of_bounds_rejected(self, field_name, invalid_val):
        """Temperatures < 0.0 or > 2.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            LLMSettings(**{field_name: invalid_val})

    @pytest.mark.parametrize("field_name", ["dialogue_generator_temperature", "judger_temperature", "healer_temperature"])
    def test_non_numeric_temperature_rejected(self, field_name):
        """Non-numeric values must raise ValidationError."""
        with pytest.raises(ValidationError):
            LLMSettings(**{field_name: "invalid"})

    def test_frozen_immutability(self):
        """LLMSettings instance is frozen and cannot be mutated."""
        settings = LLMSettings()
        with pytest.raises(ValidationError):
            settings.dialogue_generator_temperature = 0.9


# --------------------------------------------------------------------------- #
# 3. Model: AppSettings
# --------------------------------------------------------------------------- #

class TestAppSettingsModel:
    """Tests for AppSettings model defaults, field types, and immutability."""

    def test_default_values(self):
        """Verify default values of all AppSettings fields."""
        settings = AppSettings()
        assert isinstance(settings.llm, LLMSettings)
        assert settings.llm.dialogue_generator_temperature == 0.7
        assert settings.language == Language.ENGLISH
        assert settings.profiling is False
        assert settings.fairness_filter is True
        assert settings.profanity_filter is True
        assert settings.censor_word == "[CENSORED]"
        assert settings.refine_dialogue is True
        assert settings.refiner_max_iterations == 3

    def test_custom_values(self):
        """Verify AppSettings initialization with custom values."""
        custom_llm = LLMSettings(dialogue_generator_temperature=1.0)
        settings = AppSettings(
            llm=custom_llm,
            language=Language.ITALIAN,
            profiling=True,
            fairness_filter=False,
            profanity_filter=False,
            censor_word="***BLOCKED***",
            refine_dialogue=False,
            refiner_max_iterations=5,
        )
        assert settings.llm == custom_llm
        assert settings.language == Language.ITALIAN
        assert settings.profiling is True
        assert settings.fairness_filter is False
        assert settings.profanity_filter is False
        assert settings.censor_word == "***BLOCKED***"
        assert settings.refine_dialogue is False
        assert settings.refiner_max_iterations == 5

    @pytest.mark.parametrize("lang_str, expected_enum", [("en", Language.ENGLISH), ("it", Language.ITALIAN)])
    def test_language_coercion_from_string(self, lang_str, expected_enum):
        """String language codes should be coerced to Language enum."""
        settings = AppSettings(language=lang_str)
        assert settings.language == expected_enum

    def test_invalid_language_raises_validation_error(self):
        """Invalid language code string must raise ValidationError."""
        with pytest.raises(ValidationError):
            AppSettings(language="fr")

    def test_frozen_immutability(self):
        """AppSettings is frozen and fields cannot be mutated in place."""
        settings = AppSettings()
        with pytest.raises(ValidationError):
            settings.fairness_filter = False


# --------------------------------------------------------------------------- #
# 4. Settings: Singleton Lifecycle & Configuration
# --------------------------------------------------------------------------- #

class TestSettingsLifecycle:
    """Tests for Settings initialization, configure(), and singleton pattern."""

    def test_unconfigured_instantiation_raises_runtime_error(self):
        """Accessing Settings() before calling Settings.configure() raises RuntimeError."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = None
            Settings._settings = None

        with pytest.raises(RuntimeError) as exc_info:
            Settings()

        assert "Settings.configure(config_dir) must be called" in str(exc_info.value)

    def test_configure_sets_path_from_str(self, tmp_path):
        """Settings.configure() accepts string and converts to Path."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = None

        Settings.configure(str(tmp_path))
        assert Settings._config_dir == Path(tmp_path)

    def test_configure_sets_path_from_path_object(self, tmp_path):
        """Settings.configure() accepts Path object."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = None

        Settings.configure(tmp_path)
        assert Settings._config_dir == tmp_path

    def test_configure_when_already_instantiated_raises_runtime_error(self, temp_config_dir):
        """Calling configure() after Settings instance has already been created raises RuntimeError."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        with pytest.raises(RuntimeError) as exc_info:
            Settings.configure(temp_config_dir)

        assert "Settings already instantiated: configure() must be called first" in str(exc_info.value)

    def test_singleton_identity(self, temp_config_dir):
        """Multiple calls to Settings() return the exact same instance."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir

        instance1 = Settings()
        instance2 = Settings()
        assert instance1 is instance2


# --------------------------------------------------------------------------- #
# 5. Settings: Loading & File Handling (_load)
# --------------------------------------------------------------------------- #

class TestSettingsLoad:
    """Tests for Settings._load, YAML deserialization, fallbacks, and error conditions."""

    def test_load_success_with_complete_yaml(self, temp_config_dir):
        """Successfully loads and deserializes settings from valid YAML file."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            settings = Settings()

        assert settings.language == Language.ENGLISH
        assert settings.fairness_filter is True
        assert settings.censor_word == "[CENSORED]"
        assert settings.refiner_max_iterations == 3
        assert settings.llm.dialogue_generator_temperature == 0.7

    def test_load_with_empty_yaml_uses_defaults(self, tmp_path):
        """An empty settings.yaml safely parses to an empty dict and uses all AppSettings defaults."""
        empty_dir = tmp_path / "empty_cfg"
        empty_dir.mkdir()
        (empty_dir / "settings.yaml").write_text("", encoding="utf-8")

        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = empty_dir
            settings = Settings()

        assert settings.language == Language.ENGLISH
        assert settings.fairness_filter is True
        assert settings.censor_word == "[CENSORED]"
        assert settings.llm.dialogue_generator_temperature == 0.7

    def test_load_with_partial_yaml_uses_defaults_for_missing(self, tmp_path):
        """YAML with only a few fields sets specified values and defaults for the rest."""
        partial_dir = tmp_path / "partial_cfg"
        partial_dir.mkdir()
        (partial_dir / "settings.yaml").write_text("language: it\ncensor_word: '[REDACTED]'\n", encoding="utf-8")

        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = partial_dir
            settings = Settings()

        assert settings.language == Language.ITALIAN
        assert settings.censor_word == "[REDACTED]"
        assert settings.fairness_filter is True
        assert settings.refiner_max_iterations == 3

    def test_load_resolves_via_example_fallback(self, tmp_path):
        """When settings.yaml is missing, falls back to settings.example.yaml via resolve_config_file."""
        example_dir = tmp_path / "fallback_cfg"
        example_dir.mkdir()
        example_file = example_dir / "settings.example.yaml"
        example_file.write_text("language: it\nfairness_filter: false\n", encoding="utf-8")

        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = example_dir
            settings = Settings()

        assert settings.language == Language.ITALIAN
        assert settings.fairness_filter is False

    def test_load_directory_instead_of_file_raises_is_a_directory_error(self, tmp_path):
        """When the resolved config path is a directory, raises IsADirectoryError."""
        bad_dir = tmp_path / "as_dir"
        bad_dir.mkdir()
        (bad_dir / "settings.yaml").mkdir()  # Make it a directory

        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = bad_dir

        with pytest.raises(IsADirectoryError) as exc_info:
            Settings()

        assert "got a directory" in str(exc_info.value)

    def test_load_file_not_found_raises_file_not_found_error(self, tmp_path):
        """When neither settings.yaml nor settings.example.yaml exists, raises FileNotFoundError."""
        empty_dir = tmp_path / "missing_dir"
        empty_dir.mkdir()

        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = empty_dir

        with pytest.raises(FileNotFoundError):
            Settings()

    def test_load_malformed_yaml_syntax_raises_yaml_error(self, tmp_path):
        """Malformed YAML syntax raises yaml.YAMLError."""
        corrupt_dir = tmp_path / "corrupt_cfg"
        corrupt_dir.mkdir()
        (corrupt_dir / "settings.yaml").write_text("invalid: yaml: : [bad", encoding="utf-8")

        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = corrupt_dir

        with pytest.raises(yaml.YAMLError):
            Settings()

    def test_load_invalid_schema_data_raises_validation_error(self, tmp_path):
        """YAML data that fails AppSettings validation raises ValidationError."""
        invalid_data_dir = tmp_path / "invalid_data_cfg"
        invalid_data_dir.mkdir()
        (invalid_data_dir / "settings.yaml").write_text("llm:\n  judger_temperature: 99.0\n", encoding="utf-8")

        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = invalid_data_dir

        with pytest.raises(ValidationError):
            Settings()


# --------------------------------------------------------------------------- #
# 6. Settings: Attribute Access & Snapshots
# --------------------------------------------------------------------------- #

class TestSettingsAttributeAccessAndSnapshot:
    """Tests for __getattr__, get_current(), and snapshot() context manager."""

    def test_getattr_proxies_to_settings(self, temp_config_dir):
        """__getattr__ proxies field access to the underlying AppSettings object."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            settings = Settings()

        assert settings.language == Language.ENGLISH
        assert settings.fairness_filter is True
        assert settings.profanity_filter is True
        assert settings.censor_word == "[CENSORED]"
        assert settings.refine_dialogue is True
        assert settings.refiner_max_iterations == 3
        assert settings.profiling is False
        assert isinstance(settings.llm, LLMSettings)

    def test_getattr_unknown_attribute_raises_attribute_error(self, temp_config_dir):
        """Accessing a non-existent attribute raises AttributeError."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            settings = Settings()

        with pytest.raises(AttributeError) as exc_info:
            _ = settings.unknown_field

        assert "'Settings' object has no attribute 'unknown_field'" in str(exc_info.value)

    def test_getattr_when_no_settings_loaded_raises_attribute_error(self):
        """Accessing attribute when _settings is None raises AttributeError."""
        instance = object.__new__(Settings)
        with patch.object(Settings, "_settings", None):
            with pytest.raises(AttributeError):
                _ = instance.some_attribute

    def test_get_current_returns_app_settings(self, temp_config_dir):
        """Settings.get_current() returns the current AppSettings instance."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            _ = Settings()

        current = Settings.get_current()
        assert isinstance(current, AppSettings)
        assert current.language == Language.ENGLISH

    def test_get_current_forces_loading_if_unloaded(self, temp_config_dir):
        """If _settings is None, get_current() triggers _ensure_loaded()."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings._settings = None

        current = Settings.get_current()
        assert current is not None
        assert current.language == Language.ENGLISH

    def test_snapshot_context_manager_isolates_reads(self, temp_config_dir):
        """Under snapshot(), reads return the snapshotted state even if settings change."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            settings = Settings()

        assert settings.censor_word == "[CENSORED]"

        with Settings.snapshot() as snap:
            assert snap.censor_word == "[CENSORED]"
            assert Settings.get_current().censor_word == "[CENSORED]"
            assert settings.censor_word == "[CENSORED]"

            # Update settings behind the scenes
            Settings.update_censor_word("[NEW_WORD]")

            # Outside snapshot is updated, but inside snapshot reads unchanged value
            assert Settings._settings.censor_word == "[NEW_WORD]"
            assert Settings.get_current().censor_word == "[CENSORED]"
            assert settings.censor_word == "[CENSORED]"

        # After exiting snapshot, reads see the updated value
        assert Settings.get_current().censor_word == "[NEW_WORD]"
        assert settings.censor_word == "[NEW_WORD]"

    def test_snapshot_with_explicit_snapshot_object(self, temp_config_dir):
        """Settings.snapshot(custom_snap) pins the context to the provided AppSettings."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            settings = Settings()

        custom_snap = AppSettings(censor_word="[EXPLICIT_SNAP]")

        with Settings.snapshot(custom_snap) as snap:
            assert snap is custom_snap
            assert Settings.get_current().censor_word == "[EXPLICIT_SNAP]"
            assert settings.censor_word == "[EXPLICIT_SNAP]"

        assert settings.censor_word == "[CENSORED]"

    def test_snapshot_resets_token_after_exception(self, temp_config_dir):
        """Snapshot ContextVar is reset even if an exception is raised in the block."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            _ = Settings()

        assert _snapshot.get() is None

        with pytest.raises(ValueError):
            with Settings.snapshot():
                assert _snapshot.get() is not None
                raise ValueError("Intentional error inside snapshot")

        assert _snapshot.get() is None

    def test_nested_snapshots(self, temp_config_dir):
        """Nested snapshots restore the outer snapshot upon exit of the inner block."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            settings = Settings()

        snap1 = AppSettings(censor_word="[SNAP_1]")
        snap2 = AppSettings(censor_word="[SNAP_2]")

        with Settings.snapshot(snap1):
            assert settings.censor_word == "[SNAP_1]"
            with Settings.snapshot(snap2):
                assert settings.censor_word == "[SNAP_2]"
            assert settings.censor_word == "[SNAP_1]"

        assert settings.censor_word == "[CENSORED]"


# --------------------------------------------------------------------------- #
# 7. Settings: Persistence (save & reload)
# --------------------------------------------------------------------------- #

class TestSettingsPersistence:
    """Tests for Settings.save() and Settings.reload()."""

    def test_save_persists_settings_to_configured_directory(self, temp_config_dir):
        """Settings.save() persists current in-memory settings to settings.yaml atomically."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        Settings._settings = Settings._settings.model_copy(update={"censor_word": "[SAVED_WORD]"})
        Settings.save()

        saved_path = temp_config_dir / "settings.yaml"
        with open(saved_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["censor_word"] == "[SAVED_WORD]"

    def test_save_to_custom_directory(self, temp_config_dir, tmp_path):
        """Settings.save(custom_dir) persists to the specified custom directory."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        custom_target = tmp_path / "custom_dir"
        Settings.save(custom_target)

        custom_file = custom_target / "settings.yaml"
        assert custom_file.exists()
        with open(custom_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["language"] == "en"

    def test_save_atomic_replace_cleans_up_on_failure(self, temp_config_dir):
        """If os.replace fails during save, temporary file is unlinked and exception is re-raised."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        with patch("os.replace", side_effect=OSError("Disk write error")):
            with pytest.raises(OSError):
                Settings.save()

        # Verify no stray temporary .tmp files left behind
        tmp_files = list(temp_config_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_save_atomic_replace_cleans_up_on_dump_failure(self, temp_config_dir):
        """If yaml.safe_dump fails during save, temporary file is cleaned up."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        with patch("yaml.safe_dump", side_effect=TypeError("Serialization error")):
            with pytest.raises(TypeError):
                Settings.save()

        tmp_files = list(temp_config_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_save_forces_loading_if_settings_is_none(self, temp_config_dir):
        """Calling save() when _settings is None forces loading via cls()."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings._settings = None

        Settings.save()
        assert Settings._settings is not None

    def test_reload_without_configure_raises_runtime_error(self):
        """Calling reload() when _config_dir is None raises RuntimeError."""
        with Settings._lock:
            Settings._config_dir = None

        with pytest.raises(RuntimeError) as exc_info:
            Settings.reload()

        assert "Settings.configure(config_dir) was never called" in str(exc_info.value)

    def test_reload_reloads_updated_content_from_disk(self, temp_config_dir):
        """reload() clears instance and settings, re-reading new values from settings.yaml."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            settings = Settings()

        assert settings.refiner_max_iterations == 3

        # Directly update the file on disk without calling Settings API
        settings_file = temp_config_dir / "settings.yaml"
        with open(settings_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["refiner_max_iterations"] = 10
        with open(settings_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        reloaded = Settings.reload()
        assert reloaded.refiner_max_iterations == 10
        assert Settings.get_current().refiner_max_iterations == 10


# --------------------------------------------------------------------------- #
# 8. Settings: Updating Logic (_update)
# --------------------------------------------------------------------------- #

class TestSettingsUpdate:
    """Tests for Settings._update, change detection, and disk error tolerance."""

    def test_update_returns_true_when_value_changes(self, temp_config_dir):
        """_update returns True and persists change when value differs from old."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        assert Settings.get_current().profiling is False
        changed = Settings._update("profiling", True)
        assert changed is True
        assert Settings.get_current().profiling is True

        # Check disk was persisted
        with open(temp_config_dir / "settings.yaml", "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["profiling"] is True

    def test_update_returns_false_and_no_save_when_value_unchanged(self, temp_config_dir):
        """_update returns False and avoids calling save() when value is identical to old."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        with patch.object(Settings, "save") as mock_save:
            changed = Settings._update("profiling", False)
            assert changed is False
            mock_save.assert_not_called()

    def test_update_save_failure_keeps_in_memory_and_does_not_raise(self, temp_config_dir, caplog):
        """When disk save raises an exception, in-memory update is preserved and error is logged."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        with patch.object(Settings, "save", side_effect=OSError("Disk full")):
            changed = Settings._update("profiling", True)
            assert changed is True
            assert Settings.get_current().profiling is True

        assert "Failed to persist settings to disk" in caplog.text

    def test_update_concurrent_thread_safety(self, temp_config_dir):
        """Multiple threads calling _update do not deadlock and synchronize through _lock."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        errors = []

        def worker(iteration):
            try:
                Settings._update("refiner_max_iterations", iteration)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert 1 <= Settings.get_current().refiner_max_iterations <= 9


# --------------------------------------------------------------------------- #
# 9. Settings: Public Setter Methods
# --------------------------------------------------------------------------- #

class TestSettingsSetters:
    """Tests for public helper setters modifying specific configuration domains."""

    def test_toggle_fairness_filter(self, temp_config_dir):
        """toggle_fairness_filter updates fairness_filter setting."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        assert Settings.get_current().fairness_filter is True
        Settings.toggle_fairness_filter(False)
        assert Settings.get_current().fairness_filter is False
        Settings.toggle_fairness_filter(True)
        assert Settings.get_current().fairness_filter is True

    def test_update_llm_settings(self, temp_config_dir):
        """update_llm_settings updates the entire LLMSettings object."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        new_llm = LLMSettings(
            dialogue_generator_temperature=1.5,
            judger_temperature=0.8,
            healer_temperature=1.1,
        )
        Settings.update_llm_settings(new_llm)
        assert Settings.get_current().llm == new_llm

    def test_update_profanity_filter_settings(self, temp_config_dir):
        """update_profanity_filter_settings updates profanity_filter setting."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        assert Settings.get_current().profanity_filter is True
        Settings.update_profanity_filter_settings(False)
        assert Settings.get_current().profanity_filter is False

    def test_update_censor_word(self, temp_config_dir):
        """update_censor_word updates censor_word setting."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        Settings.update_censor_word("[MUTED]")
        assert Settings.get_current().censor_word == "[MUTED]"

    def test_update_refiner_max_iterations(self, temp_config_dir):
        """update_refiner_max_iterations updates refiner_max_iterations setting."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        Settings.update_refiner_max_iterations(7)
        assert Settings.get_current().refiner_max_iterations == 7


# --------------------------------------------------------------------------- #
# 10. Settings: change_language & StateManager Integration
# --------------------------------------------------------------------------- #

class TestSettingsChangeLanguage:
    """Tests for change_language, StateManager idle requirement, and signal emission."""

    def test_change_language_idle_success_fires_signal(self, temp_config_dir, signal_receiver):
        """When middleware is IDLE, change_language updates setting and fires language_changed signal."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        state_mgr = StateManager()
        state_mgr.transition_to(MiddlewareState.IDLE)

        assert Settings.get_current().language == Language.ENGLISH
        Settings.change_language(Language.ITALIAN)

        assert Settings.get_current().language == Language.ITALIAN
        signal_receiver.assert_called_once_with(Settings, language=Language.ITALIAN)

    def test_change_language_same_language_does_not_fire_signal(self, temp_config_dir, signal_receiver):
        """When the requested language is already set, no signal is emitted and setting remains unchanged."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        state_mgr = StateManager()
        state_mgr.transition_to(MiddlewareState.IDLE)

        assert Settings.get_current().language == Language.ENGLISH
        Settings.change_language(Language.ENGLISH)

        assert Settings.get_current().language == Language.ENGLISH
        signal_receiver.assert_not_called()

    @pytest.mark.parametrize(
        "non_idle_state",
        [
            MiddlewareState.GENERATING,
            MiddlewareState.STARTING,
            MiddlewareState.SETTING_CONTEXT,
        ],
    )
    def test_change_language_non_idle_raises_middleware_error(
        self, temp_config_dir, signal_receiver, non_idle_state, caplog
    ):
        """When middleware is not in IDLE state, change_language raises MiddlewareError and logs a warning."""
        with Settings._lock:
            Settings._instance = None
            Settings._config_dir = temp_config_dir
            Settings()

        state_mgr = StateManager()
        state_mgr.transition_to(non_idle_state)

        with pytest.raises(MiddlewareError) as exc_info:
            Settings.change_language(Language.ITALIAN)

        assert exc_info.value.code == MiddlewareErrorCode.GENERATING
        assert non_idle_state.value in str(exc_info.value)
        assert "Language cannot be changed while the middleware is" in caplog.text

        # Setting must remain unchanged and signal must not be fired
        assert Settings.get_current().language == Language.ENGLISH
        signal_receiver.assert_not_called()
