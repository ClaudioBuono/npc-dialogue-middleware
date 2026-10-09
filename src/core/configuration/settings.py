import logging
import os
import tempfile
import yaml
from enum import Enum
from pathlib import Path
from threading import RLock
from blinker import Signal
from pydantic import BaseModel, ConfigDict, Field
from core.helpers.paths import resolve_config_file
from core.types.enums import Language
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)

_snapshot: ContextVar["AppSettings | None"] = ContextVar("settings_snapshot", default=None)

def _format_setting_value(value) -> str:
    """Compact, readable representation of a setting value for log messages."""
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, BaseModel):
        return ", ".join(f"{k}={v}" for k, v in value.model_dump().items())
    return repr(value) if isinstance(value, str) else str(value)


class LLMSettings(BaseModel):
    """Configuration for the language model used to generate dialogue."""
    model_config = ConfigDict(frozen=True)
    dialogue_generator_temperature: float = Field(0.7, ge=0.0, le=2.0)
    judger_temperature: float = Field(0.2, ge=0.0, le=2.0)
    healer_temperature: float = Field(0.4, ge=0.0, le=2.0)


class AppSettings(BaseModel):
    """User-configurable application settings, loaded from settings.yaml."""
    model_config = ConfigDict(frozen=True)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    language: Language = Language.ENGLISH
    profiling: bool = False
    fairness_filter: bool = True # Fairness Filter in prompt + Judger Questions
    profanity_filter: bool = True  # Stream: censors with censor_word. Generate: refiner questions if refine_dialogue is on, else censors
    censor_word: str = "[CENSORED]"
    refine_dialogue: bool = True # Enables the refiner in Generate Mode
    refiner_max_iterations: int = 3

class Settings:
    """Singleton that loads the application configuration from a YAML file
    and exposes its attributes directly

    Settings.configure(config_dir) must be called once, at application
    startup (main.py), before any other access to Settings().

    Every setter persists the change to disk automatically.
    """

    SETTINGS_FILENAME = "settings.yaml"

    _instance: "Settings | None" = None
    _lock: RLock = RLock()
    _settings: AppSettings | None = None
    _config_dir: Path | None = None

    # Events
    language_changed = Signal("language-changed")

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if cls._config_dir is None:
                        raise RuntimeError(
                            "Settings.configure(config_dir) must be called before accessing Settings()"
                        )
                    instance = super().__new__(cls)
                    instance._load(cls._config_dir / cls.SETTINGS_FILENAME)
                    cls._instance = instance
        return cls._instance

    def _load(self, path: Path) -> None:
        path = resolve_config_file(path.parent, path.name)
        if path.is_dir():
            raise IsADirectoryError(
                f"Expected the file {self.SETTINGS_FILENAME}, got a directory: {path}"
            )
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        settings = AppSettings(**raw)
        logger.debug("Settings loaded from %s:\n%s", path, settings.model_dump_json(indent=2))
        type(self)._settings = settings

    def __getattr__(self, item):
        settings = _snapshot.get()
        if settings is None:
            settings = type(self)._settings
        if settings is not None and hasattr(settings, item):
            return getattr(settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @classmethod
    def get_current(cls) -> AppSettings:
        snap = _snapshot.get()
        if snap is not None:
            return snap
        cls._ensure_loaded()
        return cls._settings

    @classmethod
    @contextmanager
    def snapshot(cls, snap: "AppSettings | None" = None):
        """Pin the settings seen by this context to a fixed snapshot.

        Without arguments, snapshots the current settings. Pass an existing
        snapshot to re-enter it (e.g. inside a streaming generator).
        """
        if snap is None:
            cls._ensure_loaded()
            snap = cls._settings
        token = _snapshot.set(snap)
        try:
            yield snap
        finally:
            _snapshot.reset(token)

    @classmethod
    def configure(cls, config_dir: str | Path) -> None:
        with cls._lock:
            if cls._instance is not None:
                raise RuntimeError("Settings already instantiated: configure() must be called first")
            cls._config_dir = Path(config_dir)

    @classmethod
    def reload(cls) -> "Settings":
        if cls._config_dir is None:
            raise RuntimeError("Settings.configure(config_dir) was never called")
        logger.info("Reloading settings from disk")
        with cls._lock:
            cls._settings = None
            cls._instance = None
        return cls()

    @classmethod
    def save(cls, config_dir: str | Path | None = None) -> None:
        """Persist the current settings back to a YAML file on disk.

        The write is atomic: data goes to a temp file first, then replaces
        the target, so a crash mid-write cannot corrupt settings.yaml.

        Args:
            config_dir: Optional target directory. Defaults to the
                directory configured via configure().
        """
        with cls._lock:
            if cls._settings is None:
                cls()
            target_dir = Path(config_dir) if config_dir else cls._config_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            path = target_dir / cls.SETTINGS_FILENAME
            data = cls._settings.model_dump(mode="json")

            fd, tmp_name = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                os.replace(tmp_name, path)  # atomico
            except Exception:
                Path(tmp_name).unlink(missing_ok=True)
                raise
            logger.debug("Settings saved to %s", path)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._settings is None:
            cls()  # force loading

    @classmethod
    def _update(cls, name: str, value) -> bool:
        """Set a top-level setting, persist it and log old -> new.

        The in-memory change is applied first; if writing to disk fails the
        error is logged and the app keeps working with the new value.

        Args:
            name: Name of the AppSettings field to update.
            value: The new value.

        Returns:
            True if the value changed, False if it was already set.
        """
        with cls._lock:
            cls._ensure_loaded()
            old = getattr(cls._settings, name)
            if old == value:
                logger.info("Setting '%s' unchanged: %s", name, _format_setting_value(value))
                return False
            cls._settings = cls._settings.model_copy(update={name: value})
            try:
                cls.save()
            except Exception:
                logger.exception("Failed to persist settings to disk: the change is applied in memory only")

            logger.info("Setting '%s' changed: %s -> %s", name, _format_setting_value(old), _format_setting_value(value))
            return True

    @classmethod
    def change_language(cls, language: Language) -> None:
        """Update the active language and persist it. Notifies listeners only if it changed."""
        if cls._update("language", language):
            cls.language_changed.send(cls, language=language)

    @classmethod
    def toggle_fairness_filter(cls, flag: bool) -> None:
        """Enable or disable the prompt fairness filter and persist it."""
        cls._update("fairness_filter", flag)

    @classmethod
    def update_llm_settings(cls, llm_settings: LLMSettings) -> None:
        """Replace the LLM settings (temperature) and persist them."""
        cls._update("llm", llm_settings)

    @classmethod
    def update_profanity_filter_settings(cls, profanity_filter: bool) -> None:
        """Replace the profanity filter and persist it."""
        cls._update("profanity_filter", profanity_filter)

    @classmethod
    def update_censor_word(cls, censor_word: str) -> None:
        """Update the censor word and persist it."""
        cls._update("censor_word", censor_word)

    @classmethod
    def update_refiner_max_iterations(cls, max_iterations: int) -> None:
        """Update the refiner max iterations and persist it."""
        cls._update("refiner_max_iterations", max_iterations)
