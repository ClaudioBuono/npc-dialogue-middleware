import logging
import os
import tempfile
import yaml
from pathlib import Path
from threading import RLock
from blinker import Signal
from pydantic import BaseModel, Field
from core.helpers.paths import resolve_config_file
from core.types.enums import Language

logger = logging.getLogger(__name__)


class LLMSettings(BaseModel):
    """Configuration for the language model used to generate dialogue."""

    dialogue_generator_temperature: float = Field(0.7, ge=0.0, le=2.0)
    judger_temperature: float = Field(0.2, ge=0.0, le=2.0)
    healer_temperature: float = Field(0.4, ge=0.0, le=2.0)


class AppSettings(BaseModel):
    """User-configurable application settings, loaded from settings.yaml."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    language: Language = Language.ENGLISH
    profiling: bool = False
    fairness_filter: bool = True # Fairness Filter in prompt + Judger Questions
    profanity_filter: bool = True  # Stream: censors with censor_word. Generate: refiner questions if refine_dialogue is on, else censors
    censor_word: str = "[CENSORED]"
    refine_dialogue: bool = True # Enables the refiner in Generate Mode
    refiner_max_iterations: int = 3
    number_of_options: int = 2

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
        logger.debug("Settings loaded from %s: %s", path, settings)
        type(self)._settings = settings

    def __getattr__(self, item):
        settings = type(self)._settings
        if settings is not None and hasattr(settings, item):
            return getattr(settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

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

    @classmethod
    def _persist(cls) -> None:
        """Save to disk without letting an I/O error crash the caller.

        The in-memory change is already applied; if writing fails we log
        the error so the app keeps working with the new value.
        """
        try:
            cls.save()
        except Exception:
            logger.exception("Failed to persist settings to disk")

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._settings is None:
            cls()  # force loading

    @classmethod
    def change_language(cls, language: Language) -> None:
        """Update the active language and persist it."""
        with cls._lock:
            cls._ensure_loaded()
            cls._settings.language = language
            cls._persist()
        logger.info(f"Language changed to {language.index}")
        cls.language_changed.send(cls, language=language)

    @classmethod
    def toggle_fairness_filter(cls, flag: bool) -> None:
        """Enable or disable the prompt fairness filter and persist it."""
        with cls._lock:
            cls._ensure_loaded()
            cls._settings.fairness_filter = flag
            cls._persist()
        logger.info(f"Prompt fairness filter {'ON' if flag else 'OFF'}")

    @classmethod
    def set_number_of_options(cls, value: int) -> None:
        """Set the number of dialogue options per turn and persist it."""
        with cls._lock:
            cls._ensure_loaded()
            cls._settings.number_of_options = value
            cls._persist()
        logger.info(f"Number of options set to {value}")

    @classmethod
    def update_llm_settings(cls, llm_settings: LLMSettings) -> None:
        """Replace the LLM settings (temperature) and persist them."""
        with cls._lock:
            cls._ensure_loaded()
            cls._settings.llm = llm_settings
            cls._persist()
        logger.info(f"LLM settings updated: {llm_settings}")

    @classmethod
    def update_profanity_filter_settings(cls, profanity_filter: bool) -> None:
        """Replace the profanity filter and persist it."""
        with cls._lock:
            cls._ensure_loaded()
            cls._settings.profanity_filter = profanity_filter
            cls._persist()
        logger.info(f"Profanity filter setting updated: {profanity_filter}")

    @classmethod
    def update_censor_word(cls, censor_word: str) -> None:
        """Update the censor word and persist it."""
        with cls._lock:
            cls._ensure_loaded()
            cls._settings.censor_word = censor_word
            cls._persist()
        logger.info(f"Censor word updated: {censor_word}")

    @classmethod
    def update_refiner_max_iterations(cls, max_iterations: int) -> None:
        """Update the refiner max iterations and persist it."""
        with cls._lock:
            cls._ensure_loaded()
            cls._settings.refiner_max_iterations = max_iterations
            cls._persist()
        logger.info(f"Refiner max iterations updated: {max_iterations}")

    @classmethod
    def get_current(cls) -> AppSettings:
        cls._ensure_loaded()
        return cls._settings