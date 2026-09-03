import logging
from pydantic import BaseModel, Field
from pathlib import Path
from threading import Lock
import yaml

from core.types.enums import Language

logger = logging.getLogger(__name__)


class LLMSettings(BaseModel):
    """Configuration for the language model used to generate dialogue."""

    default_temperature: float = Field(0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(1000, gt=0)


class AppSettings(BaseModel):
    """User-configurable application settings, loaded from settings.yaml."""

    language: Language = Language.ENGLISH
    llm: LLMSettings = Field(default_factory=LLMSettings)
    profanity_filter: bool = True
    prompt_fairness_filter: bool = True
    number_of_options: int = 2


class Settings:
    """Singleton that loads the application configuration from a YAML file
    and exposes its attributes directly, e.g. Settings().profanity_filter

    Settings.configure(config_dir) must be called once, at application
    startup (main.py), before any other access to Settings().
    """

    SETTINGS_FILENAME = "settings.yaml"

    _instance: "Settings | None" = None
    _lock: Lock = Lock()
    _settings: AppSettings | None = None
    _config_dir: Path | None = None

    def __new__(cls):
        """Return the existing singleton instance, or create and load it
        on first access.

        Raises:
            RuntimeError: If configure() has not been called yet.
        """
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
        """Load configuration from disk and validate it against AppSettings.

        Args:
            path: Full path to the settings.yaml file to load.

        Raises:
            IsADirectoryError: If path points to a directory instead of a file.
            FileNotFoundError: If the settings file does not exist.
        """
        if path.is_dir():
            raise IsADirectoryError(
                f"Expected the file {self.SETTINGS_FILENAME}, got a directory: {path}"
            )
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        settings = AppSettings(**raw)

        type(self)._settings = settings

    def __getattr__(self, item):
        """Delegate attribute access to the loaded AppSettings instance.

        Called only if the attribute is not found on the Settings instance
        itself, so it forwards lookups like .language or .profanity_filter
        to the underlying validated config.

        Raises:
            AttributeError: If the loaded settings have no such attribute.
        """
        settings = type(self)._settings
        if settings is not None and hasattr(settings, item):
            return getattr(settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @classmethod
    def configure(cls, config_dir: str | Path) -> None:
        """Set the config directory. Must be called once, before any Settings().

        Args:
            config_dir: Path to the folder containing settings.yaml.

        Raises:
            RuntimeError: If Settings has already been instantiated.
        """
        with cls._lock:
            if cls._instance is not None:
                raise RuntimeError("Settings already instantiated: configure() must be called first")
            cls._config_dir = Path(config_dir)

    @classmethod
    def reload(cls) -> "Settings":
        """Force a reload from disk, using the already configured directory.

        Returns:
            Settings: The freshly reloaded singleton instance.

        Raises:
            RuntimeError: If configure() was never called.
        """
        if cls._config_dir is None:
            raise RuntimeError("Settings.configure(config_dir) was never called")
        with cls._lock:
            cls._settings = None
            cls._instance = None
        return cls()

    @classmethod
    def save(cls, config_dir: str | Path | None = None) -> None:
        """Persist the current settings back to a YAML file on disk.

        Args:
            config_dir: Optional target directory. Defaults to the
                directory configured via configure().
        """
        if cls._settings is None:
            cls()  # force loading with defaults first
        target_dir = Path(config_dir) if config_dir else cls._config_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / cls.SETTINGS_FILENAME
        # mode="json" ensures enums, Path, etc. are serialized as plain values
        data = cls._settings.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    @classmethod
    def change_language(cls, language: Language) -> None:
        """Update the active language, loading defaults first if needed.

        Args:
            language: The new language to set.
        """
        if cls._settings is None:
            cls()  # force loading with defaults
        cls._settings.language = language
        logger.info(f"Language changed to {language.index}")

    @classmethod
    def toggle_profanity_filter(cls, flag: bool) -> None:
        """Enable or disable the profanity filter, loading defaults first if needed.

        Args:
            flag: True to enable the profanity filter, False to disable it.
        """
        if cls._settings is None:
            cls()  # force loading with defaults
        cls._settings.profanity_filter = flag
        logger.info(f"Profanity filter {'ON' if flag else 'OFF'}")