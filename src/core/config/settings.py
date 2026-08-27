from pydantic import BaseModel, Field
from pathlib import Path
from threading import Lock
import yaml

from core.types.enums import Language


class LLMSettings(BaseModel):
    default_temperature: float = Field(0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(1000, gt=0)


class AppSettings(BaseModel):
    language: Language = Language.ENGLISH
    llm: LLMSettings = Field(default_factory=LLMSettings)
    telemetry_path: str = "logs/npc_middleware.log"
    profanity_filter: bool = True

    # add other config sections here as needed


class Settings:
    """
    Singleton that loads the application configuration from a YAML file
    and exposes its attributes directly, e.g. Settings().telemetry_path
    """
    _instance: "Settings | None" = None
    _lock: Lock = Lock()
    _settings: AppSettings | None = None
    _config_path: str | Path = "src/core/config/settings.yaml"

    def __new__(cls, config_path: str | Path | None = None) -> "Settings":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_settings(config_path or cls._config_path)
        return cls._instance

    def _init_settings(self, config_path: str | Path) -> None:
        """Load configuration from disk and validate it against AppSettings."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        type(self)._settings = AppSettings(**raw_data)

    def __getattr__(self, item):
        """
        Called only if the attribute is not found on the instance itself,
        so it delegates to the loaded config (e.g. .telemetry_path, .llm, .language)
        """
        settings = type(self)._settings
        if settings is not None and hasattr(settings, item):
            return getattr(settings, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    @classmethod
    def reload(cls, config_path: str | Path = "config/settings.yaml") -> "Settings":
        """Force a reload from disk, bypassing the cache."""
        cls._settings = None
        cls._instance = None
        return cls(config_path)

    @classmethod
    def save(cls, config_path: str | Path | None = None) -> None:
        """Persist the current settings back to the YAML file on disk."""
        if cls._settings is None:
            cls()  # force loading with defaults first
        path = Path(config_path or cls._config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # mode="json" ensures enums, Path, etc. are serialized as plain values
        data = cls._settings.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            
    @classmethod
    def change_language(cls, language: Language) -> None:
        """Update the active language, loading defaults first if needed."""
        if cls._settings is None:
            cls()  # force loading with defaults
        cls._settings.language = language

    @classmethod
    def toggle_profanity_filter(cls, flag: bool) -> None:
        """Enable or disable the profanity filter, loading defaults first if needed."""
        if cls._settings is None:
            cls()  # force loading with defaults
        cls._settings.profanity_filter = flag