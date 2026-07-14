import json
from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field, ValidationError
from core.types.enums import ComplexityTier
from pathlib import Path

class ModelConfig(BaseModel):
    """A single model entry as declared in the models configuration file."""

    id: str = Field(..., description="Unique identifier for this model within the middleware.")
    endpoint: str = Field(..., description="Base URL of the model's OpenAI-compatible API endpoint.")
    intended_tier: Optional[ComplexityTier] = Field(
        None,
        description="Developer-declared power tier. Used directly when profiler=False, "
                    "and as a fallback when automatic profiling is not yet available."
    )
    max_context_tokens: Optional[int] = Field(None, description="Maximum context window size supported by the model, in tokens.")
    api_key: Optional[str] = Field(None, description="API key, if required by the provider.")


def load_model_configs(models_config: List[dict[str, Any]]) -> List[ModelConfig]:
    """Loads and validates model configurations from an already-parsed list of dicts.

    Args:
        models_config: Parsed configuration data (JSON already loaded by the
            caller/API layer) — a list where each entry matches ModelConfig.

    Returns:
        A list of validated ModelConfig instances.

    Raises:
        ValueError: If `models_config` is not a list.
        ValidationError: If any entry does not match the expected schema.
    """
    if not isinstance(models_config, list):
        raise ValueError(
            f"Expected models_config to be a list, got {type(models_config).__name__}"
        )

    try:
        return [ModelConfig.model_validate(entry) for entry in models_config]
    except ValidationError:
        raise


def load_config_from_file(config_path: Union[str, Path]) -> List[ModelConfig]:
    """Loads and validates model configurations from a JSON file on disk.

    Args:
        config_path: Path to the JSON file. Expects the file's top-level
            content to be a list of objects, each matching ModelConfig.

    Returns:
        A list of validated ModelConfig instances.

    Raises:
        FileNotFoundError: If the file does not exist at `config_path`.
        ValueError: If the file content is not valid JSON, or is valid JSON
            but not a list.
        ValidationError: If any entry does not match the expected schema.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    try:
        raw_content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"Could not read config file {path}: {e}") from e

    try:
        models_config = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {path}: {e}") from e

    return load_model_configs(models_config)
