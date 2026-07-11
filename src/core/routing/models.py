from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class IntendedTier(str, Enum):
    """Developer-declared power tier for a model, used for self-assessment and 
    as a fallback bucket when real profiling is unavailable."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# TODO: define the input file structure
class ModelConfig(BaseModel):
    """A single model entry as declared in the models configuration file."""

    id: str = Field(..., description="Unique identifier for this model within the middleware.")
    endpoint: str = Field(..., description="Base URL of the model's OpenAI-compatible API endpoint.")
    intended_tier: Optional[IntendedTier] = Field(
        None,
        description="Developer-declared power tier. Used directly when profiler=False, "
                    "and as a fallback when automatic profiling is not yet available."
    )
    max_context_tokens: Optional[int] = Field(None, description="Maximum context window size supported by the model, in tokens.")
    api_key: Optional[str] = Field(None, description="API key, if required by the provider.")


# TODO: pass the file from the API, this function will not be called by the user
def load_model_configs(config_path: str) -> List[ModelConfig]:
    """Loads and validates model configurations from a YAML file.

    Args:
        config_path: Path to the YAML configuration file (expects a
            top-level `models:` list, each entry matching ModelConfig).

    Returns:
        A list of validated ModelConfig instances.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValidationError: If any entry does not match the expected schema.
    """
    raise NotImplementedError("TODO: pass the file from the API, this function will not be called by the user")

    return [ModelConfig.model_validate(entry) for entry in raw.get("models", [])]