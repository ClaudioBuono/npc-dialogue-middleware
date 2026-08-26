import json
import pytest
from pydantic import ValidationError

from core.routing.models import ModelConfig, load_model_configs, load_config_from_file
from core.types.enums import ComplexityTier


# ------------------------------------------------------------------
# ModelConfig Validation Tests
# ------------------------------------------------------------------

def test_model_config_validation_happy_path():
    """Verifies that a fully populated and a default ModelConfig are instantiated correctly."""
    
    # Fully populated config
    config = ModelConfig(
        id="test-model",
        endpoint="http://localhost:8000/v1",
        intended_tier=ComplexityTier.LOW,
        max_context_tokens=4096,
        api_key="test-key"
    )
    assert config.id == "test-model"
    assert config.endpoint == "http://localhost:8000/v1"
    assert config.intended_tier == ComplexityTier.LOW
    assert config.max_context_tokens == 4096
    assert config.api_key == "test-key"

    # Default optional fields
    config_default = ModelConfig(
        id="test-model",
        endpoint="http://localhost:8000/v1"
    )
    assert config_default.intended_tier is None
    assert config_default.max_context_tokens is None
    assert config_default.api_key is None


@pytest.mark.parametrize(
    "payload, missing_field",
    [
        ({"endpoint": "http://localhost:8000/v1"}, "id"),
        ({"id": "test-model"}, "endpoint"),
    ]
)
def test_model_config_missing_required_fields(payload, missing_field):
    """Verifies that missing required fields explicitly raise a ValidationError."""
    
    with pytest.raises(ValidationError) as excinfo:
        ModelConfig(**payload)
    
    # Ensure the error is specifically about the missing field
    assert missing_field in str(excinfo.value)
    assert "Field required" in str(excinfo.value)


# ------------------------------------------------------------------
# load_model_configs Tests
# ------------------------------------------------------------------

def test_load_model_configs_valid():
    """Verifies successful parsing of a valid list of configuration dictionaries."""

    data = [
        {"id": "model-1", "endpoint": "http://ep1"},
        {"id": "model-2", "endpoint": "http://ep2", "intended_tier": "medium"}
    ]
    configs = load_model_configs(data)
    assert len(configs) == 2
    assert configs[0].id == "model-1"
    assert configs[1].intended_tier == ComplexityTier.MEDIUM


def test_load_model_configs_invalid_types():
    """Verifies rejection of non-list inputs."""

    with pytest.raises(ValueError) as excinfo:
        load_model_configs({"not": "a-list"})
    assert "Expected models_config to be a list" in str(excinfo.value)


def test_load_model_configs_invalid_schema():
    """Verifies that incorrect item schemas raise ValidationErrors."""

    with pytest.raises(ValidationError) as excinfo:
        load_model_configs([{"id": "model-1"}]) # Missing 'endpoint'
    assert "endpoint" in str(excinfo.value)


# ------------------------------------------------------------------
# load_config_from_file File I/O Tests
# ------------------------------------------------------------------

def test_load_config_from_file_success(tmp_path):
    """Verifies successful loading and parsing from a valid JSON file."""

    valid_file = tmp_path / "valid.json"
    valid_data = [{"id": "model-1", "endpoint": "http://ep1", "intended_tier": "high"}]
    valid_file.write_text(json.dumps(valid_data), encoding="utf-8")

    configs = load_config_from_file(valid_file)
    assert len(configs) == 1
    assert configs[0].id == "model-1"
    assert configs[0].intended_tier == ComplexityTier.HIGH


def test_load_config_from_file_failures(tmp_path):
    """Verifies all failure modes during file reading and parsing."""
    
    # 1. File not found
    with pytest.raises(FileNotFoundError):
        load_config_from_file(tmp_path / "does_not_exist.json")

    # 2. Path is a directory instead of a file
    with pytest.raises(ValueError) as excinfo:
        load_config_from_file(tmp_path)
    assert "Config path is not a file" in str(excinfo.value)

    # 3. Invalid JSON formatting
    invalid_json_file = tmp_path / "invalid_json.json"
    invalid_json_file.write_text("invalid json content", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config_from_file(invalid_json_file)
    assert "Invalid JSON in config file" in str(excinfo.value)

    # 4. JSON is valid but does not match expected list structure
    non_list_file = tmp_path / "non_list.json"
    non_list_file.write_text(json.dumps({"id": "model-1", "endpoint": "http://ep"}), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config_from_file(non_list_file)
    assert "Expected models_config to be a list" in str(excinfo.value)

    # 5. Schema validation failure inside the file
    invalid_schema_file = tmp_path / "invalid_schema.json"
    invalid_schema_file.write_text(json.dumps([{"id": "only-id"}]), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config_from_file(invalid_schema_file)