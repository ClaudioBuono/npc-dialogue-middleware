import pytest
from pydantic import ValidationError
import yaml

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
    """Verifies successful loading and parsing from a valid YAML file."""

    valid_file = tmp_path / "valid.yaml"
    valid_data = [{"id": "model-1", "endpoint": "http://ep1", "intended_tier": "high"}]
    valid_file.write_text(yaml.safe_dump(valid_data), encoding="utf-8")

    configs = load_config_from_file(valid_file)
    assert len(configs) == 1
    assert configs[0].id == "model-1"
    assert configs[0].intended_tier == ComplexityTier.HIGH


def test_load_config_from_file_failures(tmp_path):
    """Verifies all failure modes during file reading and parsing."""

    # 1. File not found
    with pytest.raises(FileNotFoundError):
        load_config_from_file(tmp_path / "does_not_exist.yaml")

    # 2. Path is a directory instead of a file
    with pytest.raises(ValueError) as excinfo:
        load_config_from_file(tmp_path)
    assert "Config path is not a file" in str(excinfo.value)

    # 3. Invalid YAML formatting (unclosed flow sequence)
    invalid_yaml_file = tmp_path / "invalid_yaml.yaml"
    invalid_yaml_file.write_text("models: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config_from_file(invalid_yaml_file)
    assert "Invalid YAML in config file" in str(excinfo.value)

    # 4. YAML is valid but does not match expected list structure
    non_list_file = tmp_path / "non_list.yaml"
    non_list_file.write_text(
        yaml.safe_dump({"id": "model-1", "endpoint": "http://ep"}), encoding="utf-8"
    )
    with pytest.raises(ValueError) as excinfo:
        load_config_from_file(non_list_file)
    assert "Expected models_config to be a list" in str(excinfo.value)

    # 5. Schema validation failure inside the file
    invalid_schema_file = tmp_path / "invalid_schema.yaml"
    invalid_schema_file.write_text(yaml.safe_dump([{"id": "only-id"}]), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config_from_file(invalid_schema_file)

    # 6. Empty YAML file (yaml.safe_load returns None -> non-list ValueError)
    empty_yaml_file = tmp_path / "empty.yaml"
    empty_yaml_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config_from_file(empty_yaml_file)
    assert "Expected models_config to be a list" in str(excinfo.value)


def test_load_config_from_file_os_error(tmp_path):
    """Verifies that an OSError during read_text is wrapped in a ValueError."""
    dummy_file = tmp_path / "unreadable.yaml"
    dummy_file.write_text("dummy", encoding="utf-8")

    from unittest.mock import patch
    with patch("pathlib.Path.read_text", side_effect=OSError("Permission denied")):
        with pytest.raises(ValueError) as excinfo:
            load_config_from_file(dummy_file)
        assert "Could not read config file" in str(excinfo.value)


def test_load_config_from_file_string_path(tmp_path):
    """Verifies that string paths (not just Path objects) are supported."""
    valid_file = tmp_path / "valid_str.yaml"
    valid_data = [{"id": "model-str", "endpoint": "http://ep-str"}]
    valid_file.write_text(yaml.safe_dump(valid_data), encoding="utf-8")

    configs = load_config_from_file(str(valid_file))
    assert len(configs) == 1
    assert configs[0].id == "model-str"


def test_load_config_from_file_empty_list(tmp_path):
    """Verifies loading a YAML file containing an empty list returns an empty list."""
    empty_list_file = tmp_path / "empty_list.yaml"
    empty_list_file.write_text(yaml.safe_dump([]), encoding="utf-8")

    configs = load_config_from_file(empty_list_file)
    assert configs == []


def test_load_model_configs_empty_list():
    """Verifies that an empty list returns an empty list without error."""
    assert load_model_configs([]) == []


def test_load_model_configs_non_dict_element():
    """Verifies that non-dict elements inside the list trigger a ValidationError."""
    with pytest.raises(ValidationError):
        load_model_configs(["not-a-dict"])


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"id": "m1", "endpoint": "http://ep", "intended_tier": "invalid-tier"},
        {"id": "m1", "endpoint": "http://ep", "max_context_tokens": "not-an-int"},
    ]
)
def test_model_config_invalid_field_types(invalid_payload):
    """Verifies that invalid enum values or type mismatches raise a ValidationError."""
    with pytest.raises(ValidationError):
        ModelConfig(**invalid_payload)