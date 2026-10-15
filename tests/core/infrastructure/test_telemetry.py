"""Unit tests for core.infrastructure.telemetry – request metrics, aggregation, and store persistence."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import json
import logging
import math
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.infrastructure.telemetry import (
    ModelAggregateStats,
    RequestTelemetry,
    TelemetryRecorder,
    TelemetryStore,
    VramReading,
)


@pytest.fixture(autouse=True)
def clean_telemetry_store():
    """Ensure every test starts and ends with a clean TelemetryStore singleton state."""
    TelemetryStore._instance = None
    yield
    TelemetryStore._instance = None


# ===========================================================================
# VramReading Tests
# ===========================================================================

class TestVramReading:
    """Verification of VramReading dataclass immutability and factory methods."""

    def test_vram_reading_instantiation(self):
        """VramReading correctly stores used_mb and source."""
        reading = VramReading(used_mb=4096.5, source="nvml")
        assert reading.used_mb == 4096.5
        assert reading.source == "nvml"

    def test_vram_reading_frozen_immutability(self):
        """VramReading is a frozen dataclass; attribute assignment raises FrozenInstanceError."""
        reading = VramReading(used_mb=2048.0, source="ollama_ps")
        with pytest.raises(FrozenInstanceError):
            reading.used_mb = 1024.0  # type: ignore

    def test_vram_reading_unavailable_factory(self):
        """unavailable() factory method sets used_mb to None and source to 'unavailable'."""
        reading = VramReading.unavailable()
        assert reading.used_mb is None
        assert reading.source == "unavailable"

    @pytest.mark.parametrize("source", ["ollama_ps", "nvml", "unavailable"])
    def test_vram_reading_supported_sources(self, source):
        """Validates all valid VramSource literal values."""
        reading = VramReading(used_mb=100.0, source=source)
        assert reading.source == source


# ===========================================================================
# RequestTelemetry Tests
# ===========================================================================

class TestRequestTelemetry:
    """Verification of RequestTelemetry defaults and finalize() throughput calculation."""

    def test_default_values(self):
        """RequestTelemetry initializes with sensible default attributes."""
        telemetry = RequestTelemetry(model_identifier="test-model")
        assert telemetry.model_identifier == "test-model"
        assert telemetry.ttft_ms is None
        assert telemetry.total_duration_ms is None
        assert telemetry.tokens_generated is None
        assert telemetry.tokens_source == "estimated"
        assert telemetry.throughput_tok_s is None
        assert telemetry.vram_allocated_mb is None
        assert telemetry.vram_source == "unavailable"
        assert telemetry.success is True
        assert telemetry.error is None

    def test_finalize_calculates_throughput_with_ttft(self):
        """Throughput uses generation time: (total_duration_ms - ttft_ms)."""
        # Duration: 2000ms, TTFT: 500ms -> Generation time = 1500ms = 1.5s
        # 60 tokens / 1.5s = 40.0 tok/s
        telemetry = RequestTelemetry(
            model_identifier="test-model",
            total_duration_ms=2000.0,
            ttft_ms=500.0,
            tokens_generated=60,
        )
        telemetry.finalize()
        assert telemetry.throughput_tok_s == 40.0

    def test_finalize_calculates_throughput_without_ttft(self):
        """When ttft_ms is None, throughput is computed over full duration."""
        # Duration: 1000ms = 1.0s, TTFT: None
        # 50 tokens / 1.0s = 50.0 tok/s
        telemetry = RequestTelemetry(
            model_identifier="test-model",
            total_duration_ms=1000.0,
            ttft_ms=None,
            tokens_generated=50,
        )
        telemetry.finalize()
        assert telemetry.throughput_tok_s == 50.0

    def test_finalize_skips_when_total_duration_is_none(self):
        """throughput_tok_s remains None when total_duration_ms is missing."""
        telemetry = RequestTelemetry(
            model_identifier="test-model",
            total_duration_ms=None,
            tokens_generated=50,
        )
        telemetry.finalize()
        assert telemetry.throughput_tok_s is None

    def test_finalize_skips_when_tokens_generated_is_none(self):
        """throughput_tok_s remains None when tokens_generated is missing."""
        telemetry = RequestTelemetry(
            model_identifier="test-model",
            total_duration_ms=1000.0,
            tokens_generated=None,
        )
        telemetry.finalize()
        assert telemetry.throughput_tok_s is None

    def test_finalize_skips_when_tokens_generated_is_zero(self):
        """throughput_tok_s remains None when tokens_generated is 0 (falsy check)."""
        telemetry = RequestTelemetry(
            model_identifier="test-model",
            total_duration_ms=1000.0,
            tokens_generated=0,
        )
        telemetry.finalize()
        assert telemetry.throughput_tok_s is None

    def test_finalize_guards_against_zero_or_negative_generation_time(self):
        """Clamps generation time to 1e-6 seconds when total_duration_ms <= ttft_ms."""
        # total_duration_ms equals ttft_ms -> subtraction yields 0.0 -> clamped to 1e-6
        telemetry = RequestTelemetry(
            model_identifier="test-model",
            total_duration_ms=500.0,
            ttft_ms=500.0,
            tokens_generated=10,
        )
        telemetry.finalize()
        assert telemetry.throughput_tok_s is not None
        assert telemetry.throughput_tok_s > 0

        # Anomaly where ttft_ms > total_duration_ms
        telemetry2 = RequestTelemetry(
            model_identifier="test-model",
            total_duration_ms=400.0,
            ttft_ms=500.0,
            tokens_generated=10,
        )
        telemetry2.finalize()
        assert telemetry2.throughput_tok_s is not None
        assert telemetry2.throughput_tok_s > 0

    def test_finalize_rounds_to_two_decimal_places(self):
        """Throughput is properly rounded to 2 decimals."""
        # 33 tokens in 0.7 seconds (1000ms duration - 300ms ttft) -> 33 / 0.7 = 47.1428... -> 47.14
        telemetry = RequestTelemetry(
            model_identifier="test-model",
            total_duration_ms=1000.0,
            ttft_ms=300.0,
            tokens_generated=33,
        )
        telemetry.finalize()
        assert telemetry.throughput_tok_s == 47.14


# ===========================================================================
# ModelAggregateStats Tests
# ===========================================================================

class TestModelAggregateStats:
    """Verification of cumulative running average statistics and error isolation."""

    def test_initial_aggregate_values(self):
        """New ModelAggregateStats initializes counts to 0 and min_ttft to infinity."""
        stats = ModelAggregateStats(model_identifier="meta/llama-3")
        assert stats.model_identifier == "meta/llama-3"
        assert stats.request_count == 0
        assert stats.error_count == 0
        assert stats.avg_ttft_ms == 0.0
        assert stats.avg_throughput_tok_s == 0.0
        assert stats.avg_total_duration_ms == 0.0
        assert stats.avg_vram_mb == 0.0
        assert stats.min_ttft_ms == float("inf")
        assert stats.max_ttft_ms == 0.0

    def test_update_single_successful_sample(self):
        """First sample sets initial running average and min/max bounds."""
        stats = ModelAggregateStats(model_identifier="meta/llama-3")
        sample = RequestTelemetry(
            model_identifier="meta/llama-3",
            ttft_ms=250.0,
            throughput_tok_s=45.0,
            total_duration_ms=1200.0,
            vram_allocated_mb=4096.0,
            success=True,
        )
        stats.update(sample)

        assert stats.request_count == 1
        assert stats.error_count == 0
        assert stats.avg_ttft_ms == 250.0
        assert stats.min_ttft_ms == 250.0
        assert stats.max_ttft_ms == 250.0
        assert stats.avg_throughput_tok_s == 45.0
        assert stats.avg_total_duration_ms == 1200.0
        assert stats.avg_vram_mb == 4096.0

    def test_update_multiple_successful_samples_running_average(self):
        """Incremental average formula accurately tracks arithmetic mean across samples."""
        stats = ModelAggregateStats(model_identifier="meta/llama-3")

        samples = [
            RequestTelemetry("meta/llama-3", ttft_ms=100.0, throughput_tok_s=20.0, total_duration_ms=1000.0, vram_allocated_mb=1000.0),
            RequestTelemetry("meta/llama-3", ttft_ms=200.0, throughput_tok_s=40.0, total_duration_ms=2000.0, vram_allocated_mb=3000.0),
            RequestTelemetry("meta/llama-3", ttft_ms=300.0, throughput_tok_s=60.0, total_duration_ms=3000.0, vram_allocated_mb=2000.0),
        ]

        for s in samples:
            stats.update(s)

        assert stats.request_count == 3
        assert stats.error_count == 0
        assert stats.avg_ttft_ms == pytest.approx(200.0)
        assert stats.min_ttft_ms == pytest.approx(100.0)
        assert stats.max_ttft_ms == pytest.approx(300.0)
        assert stats.avg_throughput_tok_s == pytest.approx(40.0)
        assert stats.avg_total_duration_ms == pytest.approx(2000.0)
        assert stats.avg_vram_mb == pytest.approx(2000.0)

    def test_failed_sample_increments_error_count_without_polluting_averages(self):
        """Failed requests increment error_count and return early, preserving latency averages."""
        stats = ModelAggregateStats(model_identifier="meta/llama-3")

        # Initial successful sample
        sample_ok = RequestTelemetry(
            model_identifier="meta/llama-3",
            ttft_ms=150.0,
            throughput_tok_s=50.0,
            total_duration_ms=1000.0,
            vram_allocated_mb=2048.0,
            success=True,
        )
        stats.update(sample_ok)

        # Subsequent failed sample
        sample_fail = RequestTelemetry(
            model_identifier="meta/llama-3",
            ttft_ms=9999.0,
            throughput_tok_s=0.1,
            total_duration_ms=10000.0,
            vram_allocated_mb=8192.0,
            success=False,
            error="Connection timed out",
        )
        stats.update(sample_fail)

        assert stats.request_count == 2
        assert stats.error_count == 1
        # Metrics must remain exactly what they were before the failure
        assert stats.avg_ttft_ms == 150.0
        assert stats.min_ttft_ms == 150.0
        assert stats.max_ttft_ms == 150.0
        assert stats.avg_throughput_tok_s == 50.0
        assert stats.avg_total_duration_ms == 1000.0
        assert stats.avg_vram_mb == 2048.0

    def test_update_sample_with_partial_none_metrics(self):
        """Fields with None values in the sample do not update or crash running averages."""
        stats = ModelAggregateStats(model_identifier="meta/llama-3")
        sample = RequestTelemetry(
            model_identifier="meta/llama-3",
            ttft_ms=None,
            throughput_tok_s=None,
            total_duration_ms=800.0,
            vram_allocated_mb=None,
            success=True,
        )
        stats.update(sample)

        assert stats.request_count == 1
        assert stats.avg_total_duration_ms == 800.0
        assert stats.avg_ttft_ms == 0.0
        assert stats.avg_throughput_tok_s == 0.0
        assert stats.avg_vram_mb == 0.0
        assert stats.min_ttft_ms == float("inf")
        assert stats.max_ttft_ms == 0.0


# ===========================================================================
# TelemetryStore Tests
# ===========================================================================

class TestTelemetryStore:
    """Verification of TelemetryStore singleton, atomic persistence, and sanitization."""

    def test_singleton_instance(self):
        """TelemetryStore.instance() returns the identical store singleton."""
        store1 = TelemetryStore.instance()
        store2 = TelemetryStore.instance()
        assert store1 is store2

    def test_initialize_models_creates_empty_records(self):
        """initialize_models creates ModelAggregateStats records with 0 requests."""
        store = TelemetryStore.instance()
        store.initialize_models(["model-a", "model-b"])

        snapshot = store.snapshot()
        assert "model-a" in snapshot
        assert "model-b" in snapshot
        assert snapshot["model-a"]["request_count"] == 0
        assert snapshot["model-b"]["request_count"] == 0

    def test_initialize_models_preserves_existing_stats(self):
        """initialize_models does not overwrite existing models that already collected data."""
        store = TelemetryStore.instance()
        sample = RequestTelemetry("model-a", total_duration_ms=500.0, success=True)
        store.record(sample)

        # Call initialize_models including model-a and new model-c
        store.initialize_models(["model-a", "model-c"])
        snapshot = store.snapshot()
        assert snapshot["model-a"]["request_count"] == 1
        assert snapshot["model-c"]["request_count"] == 0

    def test_record_creates_and_updates_model_stats(self):
        """record adds a new sample and updates the aggregate stats."""
        store = TelemetryStore.instance()
        sample = RequestTelemetry(
            model_identifier="mistral/small",
            ttft_ms=180.0,
            throughput_tok_s=35.0,
            total_duration_ms=900.0,
            tokens_generated=25,
            success=True,
        )
        store.record(sample)

        snapshot = store.snapshot()
        assert "mistral/small" in snapshot
        model_stats = snapshot["mistral/small"]
        assert model_stats["request_count"] == 1
        assert model_stats["avg_ttft_ms"] == 180.0
        assert model_stats["avg_throughput_tok_s"] == 35.0

    def test_record_logs_to_telemetry_logger(self, caplog):
        """record emits a JSON formatted log entry to the 'telemetry' logger."""
        store = TelemetryStore.instance()
        sample = RequestTelemetry(model_identifier="gpt-4o", success=True)

        with caplog.at_level(logging.INFO, logger="telemetry"):
            store.record(sample)

        assert any("Request telemetry:" in record.message for record in caplog.records)
        assert any('"model_identifier": "gpt-4o"' in record.message for record in caplog.records)

    def test_snapshot_sanitizes_infinite_min_ttft(self):
        """Unmeasured min_ttft_ms (float('inf')) is converted to None in snapshot."""
        store = TelemetryStore.instance()
        store.initialize_models(["uncalled-model"])

        raw_snapshot = store.snapshot()
        assert raw_snapshot["uncalled-model"]["min_ttft_ms"] is None

    def test_sanitize_floats_helper(self):
        """_sanitize_floats converts non-finite floats to None recursively."""
        payload = {
            "a": float("inf"),
            "b": float("-inf"),
            "c": float("nan"),
            "d": 42.5,
            "nested": {
                "e": [1.0, float("inf"), {"inner": float("nan")}],
            },
            "int_val": 10,
            "str_val": "text",
            "none_val": None,
        }

        sanitized = TelemetryStore._sanitize_floats(payload)
        assert sanitized["a"] is None
        assert sanitized["b"] is None
        assert sanitized["c"] is None
        assert sanitized["d"] == 42.5
        assert sanitized["nested"]["e"][0] == 1.0
        assert sanitized["nested"]["e"][1] is None
        assert sanitized["nested"]["e"][2]["inner"] is None
        assert sanitized["int_val"] == 10
        assert sanitized["str_val"] == "text"
        assert sanitized["none_val"] is None

    def test_reset_entire_store(self):
        """reset() without arguments clears all model stats."""
        store = TelemetryStore.instance()
        store.initialize_models(["model-1", "model-2"])
        assert len(store.snapshot()) == 2

        store.reset()
        assert len(store.snapshot()) == 0

    def test_reset_specific_model(self):
        """reset(model_id) removes only the specified model."""
        store = TelemetryStore.instance()
        store.initialize_models(["keep-model", "remove-model"])

        store.reset("remove-model")
        snapshot = store.snapshot()
        assert "keep-model" in snapshot
        assert "remove-model" not in snapshot

    def test_persistence_disabled_by_default(self):
        """When persistence is not configured, _save_to_file is a silent no-op."""
        store = TelemetryStore.instance()
        # _output_path is None by default
        store.record(RequestTelemetry("model-test"))
        store.reset()
        # No error raised, nothing persisted

    def test_configure_persistence_creates_parent_directories(self, tmp_path):
        """configure_persistence creates missing parent directories for the output file."""
        target_dir = tmp_path / "deeply" / "nested" / "metrics"
        target_file = target_dir / "telemetry_snapshot.json"

        store = TelemetryStore.instance()
        store.configure_persistence(target_file)

        assert target_dir.exists()
        assert target_dir.is_dir()

    def test_persistence_atomic_file_write_and_update(self, tmp_path):
        """TelemetryStore saves state atomically and writes valid JSON to disk."""
        target_file = tmp_path / "telemetry.json"
        store = TelemetryStore.instance()
        store.configure_persistence(target_file)

        store.initialize_models(["model-persist"])
        assert target_file.exists()

        content = json.loads(target_file.read_text(encoding="utf-8"))
        assert "model-persist" in content
        assert content["model-persist"]["request_count"] == 0

        # Record a sample and verify updated file contents
        sample = RequestTelemetry("model-persist", total_duration_ms=450.0, success=True)
        store.record(sample)

        updated_content = json.loads(target_file.read_text(encoding="utf-8"))
        assert updated_content["model-persist"]["request_count"] == 1

        # Verify temporary file is cleaned up after atomic replace
        tmp_file = target_file.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_save_to_file_handles_exceptions_gracefully(self, tmp_path, caplog):
        """Exceptions during file writing are logged and do not propagate to the caller."""
        target_file = tmp_path / "protected_dir" / "telemetry.json"
        store = TelemetryStore.instance()
        store.configure_persistence(target_file)

        with patch.object(Path, "write_text", side_effect=OSError("Disk full")):
            with caplog.at_level(logging.ERROR, logger="telemetry"):
                # Must not raise exception
                store.record(RequestTelemetry("model-err"))

        assert any("Failed to persist telemetry snapshot" in record.message for record in caplog.records)

    def test_concurrent_recording_and_snapshotting(self):
        """Multiple threads recording samples while snapshotting without data races."""
        store = TelemetryStore.instance()
        models = ["model-alpha", "model-beta", "model-gamma"]
        store.initialize_models(models)

        def worker(thread_idx: int):
            for i in range(15):
                m_id = models[(thread_idx + i) % len(models)]
                sample = RequestTelemetry(
                    model_identifier=m_id,
                    ttft_ms=100.0 + i,
                    total_duration_ms=500.0 + i,
                    tokens_generated=10 + i,
                    success=True,
                )
                store.record(sample)
                _ = store.snapshot()

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, i) for i in range(6)]
            for f in futures:
                f.result()

        snapshot = store.snapshot()
        for m in models:
            assert snapshot[m]["request_count"] > 0


# ===========================================================================
# TelemetryRecorder Tests
# ===========================================================================

class TestTelemetryRecorder:
    """Verification of TelemetryRecorder context manager lifecycle and metric collection."""

    def test_context_manager_normal_lifecycle(self):
        """Recording chunks captures ttft on first chunk, duration on exit, and records to store."""
        with TelemetryRecorder("model-rec") as recorder:
            recorder.record_chunk()
            recorder.record_chunk()
            recorder.record_chunk()

        assert recorder.metrics.ttft_ms is not None
        assert recorder.metrics.ttft_ms >= 0.0
        assert recorder.metrics.total_duration_ms is not None
        assert recorder.metrics.total_duration_ms >= recorder.metrics.ttft_ms
        assert recorder.metrics.tokens_generated == 3
        assert recorder.metrics.tokens_source == "estimated"
        assert recorder.metrics.success is True
        assert recorder.metrics.error is None
        assert recorder.metrics.throughput_tok_s is not None

        # Confirms sample was registered in the store
        snapshot = TelemetryStore.instance().snapshot()
        assert "model-rec" in snapshot
        assert snapshot["model-rec"]["request_count"] == 1

    def test_record_chunk_does_not_overwrite_first_ttft(self):
        """Subsequent record_chunk calls only increment token count and keep initial TTFT."""
        recorder = TelemetryRecorder("model-rec")
        # 1.0 (__enter__), 1.25 (first record_chunk), 2.0 (__exit__)
        with patch("core.infrastructure.telemetry.time.perf_counter", side_effect=[1.0, 1.25, 2.0]):
            with recorder:
                recorder.record_chunk()  # first token at 1.25 -> ttft = (1.25 - 1.0) * 1000 = 250ms
                first_ttft = recorder.metrics.ttft_ms
                recorder.record_chunk()  # subsequent tokens only increment tokens_generated
                recorder.record_chunk()

            assert recorder.metrics.ttft_ms == first_ttft == 250.0
            assert recorder.metrics.tokens_generated == 3

    def test_set_usage_overrides_estimated_tokens(self):
        """set_usage overrides chunk-based token count and switches tokens_source to 'usage'."""
        with TelemetryRecorder("model-rec") as recorder:
            recorder.record_chunk()
            recorder.record_chunk()
            assert recorder.metrics.tokens_generated == 2
            assert recorder.metrics.tokens_source == "estimated"

            recorder.set_usage(142)
            assert recorder.metrics.tokens_generated == 142
            assert recorder.metrics.tokens_source == "usage"

    def test_set_vram_records_snapshot(self):
        """set_vram stores allocated VRAM megabytes and source probe identifier."""
        reading = VramReading(used_mb=3580.25, source="nvml")
        with TelemetryRecorder("model-rec") as recorder:
            recorder.set_vram(reading)

        assert recorder.metrics.vram_allocated_mb == 3580.25
        assert recorder.metrics.vram_source == "nvml"

    def test_context_manager_captures_exception_and_does_not_suppress(self):
        """When an exception occurs in the block, records failure state and re-raises exception."""
        recorder_instance = None
        with pytest.raises(ValueError, match="LLM upstream failure"):
            with TelemetryRecorder("model-fail") as recorder:
                recorder_instance = recorder
                recorder.record_chunk()
                raise ValueError("LLM upstream failure")

        assert recorder_instance is not None
        assert recorder_instance.metrics.success is False
        assert recorder_instance.metrics.error == "LLM upstream failure"
        assert recorder_instance.metrics.total_duration_ms is not None

        # Verify sample was recorded into aggregate store as an error
        snapshot = TelemetryStore.instance().snapshot()
        assert "model-fail" in snapshot
        assert snapshot["model-fail"]["request_count"] == 1
        assert snapshot["model-fail"]["error_count"] == 1

    def test_deterministic_timings_and_throughput(self):
        """Verifies deterministic math with controlled perf_counter sequence."""
        # 0.0s: enter
        # 0.2s: first chunk (ttft = 200ms)
        # (subsequent chunks only increment token count and do not call perf_counter)
        # 1.2s: exit (total = 1200ms)
        # gen_time = (1200 - 200) / 1000 = 1.0s
        # 2 tokens / 1.0s = 2.0 tok/s
        perf_values = [0.0, 0.2, 1.2]
        with patch("core.infrastructure.telemetry.time.perf_counter", side_effect=perf_values):
            with TelemetryRecorder("model-timing") as recorder:
                recorder.record_chunk()
                recorder.record_chunk()

        assert recorder.metrics.ttft_ms == 200.0
        assert recorder.metrics.total_duration_ms == 1200.0
        assert recorder.metrics.tokens_generated == 2
        assert recorder.metrics.throughput_tok_s == 2.0

    def test_no_chunks_recorded(self):
        """When no chunks are recorded, ttft_ms and tokens_generated remain None."""
        with TelemetryRecorder("empty-stream") as recorder:
            pass

        assert recorder.metrics.ttft_ms is None
        assert recorder.metrics.tokens_generated is None
        assert recorder.metrics.throughput_tok_s is None
        assert recorder.metrics.total_duration_ms is not None
        assert recorder.metrics.success is True
