from __future__ import annotations
import json
from pathlib import Path
import time
import logging
import threading
from dataclasses import dataclass, asdict, field

telemetry_logger = logging.getLogger("telemetry")  # raw log, one line per request


@dataclass
class RequestTelemetry:
    model_identifier: str
    ttft_ms: float | None = None
    total_duration_ms: float | None = None
    tokens_generated: int | None = None
    tokens_source: str = "estimated"
    throughput_tok_s: float | None = None
    vram_allocated_mb: float | None = None
    vram_source: str = "unavailable"
    success: bool = True
    error: str | None = None

    def finalize(self):
        if self.total_duration_ms and self.tokens_generated:
            gen_time_s = max(self.total_duration_ms - (self.ttft_ms or 0), 1e-6) / 1000
            self.throughput_tok_s = round(self.tokens_generated / gen_time_s, 2)


@dataclass
class ModelAggregateStats:
    """Cumulative statistics for a single model, updated incrementally
    (running average) so we don't need to keep every sample in memory."""
    model_identifier: str
    request_count: int = 0
    error_count: int = 0
    avg_ttft_ms: float = 0.0
    avg_throughput_tok_s: float = 0.0
    avg_total_duration_ms: float = 0.0
    avg_vram_mb: float = 0.0
    min_ttft_ms: float = field(default=float("inf"))
    max_ttft_ms: float = 0.0

    def update(self, sample: RequestTelemetry):
        self.request_count += 1
        n = self.request_count

        if not sample.success:
            self.error_count += 1
            return  # don't let failed requests pollute latency/throughput averages

        # incremental average: avg_new = avg_old + (x - avg_old) / n
        if sample.ttft_ms is not None:
            self.avg_ttft_ms += (sample.ttft_ms - self.avg_ttft_ms) / n
            self.min_ttft_ms = min(self.min_ttft_ms, sample.ttft_ms)
            self.max_ttft_ms = max(self.max_ttft_ms, sample.ttft_ms)
        if sample.throughput_tok_s is not None:
            self.avg_throughput_tok_s += (sample.throughput_tok_s - self.avg_throughput_tok_s) / n
        if sample.total_duration_ms is not None:
            self.avg_total_duration_ms += (sample.total_duration_ms - self.avg_total_duration_ms) / n
        if sample.vram_allocated_mb is not None:
            self.avg_vram_mb += (sample.vram_allocated_mb - self.avg_vram_mb) / n


class TelemetryStore:
    """Thread-safe singleton that aggregates telemetry samples across all
    requests, grouped by model, for the lifetime of the process."""

    _instance: "TelemetryStore | None" = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: dict[str, ModelAggregateStats] = {}
        self._output_path: Path | None = None  # set via configure_persistence()

    @classmethod
    def instance(cls) -> "TelemetryStore":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def configure_persistence(self, output_path: Path):
        """Sets the file the snapshot will be saved to. Should be called once
        at startup, before any request comes in."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = output_path

    def initialize_models(self, model_identifiers: list[str]):
        """Creates empty entries (zero requests) for all models known to the
        Model Registry, so the telemetry file exists from startup with the
        full model list, even before any of them receive a request."""
        with self._lock:
            for model_id in model_identifiers:
                self._stats.setdefault(model_id, ModelAggregateStats(model_id))
        self._save_to_file()  # write the file immediately with the empty entries

    def record(self, sample: RequestTelemetry):
        telemetry_logger.info(asdict(sample))

        with self._lock:
            stats = self._stats.setdefault(
                sample.model_identifier, ModelAggregateStats(sample.model_identifier)
            )
            stats.update(sample)

        self._save_to_file()  # <-- update the file after every request

    def snapshot(self) -> dict[str, dict]:
        """Returns an immutable copy of the current state, for reporting."""
        with self._lock:
            return {name: asdict(s) for name, s in self._stats.items()}

    def reset(self, model_identifier: str | None = None):
        with self._lock:
            if model_identifier:
                self._stats.pop(model_identifier, None)
            else:
                self._stats.clear()
        self._save_to_file()

    def _save_to_file(self):
        if self._output_path is None:
            return  # persistence not configured, silent no-op

        try:
            data = self.snapshot()
            tmp_path = self._output_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.replace(self._output_path)  # atomic write: avoids a corrupted file if the process crashes mid-write
        except Exception:
            telemetry_logger.exception("Failed to persist telemetry snapshot")


class TelemetryRecorder:
    """Context manager for a single request. Once measurement is complete,
    forwards the sample to the aggregate store (not just to the log)."""

    def __init__(self, model_identifier: str):
        self.metrics = RequestTelemetry(model_identifier=model_identifier)
        self._start: float | None = None
        self._first_token_seen = False

    def __enter__(self) -> "TelemetryRecorder":
        self._start = time.perf_counter()
        return self

    def record_chunk(self):
        """Called for each chunk received during streaming."""
        if not self._first_token_seen:
            self.metrics.ttft_ms = round((time.perf_counter() - self._start) * 1000, 2)
            self._first_token_seen = True
        self.metrics.tokens_generated = (self.metrics.tokens_generated or 0) + 1

    def set_usage(self, completion_tokens: int):
        """If the provider returns exact usage, overrides the chunk-based estimate."""
        self.metrics.tokens_generated = completion_tokens
        self.metrics.tokens_source = "usage"

    def set_vram(self, mb: float | None, source: str):
        self.metrics.vram_allocated_mb = mb
        self.metrics.vram_source = source

    def __exit__(self, exc_type, exc, tb):
        self.metrics.total_duration_ms = round((time.perf_counter() - self._start) * 1000, 2)
        if exc is not None:
            self.metrics.success = False
            self.metrics.error = str(exc)
        self.metrics.finalize()
        TelemetryStore.instance().record(self.metrics)
        return False  # does not suppress exceptions