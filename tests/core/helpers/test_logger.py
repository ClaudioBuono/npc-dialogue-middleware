"""Unit tests for core.helpers.logger – daily log file rotation, retention cleanup, and configuration."""
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from core.helpers.logger import (
    LOG_DATE_FORMAT,
    LOG_FILENAME,
    LOG_RETENTION_DAYS,
    DailyFileHandler,
    _daily_log_path,
    _delete_old_logs,
    setup_logging,
)


@pytest.fixture
def mock_base_path(tmp_path, monkeypatch):
    """Direct all log file creation and reading to tmp_path."""
    monkeypatch.setattr("core.helpers.logger.get_base_path", lambda: tmp_path)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def restore_root_logger():
    """Ensure root logger handlers and log levels are completely restored after each test.

    Crucial on Windows to avoid open file handle locks (WinError 32) and logger pollution.
    """
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        yield root
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        root.handlers = original_handlers
        root.setLevel(original_level)


class TestDailyLogPath:
    """Tests for _daily_log_path helper function."""

    def test_daily_log_path_with_explicit_date(self, mock_base_path):
        """Builds path with explicit date: <base>/logs/npc_middleware_YYYYMMDD.log."""
        target_day = date(2026, 10, 15)
        path = _daily_log_path(target_day)
        expected = mock_base_path / "logs" / "npc_middleware_20261015.log"
        assert path == expected

    def test_daily_log_path_defaults_to_today(self, mock_base_path):
        """When day is None, defaults to today's date formatted with LOG_DATE_FORMAT."""
        today = date.today()
        path = _daily_log_path(None)
        expected_name = f"npc_middleware_{today.strftime(LOG_DATE_FORMAT)}.log"
        assert path == mock_base_path / "logs" / expected_name

    def test_logger_constants(self):
        """Verify standard log filename, date format, and retention period constants."""
        assert LOG_FILENAME == "npc_middleware.log"
        assert LOG_DATE_FORMAT == "%Y%m%d"
        assert LOG_RETENTION_DAYS == 14


class TestDeleteOldLogs:
    """Tests for _delete_old_logs log retention cleanup."""

    def test_deletes_logs_older_than_retention_days(self, mock_base_path):
        """Files strictly older than retention_days are unlinked, recent ones kept."""
        logs_dir = mock_base_path / "logs"
        today = date.today()

        # Create files of varying ages
        file_20_days_ago = logs_dir / f"npc_middleware_{(today - timedelta(days=20)).strftime(LOG_DATE_FORMAT)}.log"
        file_15_days_ago = logs_dir / f"npc_middleware_{(today - timedelta(days=15)).strftime(LOG_DATE_FORMAT)}.log"
        file_14_days_ago = logs_dir / f"npc_middleware_{(today - timedelta(days=14)).strftime(LOG_DATE_FORMAT)}.log"
        file_5_days_ago = logs_dir / f"npc_middleware_{(today - timedelta(days=5)).strftime(LOG_DATE_FORMAT)}.log"
        file_today = logs_dir / f"npc_middleware_{today.strftime(LOG_DATE_FORMAT)}.log"

        for f in (file_20_days_ago, file_15_days_ago, file_14_days_ago, file_5_days_ago, file_today):
            f.write_text("log content", encoding="utf-8")

        deleted = _delete_old_logs(retention_days=14)

        # 20 and 15 days ago are < cutoff (today - 14 days)
        expected_deleted = sorted([file_20_days_ago.name, file_15_days_ago.name])
        assert deleted == expected_deleted

        assert not file_20_days_ago.exists()
        assert not file_15_days_ago.exists()
        assert file_14_days_ago.exists()  # Exactly cutoff: not < cutoff
        assert file_5_days_ago.exists()
        assert file_today.exists()

    def test_ignores_undated_or_unrelated_files(self, mock_base_path):
        """Unrelated files and undated base log files are preserved."""
        logs_dir = mock_base_path / "logs"
        base_log = logs_dir / "npc_middleware.log"
        other_log = logs_dir / "other_service_20260101.log"
        readme = logs_dir / "README.txt"

        for f in (base_log, other_log, readme):
            f.write_text("info", encoding="utf-8")

        deleted = _delete_old_logs(retention_days=14)
        assert deleted == []
        assert base_log.exists()
        assert other_log.exists()
        assert readme.exists()

    def test_ignores_matching_stem_with_invalid_date(self, mock_base_path):
        """Files matching the stem and suffix but with invalid date strings are skipped."""
        logs_dir = mock_base_path / "logs"
        invalid_dated = logs_dir / "npc_middleware_notadate.log"
        invalid_month = logs_dir / "npc_middleware_20269999.log"

        for f in (invalid_dated, invalid_month):
            f.write_text("data", encoding="utf-8")

        deleted = _delete_old_logs(retention_days=14)
        assert deleted == []
        assert invalid_dated.exists()
        assert invalid_month.exists()

    def test_handles_empty_or_clean_logs_dir(self, mock_base_path):
        """Returns empty list when no log files exist."""
        deleted = _delete_old_logs(retention_days=14)
        assert deleted == []

    def test_handles_oserror_during_unlink(self, mock_base_path, monkeypatch):
        """When unlink raises OSError (e.g. file in use by another process), it is caught gracefully."""
        logs_dir = mock_base_path / "logs"
        old_day = date.today() - timedelta(days=30)
        locked_file = logs_dir / f"npc_middleware_{old_day.strftime(LOG_DATE_FORMAT)}.log"
        locked_file.write_text("locked", encoding="utf-8")

        def failing_unlink(self):
            raise OSError("Access is denied")

        monkeypatch.setattr(Path, "unlink", failing_unlink)

        deleted = _delete_old_logs(retention_days=14)
        assert deleted == []


class TestDailyFileHandler:
    """Tests for DailyFileHandler rotating files across day boundaries."""

    def test_handler_initialization(self, mock_base_path):
        """Initializes handler pointing to today's log file."""
        handler = DailyFileHandler(encoding="utf-8")
        try:
            today_str = date.today().strftime(LOG_DATE_FORMAT)
            expected_filename = f"npc_middleware_{today_str}.log"
            assert handler._current_day == date.today()
            assert Path(handler.baseFilename).name == expected_filename
            assert Path(handler.baseFilename).exists()
        finally:
            handler.close()

    def test_emit_writes_to_current_day_file(self, mock_base_path):
        """Emitted record is written to the current day's log file."""
        handler = DailyFileHandler(encoding="utf-8")
        try:
            record = logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname=__file__,
                lineno=10,
                msg="Test message line",
                args=(),
                exc_info=None,
            )
            handler.emit(record)
            handler.flush()

            log_file = Path(handler.baseFilename)
            assert "Test message line" in log_file.read_text(encoding="utf-8")
        finally:
            handler.close()

    def test_emit_switches_file_at_midnight(self, mock_base_path):
        """When a record timestamp has a different date, handler rotates to the new day's file."""
        handler = DailyFileHandler(encoding="utf-8")
        try:
            today_path = Path(handler.baseFilename)

            # Record 1: created today
            rec_today = logging.LogRecord(
                name="test", level=logging.INFO, pathname=__file__, lineno=1,
                msg="Today record", args=(), exc_info=None,
            )
            handler.emit(rec_today)

            # Record 2: created tomorrow (e.g. session ran past midnight)
            tomorrow = date.today() + timedelta(days=1)
            tomorrow_dt = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=1)
            rec_tomorrow = logging.LogRecord(
                name="test", level=logging.INFO, pathname=__file__, lineno=2,
                msg="Tomorrow record", args=(), exc_info=None,
            )
            rec_tomorrow.created = tomorrow_dt.timestamp()

            handler.emit(rec_tomorrow)
            handler.flush()

            tomorrow_path = Path(handler.baseFilename)
            assert tomorrow_path != today_path
            assert handler._current_day == tomorrow

            today_content = today_path.read_text(encoding="utf-8")
            tomorrow_content = tomorrow_path.read_text(encoding="utf-8")

            assert "Today record" in today_content
            assert "Tomorrow record" not in today_content
            assert "Tomorrow record" in tomorrow_content
        finally:
            handler.close()

    def test_switch_error_falls_back_to_handle_error(self, mock_base_path, monkeypatch):
        """If _switch_to raises an exception, handleError is invoked and handler does not crash."""
        handler = DailyFileHandler(encoding="utf-8")
        try:
            handler.handleError = MagicMock()

            def broken_switch(day):
                raise PermissionError("Disk is read only")

            monkeypatch.setattr(handler, "_switch_to", broken_switch)

            tomorrow_dt = datetime.combine(date.today() + timedelta(days=1), datetime.min.time())
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname=__file__, lineno=1,
                msg="Failed rollover record", args=(), exc_info=None,
            )
            record.created = tomorrow_dt.timestamp()

            handler.emit(record)
            handler.handleError.assert_called_once_with(record)
        finally:
            handler.close()

    def test_switch_to_closes_old_stream(self, mock_base_path):
        """Switching days properly flushes and closes the previous stream."""
        handler = DailyFileHandler(encoding="utf-8")
        try:
            old_stream = handler.stream
            assert not old_stream.closed

            tomorrow = date.today() + timedelta(days=1)
            handler._switch_to(tomorrow)

            assert old_stream.closed
            assert handler.stream != old_stream
            assert not handler.stream.closed
        finally:
            handler.close()


class TestSetupLogging:
    """Tests for setup_logging project configuration function."""

    def test_early_exit_if_already_configured(self, restore_root_logger):
        """If root logger already has handlers, setup_logging does not attach duplicates."""
        root = restore_root_logger
        dummy_handler = logging.NullHandler()
        root.addHandler(dummy_handler)
        initial_handlers_count = len(root.handlers)

        setup_logging(level=logging.WARNING)

        assert len(root.handlers) == initial_handlers_count
        root.removeHandler(dummy_handler)

    def test_setup_logging_clean_configuration(self, mock_base_path, restore_root_logger):
        """Clean configuration configures root logger, handlers, and suppresses third-party logs."""
        root = restore_root_logger
        # Remove any preexisting handlers for a clean test
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()

        setup_logging(level=logging.DEBUG)

        assert root.level == logging.DEBUG
        assert len(root.handlers) == 2

        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        file_handlers = [h for h in root.handlers if isinstance(h, DailyFileHandler)]

        assert len(stream_handlers) == 1
        assert len(file_handlers) == 1
        assert stream_handlers[0].stream == sys.stdout

        # Verify third-party log level suppression
        assert logging.getLogger("openai").level == logging.INFO
        assert logging.getLogger("httpcore").level == logging.INFO
        assert logging.getLogger("httpx").level == logging.INFO

    def test_setup_logging_reports_deleted_files(self, mock_base_path, restore_root_logger, caplog):
        """When old logs are deleted during startup, the deletion summary is logged."""
        root = restore_root_logger
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()

        logs_dir = mock_base_path / "logs"
        old_day = date.today() - timedelta(days=25)
        old_file = logs_dir / f"npc_middleware_{old_day.strftime(LOG_DATE_FORMAT)}.log"
        old_file.write_text("old", encoding="utf-8")

        with caplog.at_level(logging.INFO):
            setup_logging(level=logging.INFO)

        assert not old_file.exists()
        assert f"Deleted 1 log file(s) older than {LOG_RETENTION_DAYS} days" in caplog.text
        assert old_file.name in caplog.text
