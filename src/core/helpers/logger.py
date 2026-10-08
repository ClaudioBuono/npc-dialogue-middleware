import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from core.helpers.paths import get_base_path

LOG_FILENAME = "npc_middleware.log"
LOG_DATE_FORMAT = "%Y%m%d"  # e.g. 20261008: files sort chronologically by name
LOG_RETENTION_DAYS = 14     # dated log files older than this are deleted at startup


def _daily_log_path(day: date | None = None) -> Path:
    """Build the path of the log file for the given day (default: today).

    ``npc_middleware.log`` becomes ``<exe_dir>/logs/npc_middleware_20261008.log``.
    """
    day = day or date.today()
    base = Path(LOG_FILENAME)
    name = f"{base.stem}_{day.strftime(LOG_DATE_FORMAT)}{base.suffix}"
    return get_base_path() / "logs" / name


def _delete_old_logs(retention_days: int) -> list[str]:
    """Delete dated log files older than ``retention_days``.

    Only files whose name matches ``<stem>_<date><suffix>`` are considered,
    so unrelated files in the logs folder (and the old undated
    ``npc_middleware.log``) are never touched.

    Returns:
        The names of the deleted files.
    """
    base = Path(LOG_FILENAME)
    cutoff = date.today() - timedelta(days=retention_days)
    deleted: list[str] = []

    for path in (get_base_path() / "logs").glob(f"{base.stem}_*{base.suffix}"):
        day_part = path.name[len(base.stem) + 1 : len(path.name) - len(base.suffix)]
        try:
            file_day = datetime.strptime(day_part, LOG_DATE_FORMAT).date()
        except ValueError:
            continue  # not a dated log file: leave it alone

        if file_day < cutoff:
            try:
                path.unlink()
                deleted.append(path.name)
            except OSError:
                pass  # e.g. still in use by another instance: try again next startup

    return sorted(deleted)


class DailyFileHandler(logging.FileHandler):
    """File handler that writes to one file per day and switches at midnight.

    Each record goes to the file of the day it was created, so a session that
    runs past midnight continues in the next day's file instead of staying in
    the file of the day the application started.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._current_day = date.today()
        super().__init__(_daily_log_path(self._current_day), mode="a", encoding=encoding)

    def emit(self, record: logging.LogRecord) -> None:
        # emit() is called with the handler lock held, so switching file here is thread-safe
        day = date.fromtimestamp(record.created)
        if day != self._current_day:
            try:
                self._switch_to(day)
            except Exception:
                self.handleError(record)  # keep writing to the current file
        super().emit(record)

    def _switch_to(self, day: date) -> None:
        self._current_day = day  # set first: a failure is reported once, not on every record
        new_path = _daily_log_path(day)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_stream = open(new_path, "a", encoding=self.encoding)

        old_stream = self.stream
        self.stream = new_stream
        self.baseFilename = str(new_path.resolve())
        if old_stream is not None:
            old_stream.flush()
            old_stream.close()


def setup_logging(level=logging.INFO):
    """Configure logging for the entire project.

    Sets up both a stream handler (stdout) and a file handler writing to one
    file per day (``<exe_dir>/logs/npc_middleware_yyyymmdd.log``): the file is
    created if missing, otherwise new records are appended to it, and it
    switches to the next day's file at midnight. Dated files older than
    ``LOG_RETENTION_DAYS`` are deleted at startup.

    Args:
        level: The minimum logging level for the root logger
            (e.g. logging.DEBUG, logging.INFO).
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Prevent adding handlers multiple times if called more than once
    if logger.handlers:
        return

    # Date format: dd-mm-yyyy HH:MM:SS
    date_format = "%d-%m-%Y %H:%M:%S"
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt=date_format
    )

    # Console Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Suppress noisy third-party debug logs
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)

    # File Handler
    # One file per day: <exe_dir>/logs/npc_middleware_yyyymmdd.log
    log_path = _daily_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)  # create the folder if missing
    deleted = _delete_old_logs(LOG_RETENTION_DAYS)

    file_handler = DailyFileHandler(encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    startup_logger = logging.getLogger(__name__)
    startup_logger.info("Logging to %s", log_path)
    if deleted:
        startup_logger.info(
            "Deleted %d log file(s) older than %d days: %s",
            len(deleted), LOG_RETENTION_DAYS, ", ".join(deleted),
        )