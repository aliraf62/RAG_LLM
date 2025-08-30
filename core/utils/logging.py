import logging
import time
from pathlib import Path
import json
from logging import Handler

def configure_logging(level: int = logging.INFO, log_to_file: bool = False, log_dir: str = "logs") -> None:
    """
    Configure root logging for the application.
    """
    handlers: list[Handler] = [logging.StreamHandler()]
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(Path(log_dir) / "app.log"), encoding="utf-8")
        handlers.append(file_handler)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers
    )

def get_logger(name: str = "") -> logging.Logger:
    """
    Get a logger with the given name, pre-configured.
    """
    return logging.getLogger(name)

def log_event(event_type: str, data: dict, log_dir: str = "logs") -> None:
    """
    Log an event to a timestamped JSONL file in the logs directory.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{event_type}_{time.strftime('%Y%m%d')}.jsonl"
    entry = {"timestamp": time.time(), "event_type": event_type, **data}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logging.getLogger(__name__).debug(f"Logged event: {entry}")
