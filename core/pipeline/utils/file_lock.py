"""
Cross-process file lock using the filesystem.
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Iterator


@contextlib.contextmanager
def file_lock(lock_path: Path, check_interval: float = 0.2) -> Iterator[None]:
    """
    Naïve but portable advisory lock.

    Example
    -------
    ```python
    with file_lock(Path("index.lock")):
        build_index()
    ```
    """
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(check_interval)
    try:
        yield
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)