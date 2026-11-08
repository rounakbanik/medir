"""Small JSON/JSONL and checksum helpers shared across pipeline stages.

JSONL stores one complete JSON object per line. Writes use a temporary file
followed by replacement so readers do not see a partially written individual
output file. A group of several output files is not a single transaction.
"""

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def jsonable(value):
    """Convert a top-level Pydantic record into JSON-compatible Python values.

    Pydantic JSON mode converts dates and other supported types. Other values are
    passed through; callers supply json.dumps(default=str) as a final fallback.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def atomic_text(path: Path, text: str):
    """Write complete UTF-8 text, replacing the destination only after success.

    Args:
        path: Destination file; missing parent directories are created.
        text: Complete text to persist.

    A temporary file in the same directory keeps replacement on one filesystem.
    The finally block cleans up a leftover temporary file if writing fails.
    This protects a single file, not an entire multi-file pipeline stage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create the temporary file beside the target so replacement stays local.
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        # Readers see the old complete file or the new complete file, not partial text.
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            # Remove abandoned temporary files even when serialization/writing raises.
            os.unlink(name)


def write_json(path: Path, value):
    """Serialize one value as readable, indented JSON with a final newline.

    Reuses atomic_text so an interrupted serialization does not replace a valid
    destination with an incomplete JSON file. Unicode names remain readable.
    """
    atomic_text(path, json.dumps(jsonable(value), indent=2, ensure_ascii=False, default=str) + "\n")


def write_jsonl(path: Path, rows: Iterable):
    """Stream records into a temporary JSONL file and replace the destination.

    Args:
        path: Output JSONL file, written as uncompressed UTF-8.
        rows: Iterable of Pydantic records or JSON-compatible Python objects.

    Rows may be a generator, avoiding a second full in-memory corpus copy.
    """
    # Stream to a temporary file: a corpus need not fit twice in memory.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create the temporary file beside the target so replacement stays local.
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(jsonable(row), ensure_ascii=False, default=str) + "\n")
        # Readers see the old complete file or the new complete file, not partial text.
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            # Remove abandoned temporary files even when serialization/writing raises.
            os.unlink(name)


def read_jsonl(path: Path):
    """Yield decoded objects from a plain or gzip-compressed UTF-8 JSONL file.

    Blank lines are ignored. Record schema validation is the caller's job;
    this helper checks JSON syntax and adds a file/line number to parse errors.

    Raises:
        ValueError: A nonblank line is not valid JSON.
    """
    # The filename suffix selects decompression; parsed rows have the same interface.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        # Use human-friendly one-based line numbers in malformed-input errors.
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error


def digest(path: Path) -> str:
    """Return a file's SHA-256 checksum while reading it in 1 MiB blocks.

    Chunked reads bound temporary memory even for large source exports. The
    checksum changes when any original byte changes, including whitespace.
    """
    # Hash original bytes rather than parsed JSON so any source-file edit is detected.
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()
