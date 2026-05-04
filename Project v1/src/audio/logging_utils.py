from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def logs_dir() -> Path:
    path = project_root() / "results" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_token(value: str, fallback: str = "run") -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return sanitized or fallback


def write_run_log(run_type: str, name: str | None, content: str) -> Path:
    timestamp = datetime.now().astimezone()
    timestamp_slug = timestamp.strftime("%Y%m%d_%H%M%S")
    filename_parts = [timestamp_slug, _sanitize_token(run_type, fallback="run")]
    if name:
        filename_parts.append(_sanitize_token(name, fallback="session"))

    path = logs_dir() / f"{'_'.join(filename_parts)}.txt"
    header_lines = [
        f"Timestamp: {timestamp.isoformat(timespec='seconds')}",
        f"Run type: {run_type}",
    ]
    if name:
        header_lines.append(f"Name: {name}")

    path.write_text(
        "\n".join(header_lines) + "\n\n" + content.rstrip() + "\n",
        encoding="utf-8",
    )
    return path
