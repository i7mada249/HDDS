from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


COLAB_WAV_ROOT = "/content/hdds_audio/work/wav_data/"


@dataclass(frozen=True)
class ManifestRow:
    path: str
    label: int
    source: str
    group: str
    category: str
    resolved_path: Path | None


def resolve_manifest_path(path: str, local_data_root: Path | None = None) -> Path | None:
    if local_data_root is None:
        return None

    if path.startswith(COLAB_WAV_ROOT):
        relative = path.removeprefix(COLAB_WAV_ROOT)
        return local_data_root / relative

    original = Path(path)
    if original.is_absolute():
        return original
    return local_data_root / original


def load_manifest(
    manifest_path: Path,
    local_data_root: Path | None = None,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            path = raw["path"]
            rows.append(
                ManifestRow(
                    path=path,
                    label=int(raw["label"]),
                    source=raw.get("source", ""),
                    group=raw.get("group", ""),
                    category=raw.get("category", ""),
                    resolved_path=resolve_manifest_path(
                        path,
                        local_data_root=local_data_root,
                    ),
                )
            )
    return rows
