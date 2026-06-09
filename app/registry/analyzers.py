from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.config import settings

_YAML = Path(__file__).resolve().parent / "analyzers.yaml"


@dataclass(frozen=True)
class Analyzer:
    key: str
    arch: str
    label: str
    training: str  # "stego" | "alaska2"
    weight: str  # filename under {weights_dir}/analyzers/

    @property
    def path(self) -> Path:
        return settings.weights_dir / "analyzers" / self.weight

    @property
    def available(self) -> bool:
        return self.path.exists()


@lru_cache(maxsize=1)
def _index() -> dict[str, Analyzer]:
    rows = yaml.safe_load(_YAML.read_text()) or []
    return {r["key"]: Analyzer(**r) for r in rows}


def list_analyzers() -> list[Analyzer]:
    return list(_index().values())


def resolve(key: str) -> Analyzer:
    try:
        return _index()[key]
    except KeyError as exc:
        raise KeyError(f"Unknown steganalyzer: {key!r}") from exc
