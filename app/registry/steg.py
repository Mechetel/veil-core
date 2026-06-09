from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.config import settings

_YAML = Path(__file__).resolve().parent / "steg.yaml"


@dataclass(frozen=True)
class StegModel:
    key: str
    family: str
    label: str
    dataset: str
    data_depth: int
    weight: str  # filename under {weights_dir}/steg/

    @property
    def path(self) -> Path:
        return settings.weights_dir / "steg" / self.weight

    @property
    def available(self) -> bool:
        return self.path.exists()


@lru_cache(maxsize=1)
def _index() -> dict[str, StegModel]:
    rows = yaml.safe_load(_YAML.read_text()) or []
    return {r["key"]: StegModel(**r) for r in rows}


def list_models() -> list[StegModel]:
    return list(_index().values())


def resolve(key: str) -> StegModel:
    try:
        return _index()[key]
    except KeyError as exc:
        raise KeyError(f"Unknown steganography model: {key!r}") from exc
