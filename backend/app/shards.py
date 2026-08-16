from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SHARD_PATTERN = re.compile(
    r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GgufShard:
    base: str
    index: int
    count: int


def parse_gguf_shard(filename: str) -> GgufShard | None:
    match = SHARD_PATTERN.match(filename.replace("\\", "/"))
    if not match:
        return None
    return GgufShard(
        base=match.group("base"),
        index=int(match.group("index")),
        count=int(match.group("count")),
    )


def model_name_from_filename(filename: str) -> str:
    shard = parse_gguf_shard(filename)
    if shard:
        return Path(shard.base).name
    return Path(filename).stem


def local_model_files(path: str) -> list[Path]:
    primary = Path(path)
    shard = parse_gguf_shard(primary.name)
    if not shard:
        return [primary]
    matches: list[tuple[int, Path]] = []
    for candidate in primary.parent.glob("*.gguf"):
        candidate_shard = parse_gguf_shard(candidate.name)
        if candidate_shard and candidate_shard.base == shard.base and candidate_shard.count == shard.count:
            matches.append((candidate_shard.index, candidate))
    return [candidate for _, candidate in sorted(matches)] or [primary]
