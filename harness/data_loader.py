"""
data_loader.py — Load and normalize benchmark datasets.

Every benchmark file is normalised to a flat list of BenchmarkItem dicts:
    {
        "benchmark":  str,        # filename stem, e.g. "bbq_safety"
        "id":         str,        # original record id (cast to str)
        "question":   str,
        "options":    list[str],  # original option texts, 0-indexed
        "target":     int,        # index into options that is correct
        "kind":       str | None, # subcategory / split label if present
    }

Usage:
    from data_loader import load_all, load_benchmarks

    items = load_all()                          # every benchmark in config.DATA_DIR
    items = load_benchmarks(["fairmedqa"])      # single benchmark
"""

import json
import logging
from pathlib import Path
from typing import Optional

# Imported as a module, not `from config import DATA_DIR`, so that a --data-dir
# override applied to config at startup is picked up here.
import config

logger = logging.getLogger(__name__)

OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _load_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: expected a JSON array, got {type(raw).__name__}")

    benchmark = path.stem
    items = []
    for record in raw:
        options = record.get("options")
        if not isinstance(options, list) or len(options) < 2:
            logger.warning("Skipping record %s in %s: bad options", record.get("id"), benchmark)
            continue

        target = record.get("target")
        if not isinstance(target, int) or not (0 <= target < len(options)):
            logger.warning("Skipping record %s in %s: bad target", record.get("id"), benchmark)
            continue

        # 'kind' field is called 'categories' in medethicsqa — normalise
        kind = record.get("kind") or record.get("categories")
        if isinstance(kind, list):
            kind = ", ".join(str(k) for k in kind)

        items.append(
            {
                "benchmark": benchmark,
                "id": str(record["id"]),
                "question": record["question"].strip(),
                "options": [str(o).strip() for o in options],
                "target": target,
                "kind": str(kind) if kind is not None else None,
            }
        )

    return items


def load_benchmarks(names: Optional[list[str]] = None) -> list[dict]:
    """
    Load specific benchmarks by stem name, or all if names is None.
    Returns a flat list of normalised BenchmarkItem dicts.
    """
    # rglob, not glob: benchmarks are grouped into one directory per upstream
    # source so each can carry its own LICENSE. The benchmark name is still the
    # file stem, so the layout is invisible to everything downstream.
    paths = sorted(config.DATA_DIR.rglob("*.json"))
    if names:
        name_set = set(names)
        paths = [p for p in paths if p.stem in name_set]
        missing = name_set - {p.stem for p in paths}
        if missing:
            raise FileNotFoundError(f"Benchmarks not found: {missing}")

    items = []
    for path in paths:
        loaded = _load_file(path)
        items.extend(loaded)
        logger.info("Loaded %d items from %s", len(loaded), path.name)

    logger.info("Total items loaded: %d", len(items))
    return items


def load_all() -> list[dict]:
    return load_benchmarks(names=None)


def option_letter(index: int) -> str:
    """Return the letter label for a 0-based option index (0→'A', 1→'B', …)."""
    return OPTION_LETTERS[index]


def letter_to_index(letter: str) -> Optional[int]:
    """Convert an answer letter back to a 0-based index, or None if invalid."""
    # Must be exactly one A–Z character. `letter in OPTION_LETTERS` would be a
    # substring test, which accepts "" and multi-letter runs like "AB".
    letter = letter.strip().upper()
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        return None
    return ord(letter) - ord("A")
