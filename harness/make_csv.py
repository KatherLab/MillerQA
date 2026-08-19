"""
make_csv.py — Generate CSV outputs from JSONL result files.

Called automatically at the end of run_experiment.py, or directly:

    python make_csv.py                          # uses EXPERIMENT_ID from config
    python make_csv.py --experiment-id zeroshot_v1

Output structure
────────────────
  results/<experiment_id>/csv/
    master.csv                     all records, all models, all benchmarks
    by_model/<model_label>.csv     one file per model (all benchmarks)
    by_benchmark/<benchmark>.csv   one file per benchmark (all models)

Columns written (list/dict fields from the JSONL are stringified or split):
  experiment_id, benchmark, question_id, kind,
  model_id, model_label, reasoning_enabled,
  correct_letter,
  selected_letter, selected_original_index, is_correct,
  detected_difficulty, cot_reasoning, explanation, confidence,
  parse_failed, kg_triples_retrieved,
  logprob_selected_token, logprob_selected_value,
  prompt_tokens, completion_tokens, total_tokens,
  cost_usd, latency_ms, finish_reason, error

Fields deliberately excluded from CSV (too large or structured):
  raw_content, reasoning_content, permutation,
  logprob_top_tokens, messages (these stay in JSONL / io log)

Note: cot_reasoning is NULL for zero-shot/few-shot; populated for cot_v*.
      detected_difficulty is NULL for zero-shot/few-shot/cot; populated for mdagents_v*.
      kg_triples_retrieved is NULL for non-medrag experiments; integer count for medrag_v*.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Columns to include in CSV output, in order
CSV_COLUMNS = [
    "experiment_id",
    "benchmark",
    "question_id",
    "kind",
    "model_id",
    "model_label",
    "reasoning_enabled",
    "correct_letter",
    "selected_letter",
    "selected_original_index",
    "is_correct",
    "detected_difficulty",
    "cot_reasoning",
    "explanation",
    "confidence",
    "parse_failed",
    "kg_triples_retrieved",
    "logprob_selected_token",
    "logprob_selected_value",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "latency_ms",
    "finish_reason",
    "error",
]


def _iter_jsonl(path: Path):
    """Yield parsed records from a JSONL file, skipping corrupt lines."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt line in %s", path)


def _to_csv_row(record: dict) -> dict:
    """Project a JSONL record down to the CSV column set."""
    return {col: record.get(col) for col in CSV_COLUMNS}


def _write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(_to_csv_row(rec))
    logger.info("Wrote %d rows to %s", len(records), path)


def generate_csvs(experiment_id: str) -> list[Path]:
    """
    Read all result JSONL files under results/<experiment_id>/ and write CSVs.

    Returns list of Paths to the files that were written.
    """
    exp_dir = config.RESULTS_DIR / experiment_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Collect result JSONL files only (exclude _io.jsonl and _agent_log.jsonl)
    jsonl_files = [
        p for p in exp_dir.rglob("*.jsonl")
        if not p.name.endswith("_io.jsonl")
        and not p.name.endswith("_agent_log.jsonl")
    ]

    if not jsonl_files:
        logger.warning("No result JSONL files found in %s", exp_dir)
        return []

    # Read all records
    all_records: list[dict] = []
    for path in sorted(jsonl_files):
        for rec in _iter_jsonl(path):
            all_records.append(rec)

    if not all_records:
        logger.warning("No records found across all JSONL files.")
        return []

    csv_dir = exp_dir / "csv"
    written: list[Path] = []

    # Master CSV
    master_path = csv_dir / "master.csv"
    _write_csv(master_path, all_records)
    written.append(master_path)

    # By-model CSVs
    by_model: dict[str, list[dict]] = {}
    for rec in all_records:
        label = rec.get("model_label", "unknown")
        by_model.setdefault(label, []).append(rec)

    for model_label, records in sorted(by_model.items()):
        path = csv_dir / "by_model" / f"{model_label}.csv"
        _write_csv(path, records)
        written.append(path)

    # By-benchmark CSVs
    by_benchmark: dict[str, list[dict]] = {}
    for rec in all_records:
        bmark = rec.get("benchmark", "unknown")
        by_benchmark.setdefault(bmark, []).append(rec)

    for benchmark, records in sorted(by_benchmark.items()):
        path = csv_dir / "by_benchmark" / f"{benchmark}.csv"
        _write_csv(path, records)
        written.append(path)

    logger.info(
        "CSV generation complete: %d master rows, %d model files, %d benchmark files",
        len(all_records),
        len(by_model),
        len(by_benchmark),
    )
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate CSVs from experiment JSONL results.")
    parser.add_argument(
        "--experiment-id",
        default=config.EXPERIMENT_ID,
        help=f"Experiment ID (default: {config.EXPERIMENT_ID})",
    )
    args = parser.parse_args()

    paths = generate_csvs(args.experiment_id)
    print(f"\n{len(paths)} CSV files written:")
    for p in paths:
        print(f"  {p}")
