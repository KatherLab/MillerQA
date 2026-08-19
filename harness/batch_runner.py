"""
batch_runner.py — High-throughput async runner for one (model, dataset) pair.

Concurrency model
─────────────────
Each question is an independent async task. A semaphore caps the number of
in-flight API calls at config.MAX_CONCURRENT. Tasks are submitted all at once
via asyncio.gather, so Python's event loop interleaves them efficiently without
any sequential waiting between questions.

Output
──────
Results are written incrementally to two files as each task completes:

  results/<experiment_id>/<model_label>/<benchmark>.jsonl
      One record per question — parsed answer, correctness, metrics, etc.
      This is the primary analysis file.

  results/<experiment_id>/<model_label>/<benchmark>_io.jsonl
      Full input/output log — the exact prompt sent and the raw text returned
      by the model, plus a UTC timestamp. Kept separate so it doesn't bloat
      the analysis file, but is always available for auditing / debugging.

Resume behaviour
────────────────
On any restart the runner reads already-completed question IDs from the
results JSONL and skips those questions entirely. When all items are already
done it also re-reads the file to recompute the summary statistics, so the
final report table is always accurate.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm as atqdm

import config
from api_client import call_model
from output_parser import build_result_record
from prompt_builder import build_shuffled_question

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _results_path(experiment_id: str, model_label: str, benchmark: str) -> Path:
    p = config.RESULTS_DIR / experiment_id / model_label / f"{benchmark}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _io_log_path(experiment_id: str, model_label: str, benchmark: str) -> Path:
    """Separate file for full prompt + raw response per query."""
    p = config.RESULTS_DIR / experiment_id / model_label / f"{benchmark}_io.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _load_completed_ids(path: Path) -> set[str]:
    """
    Return the set of question_ids already written to a JSONL file.
    Skips corrupt / partial lines gracefully.
    """
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ids.add(rec["question_id"])
            except Exception:
                pass
    return ids


def _is_retryable_failure(rec: dict) -> bool:
    """
    Return True for records that are worth retrying:
      - API errors (4xx/5xx, including 402 credit exhaustion and 429 rate limits)
      - parse_failed due to truncation (finish_reason == "length")
    Excludes content_filter blocks and model refusals (finish_reason == "stop"
    with no JSON) since those won't improve on retry.
    """
    if rec.get("error"):
        return True
    if rec.get("parse_failed") and rec.get("finish_reason") == "length":
        return True
    return False


def _strip_parse_failed(path: Path, io_path: Path) -> int:
    """
    Rewrite the results JSONL removing only retryable failures.
    Also rewrites the io log to match (keeps only the same question_ids).
    Returns the number of records removed.
    """
    if not path.exists():
        return 0

    kept = []
    removed_ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if _is_retryable_failure(rec):
                    removed_ids.add(rec["question_id"])
                else:
                    kept.append(line)
            except Exception:
                pass

    if not removed_ids:
        return 0

    with open(path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    # Mirror the strip in the io log
    if io_path.exists():
        io_kept = []
        with open(io_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("question_id") not in removed_ids:
                        io_kept.append(line)
                except Exception:
                    pass
        with open(io_path, "w", encoding="utf-8") as f:
            for line in io_kept:
                f.write(line + "\n")

    return len(removed_ids)


def _read_all_records(path: Path) -> list[dict]:
    """Read all valid records from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


# ---------------------------------------------------------------------------
# Single-question task
# ---------------------------------------------------------------------------

async def _run_one(
    item: dict,
    model_cfg: dict,
    semaphore: asyncio.Semaphore,
    out_file,
    io_file,
    experiment_id: str,
    lock: asyncio.Lock,
    build_prompt=None,
) -> dict:
    """Acquire semaphore, call API, parse result, write both output files."""
    if build_prompt is not None:
        sq = build_prompt(item, model_cfg["label"])
    else:
        sq = build_shuffled_question(item, model_cfg["label"], config.SHUFFLE_SEED)

    async with semaphore:
        api_resp = await call_model(model_cfg, sq.prompt_messages)

    record = build_result_record(item, sq, api_resp, model_cfg, experiment_id)

    io_record = {
        "experiment_id": experiment_id,
        "benchmark": item["benchmark"],
        "question_id": item["id"],
        "model_label": model_cfg["label"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "messages": sq.prompt_messages,
        "raw_content": api_resp.raw_content,
        "reasoning_content": api_resp.reasoning_content,
        "reasoning_tokens": api_resp.reasoning_tokens,
        "finish_reason": api_resp.finish_reason,
        "error": api_resp.error,
        # Full provider response (incl. provider-specific extras like `reasoning`,
        # `generation_id`, etc.) so post-hoc analyses can recover fields the
        # current extractor doesn't know about.
        "raw_response": api_resp.raw_response,
    }

    async with lock:
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        io_file.write(json.dumps(io_record, ensure_ascii=False) + "\n")
        io_file.flush()

    return record


# ---------------------------------------------------------------------------
# Per-benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    items: list[dict],
    model_cfg: dict,
    experiment_id: str,
    semaphore: asyncio.Semaphore,
    force: bool = False,
    retry_failed: bool = False,
    build_prompt=None,
) -> dict:
    """
    Run one model against one benchmark's items.
    Returns a summary dict with accuracy and usage statistics.

    If force=True, existing results are overwritten and all items are re-run.
    If retry_failed=True, parse_failed/errored records are stripped first so
    resume logic will re-run only those question IDs.
    """
    benchmark = items[0]["benchmark"] if items else "unknown"
    model_label = model_cfg["label"]

    out_path = _results_path(experiment_id, model_label, benchmark)
    io_path = _io_log_path(experiment_id, model_label, benchmark)

    if force:
        completed_ids: set[str] = set()
        file_mode = "w"
        logger.info("[%s / %s] Force rerun — ignoring existing results.", model_label, benchmark)
    else:
        if retry_failed:
            n_stripped = _strip_parse_failed(out_path, io_path)
            if n_stripped:
                logger.info(
                    "[%s / %s] Stripped %d parse_failed/errored records — will re-run those.",
                    model_label, benchmark, n_stripped,
                )
        completed_ids = _load_completed_ids(out_path)
        file_mode = "a"

    pending = [it for it in items if it["id"] not in completed_ids]
    skipped = len(items) - len(pending)

    if skipped:
        logger.info(
            "[%s / %s] Resuming: %d already done, %d remaining",
            model_label, benchmark, skipped, len(pending),
        )

    # All items already done — re-read the file for accurate summary stats
    if not pending:
        logger.info("[%s / %s] All items already completed.", model_label, benchmark)
        existing = _read_all_records(out_path)
        return _compute_summary(model_label, benchmark, existing, skipped=0)

    lock = asyncio.Lock()
    new_results: list[dict] = []

    with (
        open(out_path, file_mode, encoding="utf-8") as f,
        open(io_path, file_mode, encoding="utf-8") as io_f,
    ):
        tasks = [
            _run_one(item, model_cfg, semaphore, f, io_f, experiment_id, lock, build_prompt)
            for item in pending
        ]
        desc = f"{model_label} / {benchmark}"
        for coro in atqdm.as_completed(tasks, total=len(tasks), desc=desc, leave=False):
            result = await coro
            new_results.append(result)

    # Merge with already-skipped records for a complete summary
    if skipped:
        all_records = _read_all_records(out_path)
        return _compute_summary(model_label, benchmark, all_records, skipped=0)

    return _compute_summary(model_label, benchmark, new_results, skipped=0)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _compute_summary(model_label: str, benchmark: str, results: list[dict], skipped: int = 0) -> dict:
    total = len(results)

    errored = sum(1 for r in results if r.get("error"))
    parse_failed = sum(1 for r in results if r.get("parse_failed") and not r.get("error"))

    gradeable = [r for r in results if r.get("is_correct") is not None]
    accuracy = (
        sum(1 for r in gradeable if r["is_correct"]) / len(gradeable)
        if gradeable else None
    )

    total_tokens = sum(r.get("total_tokens") or 0 for r in results)
    total_cost = sum(r["cost_usd"] for r in results if r.get("cost_usd") is not None)
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
    mean_latency = sum(latencies) / len(latencies) if latencies else None

    return {
        "model_label": model_label,
        "benchmark": benchmark,
        "total": total,
        "skipped_at_resume": skipped,
        "completed": total - errored,
        "errored": errored,
        "parse_failed": parse_failed,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "mean_latency_ms": round(mean_latency, 1) if mean_latency else None,
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

async def run_experiment(
    all_items: list[dict],
    model_cfgs: list[dict],
    experiment_id: str,
    benchmarks: Optional[list[str]] = None,
    force: bool = False,
    retry_failed: bool = False,
    build_prompt=None,
) -> list[dict]:
    """
    Run experiment_id across all model_cfgs and benchmarks.

    Concurrency design (mirrors the reference runner):
    - Every (model, benchmark) pair is launched as a concurrent task via
      asyncio.gather — no sequential waiting between models or benchmarks.
    - Each model gets its own semaphore so models don't compete with each
      other for slots. Reasoning models get a higher limit (they run slower
      per request but can sustain more in-flight calls efficiently).
    - The per-model limit comes from model_cfg["concurrency"] with a
      fallback to config.MAX_CONCURRENT.

    Parameters
    ----------
    all_items    : flat list of all BenchmarkItem dicts
    model_cfgs   : list of model config dicts from config.MODELS
    experiment_id: experiment identifier (also the results sub-folder name)
    benchmarks   : if provided, only run these benchmark stems

    Returns list of per-(model, benchmark) summary dicts.
    """
    # Group items by benchmark
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for item in all_items:
        by_benchmark[item["benchmark"]].append(item)

    if benchmarks:
        missing = set(benchmarks) - set(by_benchmark)
        if missing:
            raise ValueError(f"Benchmarks not found in data: {missing}")
        by_benchmark = {k: v for k, v in by_benchmark.items() if k in benchmarks}

    # Per-model semaphore — reasoning models default to a larger pool because
    # each request takes longer, so more can safely be in-flight at once.
    model_semaphores = {
        model_cfg["label"]: asyncio.Semaphore(
            model_cfg.get("concurrency", config.MAX_CONCURRENT)
        )
        for model_cfg in model_cfgs
    }

    t_start = time.perf_counter()

    # Build all (model, benchmark) tasks and run fully in parallel
    tasks = []
    task_meta = []  # track (label, benchmark_name) for logging
    for model_cfg in model_cfgs:
        sem = model_semaphores[model_cfg["label"]]
        for benchmark_name, items in sorted(by_benchmark.items()):
            tasks.append(run_benchmark(items, model_cfg, experiment_id, sem, force=force, retry_failed=retry_failed, build_prompt=build_prompt))
            task_meta.append((model_cfg["label"], benchmark_name))

    logger.info(
        "Launching %d concurrent workers (%d models × %d benchmarks)",
        len(tasks), len(model_cfgs), len(by_benchmark),
    )

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries: list[dict] = []
    for (label, benchmark_name), result in zip(task_meta, raw_results):
        if isinstance(result, Exception):
            logger.error("[%s / %s] Worker raised: %s", label, benchmark_name, result)
            summaries.append({
                "model_label": label,
                "benchmark": benchmark_name,
                "error": str(result),
            })
        else:
            summaries.append(result)
            logger.info(
                "[%s / %s] done — acc=%s  tokens=%d  cost=$%.4f  err=%d  parse_fail=%d",
                label,
                benchmark_name,
                f"{result['accuracy']:.3f}" if result.get("accuracy") is not None else "N/A",
                result.get("total_tokens") or 0,
                result.get("total_cost_usd") or 0,
                result.get("errored") or 0,
                result.get("parse_failed") or 0,
            )

    elapsed = time.perf_counter() - t_start
    logger.info("Experiment complete in %.1f s", elapsed)

    # Persist summary JSON (overwrite — always reflects current full state)
    summary_path = config.RESULTS_DIR / experiment_id / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Summary written to %s", summary_path)

    return summaries
