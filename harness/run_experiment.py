"""
run_experiment.py — CLI entry point for running benchmark experiments.

Usage examples
──────────────
# Run all models on all benchmarks (uses EXPERIMENT_ID from config / .env)
python run_experiment.py

# Run specific models (by label) on specific benchmarks
python run_experiment.py --models gpt-4o claude-opus-4-6 --benchmarks fairmedqa mhqa

# Run against the original pre-QC release instead of the curated default
python run_experiment.py --data-dir ../benchmarks/pre_qc --experiment-id zeroshot_preqc

# Override experiment ID (creates a separate results folder)
python run_experiment.py --experiment-id zeroshot_v2

# Dry run: show what would be executed without calling the API
python run_experiment.py --dry-run

# Adjust concurrency on the fly (override config.MAX_CONCURRENT)
python run_experiment.py --concurrency 10
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import config
from batch_runner import run_experiment
from data_loader import load_all, load_benchmarks
from make_csv import generate_csvs


def _setup_logging(verbose: bool, experiment_id: str) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt))
    root.addHandler(ch)

    # File handler — one log file per experiment run
    log_dir = config.RESULTS_DIR / experiment_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _resolve_models(labels: list[str] | None) -> list[dict]:
    if not labels:
        return config.MODELS
    label_set = set(labels)
    found = [m for m in config.MODELS if m["label"] in label_set]
    missing = label_set - {m["label"] for m in found}
    if missing:
        avail = [m["label"] for m in config.MODELS]
        print(f"ERROR: Unknown model labels: {missing}")
        print(f"Available: {avail}")
        sys.exit(1)
    return found


def _print_plan(models: list[dict], benchmarks: list[str] | None, n_items: int) -> None:
    print("\n" + "=" * 60)
    print(f"  Experiment : {config.EXPERIMENT_ID}")
    print(f"  Data dir   : {config.DATA_DIR}")
    print(f"  Questions  : {n_items:,}")
    bmark_str = ", ".join(benchmarks) if benchmarks else "ALL"
    print(f"  Benchmarks : {bmark_str}")
    print(f"  Models     : {len(models)}")
    for m in models:
        reasoning = "  [reasoning]" if m.get("reasoning_config") else ""
        print(f"    · {m['label']}{reasoning}")
    print(f"  Concurrency: {config.MAX_CONCURRENT}")
    print(f"  Results dir: {config.RESULTS_DIR / config.EXPERIMENT_ID}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run zero-shot MCQ benchmark experiment via OpenRouter.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="LABEL",
        help="Model labels to run (default: all in config.MODELS)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        metavar="NAME",
        help="Benchmark stems to run (default: all in data/)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help=f"Benchmark directory to load (default: {config.DATA_DIR})",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        metavar="ID",
        help=f"Override experiment ID (default: {config.EXPERIMENT_ID})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        metavar="N",
        help=f"Max concurrent API calls (default: {config.MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results and rerun all items (ignores resume state)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run only items that previously had parse failures or errors",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan and exit without calling the API",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Apply overrides before setting up logging (log path depends on experiment_id)
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        if not data_dir.is_dir():
            print(f"ERROR: --data-dir does not exist: {data_dir}")
            sys.exit(1)
        config.DATA_DIR = data_dir
    if args.experiment_id:
        config.EXPERIMENT_ID = args.experiment_id
    if args.concurrency:
        config.MAX_CONCURRENT = args.concurrency

    _setup_logging(args.verbose, config.EXPERIMENT_ID)

    if not config.OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY is not set. Add it to your .env file or environment.")
        sys.exit(1)

    models = _resolve_models(args.models)

    # Load data
    if args.benchmarks:
        items = load_benchmarks(args.benchmarks)
    else:
        items = load_all()

    _print_plan(models, args.benchmarks, len(items))

    if args.dry_run:
        print("[dry-run] Exiting without calling API.")
        return

    summaries = asyncio.run(
        run_experiment(
            all_items=items,
            model_cfgs=models,
            experiment_id=config.EXPERIMENT_ID,
            benchmarks=args.benchmarks,
            force=args.force,
            retry_failed=args.retry_failed,
        )
    )

    # Print final table
    print("\n" + "=" * 80)
    print(f"{'MODEL':<35} {'BENCHMARK':<25} {'ACC':>6}  {'TOKENS':>8}  {'COST $':>8}")
    print("-" * 80)
    for s in summaries:
        acc = f"{s['accuracy']:.3f}" if s.get("accuracy") is not None else "  N/A"
        tokens = s.get("total_tokens") or 0
        cost = s.get("total_cost_usd") or 0
        print(f"{s['model_label']:<35} {s['benchmark']:<25} {acc:>6}  {tokens:>8,}  {cost:>8.4f}")
    print("=" * 80)

    total_cost = sum(s.get("total_cost_usd") or 0 for s in summaries)
    total_tokens = sum(s.get("total_tokens") or 0 for s in summaries)
    print(f"\nTotal tokens : {total_tokens:,}")
    print(f"Total cost   : ${total_cost:.4f}")

    # Generate master CSVs from all JSONL results
    print("\nGenerating CSVs…")
    csv_paths = generate_csvs(config.EXPERIMENT_ID)
    for p in csv_paths:
        print(f"  Written: {p}")


if __name__ == "__main__":
    main()
