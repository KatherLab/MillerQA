# Prj-BENCH

Benchmark data and zero-shot inference harness accompanying the paper.

This repository contains two things:

1. **The benchmark suite** — 24 multiple-choice benchmarks, released in both
   their original (pre-QC) and curated (post-QC) form.
2. **The zero-shot harness** — the code used to run the models over the suite
   via [OpenRouter](https://openrouter.ai).

```
benchmarks/
  pre_qc/            24 files, 10,020 items — the original suite, before QC
  curated/           24 files,  9,594 items — the post-QC suite used in the paper
  _manifest.csv      per-benchmark item counts, pre-QC vs curated
  _dropped_items.csv every item removed by QC, with the reason
harness/             zero-shot inference code
requirements.txt
.env.example
```

## Benchmark format

Every file is a JSON array of items with a common schema:

```json
{
  "id": "professional_medicine/test/131/en",
  "question": "A 62-year-old man ...",
  "options": ["Begin ...", "Order ...", "Discharge ...", "Consult ..."],
  "target": 2,
  "kind": "professional_medicine"
}
```

`target` is the **0-based index into `options`** of the correct answer, in the
order given in the file. `kind` is an optional subcategory label and is absent
in some benchmarks. Question text and options are kept fully separable, which is
what makes the choice-only baseline possible.

### pre-QC vs curated

`pre_qc/` is the suite as originally assembled: 10,020 items. `curated/` is the
same suite after the quality-control pass described in the paper: 9,594 items,
with defective items excluded and duplicates removed. Three benchmarks were
retired and replaced by corrected versions during QC, and both directories carry
only the replacements (`medcalc_verified`, `triage_ethics_v2`,
`truthfulqa_mc1`), so the two directories hold the same 24 benchmark names and
differ only in which items survive.

`_manifest.csv` gives the per-benchmark counts on both sides;
`_dropped_items.csv` lists all 447 removed items with the QC check that caught
each one and the decision taken. **Results reported in the paper are computed on
the curated set**, which is also the harness default.

## Running the zero-shot harness

### 1. Install

Python 3.10+ (the code uses `X | Y` type syntax).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Then edit `.env` and set `OPENROUTER_API_KEY`. Everything else is optional; the
defaults shown in that file are the ones used for the paper — notably
`SHUFFLE_SEED=42`, which controls answer-option shuffling and must be kept
unchanged to reproduce the published runs.

### 3. Run

All commands are run from inside `harness/`.

```bash
cd harness

# Print the execution plan without spending anything — always start here
python run_experiment.py --dry-run

# One model, one benchmark
python run_experiment.py --models gpt-4o --benchmarks fairmedqa

# The full paper run: all 26 model configurations × all 24 benchmarks
python run_experiment.py

# Run against the original pre-QC release instead
python run_experiment.py --data-dir ../benchmarks/pre_qc --experiment-id zeroshot_preqc
```

The full run is **26 model configurations × 9,594 items ≈ 249k API calls**. Cost
and time are substantial. Use `--dry-run` first and scope with `--models` /
`--benchmarks`.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--models LABEL ...` | Restrict to model labels from `harness/config.py` (default: all 26) |
| `--benchmarks NAME ...` | Restrict to benchmark file stems (default: all 24) |
| `--data-dir PATH` | Benchmark directory to load (default: `benchmarks/curated`) |
| `--experiment-id ID` | Output folder name under `results/` (default: `zeroshot_v1`) |
| `--concurrency N` | Max in-flight API calls (default: 25) |
| `--retry-failed` | Re-run only items that errored or failed to parse |
| `--force` | Ignore resume state and re-run everything |
| `--dry-run` | Print the plan and exit without calling the API |
| `-v` | Debug logging |

Runs are **resumable**: results are appended per model/benchmark, and re-running
the same command picks up where it left off rather than repeating completed
items. Use `--force` only when you intend to discard prior results.

### 4. Output

```
results/<experiment_id>/
  <model_label>/<benchmark>.jsonl   one JSON record per item
  csv/master.csv                    all records flattened
  csv/by_model/<model_label>.csv
  csv/by_benchmark/<benchmark>.csv
  run.log
```

Each record carries the prompt condition, the model's answer, and accounting:
`permutation` and `correct_letter` (the shuffle actually shown to this model),
`selected_letter`, `selected_original_index`, `is_correct`, `confidence`,
`explanation`, `raw_content`, token counts, `cost_usd`, `latency_ms`, and
`error`. Because options are shuffled per (item, model), correctness is scored
by mapping the selected letter back through `permutation` to the original option
index and comparing against `target` — never by comparing letters directly.

`is_correct` is `null` for items with no parseable answer (a refusal, a
truncation, a malformed response). These are unscored, not wrong; the paper
drops them from the denominator and reports the counts separately.

## What is in `harness/`

| File | Role |
| --- | --- |
| `run_experiment.py` | CLI entry point |
| `config.py` | Paths, API settings, and the 26-entry model registry |
| `data_loader.py` | Loads and normalises the benchmark JSON |
| `prompt_builder.py` | Deterministic per-(item, model) option shuffling and prompt assembly |
| `api_client.py` | OpenRouter calls, retries, reasoning-config handling |
| `batch_runner.py` | Concurrency, resume, per-model orchestration |
| `output_parser.py` | Response parsing into the flat record schema |
| `make_csv.py` | JSONL → CSV |

The model registry in `config.py` is the exact set of 26 model configurations
evaluated in the paper. Models tested both with and without reasoning appear
twice, with different `label`s and a `reasoning_config` that is passed through to
the provider. Note that some entries carry model-specific `max_tokens`; the
comments in that file record why, including the reasoning-truncation caveat that
applies to `glm-4-7-flash_reasoning`.

### Note on the answer resolver

The answer-letter resolver in this release differs from the code used for the
original runs in one respect. The original used `str.index` against the alphabet
to map a letter back to an index, which is a *substring* search: an empty answer
matched at position 0 and was silently scored as "A", as were multi-letter
outputs like "AB". This affected 29 of ~495k records, 10 of which were scored
correct off a non-answer — too few to move any published figure. The version here
accepts only a single A–Z character and treats anything else as unscored, which
matches how the paper's results were rescored.

## Citation

Please cite the accompanying paper. Individual benchmarks are derived from
previously published sources and remain subject to their original licences; see
the paper for the full provenance of each of the 24 sets.
