"""
Experiment configuration: models, API settings, and run parameters.

Each entry in MODELS defines one "run". Models that should be tested with and
without reasoning are listed twice — once with reasoning_config=None and once
with the appropriate reasoning_config dict for that provider.

reasoning_config keys are passed directly into the API request body so you can
accommodate provider-specific formats:
  - OpenAI o-series:       {"reasoning_effort": "high"}
  - Anthropic (OR):        {"thinking": {"type": "enabled", "budget_tokens": 8000}}
  - Google (OR):           {"thinking": {"thinkingBudget": 8000}}
  - OpenRouter generic:    {"reasoning": {"enabled": True}}   ← most other providers
  - Always-on reasoning:   list once with reasoning_config=None
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent

# Which benchmark release to run against:
#   benchmarks/curated  post-QC set, 9,594 items  (default — matches the paper)
#   benchmarks/pre_qc   original set, 10,020 items
# Override with the DATA_DIR env var or run_experiment.py --data-dir.
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "benchmarks" / "curated"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT_DIR / "results"))

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# HTTP referrer / title sent with every request (OpenRouter guidelines)
HTTP_REFERER = os.getenv("HTTP_REFERER", "https://github.com/prj-bench")
APP_TITLE = os.getenv("APP_TITLE", "Prj-BENCH")

# ---------------------------------------------------------------------------
# Concurrency / rate-limiting
# Tune MAX_CONCURRENT per your OpenRouter tier.
# A safe starting point is 20–30; raise if you have higher limits.
# ---------------------------------------------------------------------------
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "25"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
EXPERIMENT_ID = os.getenv("EXPERIMENT_ID", "zeroshot_v1")

# Seed used together with question_id + model_id to shuffle answer options.
# Change only between distinct experiment series — same seed = same shuffles.
SHUFFLE_SEED = int(os.getenv("SHUFFLE_SEED", "42"))

# Max completion tokens for the model response.
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "512"))

# Request log-probabilities when supported (top-5 tokens at first position).
REQUEST_LOGPROBS = True
TOP_LOGPROBS = 5

# ---------------------------------------------------------------------------
# Model registry
#
# Fields:
#   id              OpenRouter model identifier
#   label           Human-readable name used in output filenames / logs
#   reasoning_config  dict of extra body params to enable reasoning, or None
#   extra_params    any other model-specific params (temperature, etc.)
# ---------------------------------------------------------------------------
# concurrency: max simultaneous in-flight requests for this model.
# Reasoning/thinking models run slower per request but can sustain more
# concurrent calls before hitting provider rate limits — so they get a
# larger pool. Non-reasoning models are fast but more rate-limit-sensitive,
# so they get a smaller pool. These match the reference runner's values.
_STD = 20   # standard (non-reasoning) models
_RES = 30   # reasoning / thinking models

# Shorthand for the OpenRouter generic reasoning toggle used by most
# non-Anthropic, non-OpenAI providers (ByteDance, Qwen, GLM, Aion, etc.)
_OR_REASON_ON  = {"reasoning": {"enabled": True}}
_OR_REASON_OFF = {"reasoning": {"enabled": False}}

MODELS = [
    # ── OpenAI ───────────────────────────────────────────────────────────────
    {
        "id": "openai/gpt-4o",
        "label": "gpt-4o",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    {
        "id": "openai/gpt-5",
        "label": "gpt-5",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    {
        "id": "openai/o3-mini",
        "label": "o3-mini",
        "reasoning_config": None,   # o3-mini always reasons; no toggle needed
        "extra_params": {},
        "concurrency": _RES,
    },
    # ── Anthropic ────────────────────────────────────────────────────────────
    {
        # medcalc + hle require long step-by-step calculations before JSON
        "id": "anthropic/claude-sonnet-4-6",
        "label": "claude-sonnet-4-6",
        "reasoning_config": None,
        "extra_params": {"temperature": 0, "max_tokens": 2048},
        "concurrency": _STD,
    },
    {
        # medcalc requires long step-by-step calculations before JSON
        "id": "anthropic/claude-opus-4-6",
        "label": "claude-opus-4-6",
        "reasoning_config": None,
        "extra_params": {"temperature": 0, "max_tokens": 2048},
        "concurrency": _STD,
    },
    # ── Google ───────────────────────────────────────────────────────────────
    {
        "id": "google/gemini-3.1-flash-lite-preview",
        "label": "gemini-3-1-flash-lite",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    {
        # NOTE: gemini-3.1-pro-preview rejects reasoning=off ("Reasoning is mandatory")
        # so this model is run once, reasoning-on only.
        # thinkingBudget=8000 + answer headroom → needs >8192 max_tokens
        "id": "google/gemini-3.1-pro-preview",
        "label": "gemini-3-1-pro_thinking",
        "reasoning_config": {"thinking": {"thinkingBudget": 8000}},
        "extra_params": {"max_tokens": 16000},
        "concurrency": _RES,
    },
    {
        "id": "google/gemma-3-12b-it",
        "label": "gemma-3-12b",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    # ── Meta / Llama ──────────────────────────────────────────────────────────
    {
        "id": "meta-llama/llama-3-8b-instruct",
        "label": "llama-3-8b",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    {
        "id": "meta-llama/llama-3.1-8b-instruct",
        "label": "llama-3-1-8b",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    # ── Mistral ───────────────────────────────────────────────────────────────
    {
        # medcalc requires long step-by-step calculations before JSON
        "id": "mistralai/ministral-14b-2512",
        "label": "ministral-14b",
        "reasoning_config": None,
        "extra_params": {"temperature": 0, "max_tokens": 2048},
        "concurrency": _STD,
    },
    {
        # medcalc requires long step-by-step calculations before JSON
        "id": "mistralai/ministral-8b-2512",
        "label": "ministral-8b",
        "reasoning_config": None,
        "extra_params": {"temperature": 0, "max_tokens": 2048},
        "concurrency": _STD,
    },
    {
        # medcalc requires long step-by-step calculations before JSON
        "id": "mistralai/ministral-3b-2512",
        "label": "ministral-3b",
        "reasoning_config": None,
        "extra_params": {"temperature": 0, "max_tokens": 2048},
        "concurrency": _STD,
    },
    {
        "id": "mistralai/mistral-nemo",
        "label": "mistral-nemo",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    # ── NVIDIA ────────────────────────────────────────────────────────────────
    {
        # nemotron outputs chain-of-thought in content before JSON; needs more tokens
        "id": "nvidia/nemotron-3-super-120b-a12b",
        "label": "nemotron-120b",
        "reasoning_config": None,
        "extra_params": {"temperature": 0, "max_tokens": 4096},
        "concurrency": _STD,
    },
    # ── Liquid ────────────────────────────────────────────────────────────────
    {
        "id": "liquid/lfm2-8b-a1b",
        "label": "lfm2-8b",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    # ── Amazon ────────────────────────────────────────────────────────────────
    {
        "id": "amazon/nova-micro-v1",
        "label": "nova-micro",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    # ── ByteDance Seed ────────────────────────────────────────────────────────
    {
        # medcalc requires long step-by-step calculations before JSON
        "id": "bytedance-seed/seed-2.0-mini",
        "label": "seed-2-mini_base",
        "reasoning_config": _OR_REASON_OFF,
        "extra_params": {"temperature": 0, "max_tokens": 2048},
        "concurrency": _STD,
    },
    {
        # max_tokens set 2026-08-06 to restore the May 2026 run condition, NOT to raise
        # the budget. In May this endpoint ignored max_tokens for reasoning entirely: it
        # spent a p95 of 6,151 (max 27,997) reasoning tokens against a 512-token request
        # and truncated 0.0% of 9,838 rows. OpenRouter now charges reasoning against
        # max_tokens, so the same request truncates 66% of rows with zero visible output.
        # Same model snapshot, same parameters — only the platform's accounting changed.
        # 32,000 is above every reasoning length observed in May, so the cap does not bind
        # and the May condition is reproduced. This is a ceiling, not a spend: billing
        # follows tokens actually emitted, which is unchanged.
        "id": "bytedance-seed/seed-2.0-mini",
        "label": "seed-2-mini_reasoning",
        "reasoning_config": _OR_REASON_ON,
        "extra_params": {"max_tokens": 32000},
        "concurrency": _RES,
    },
    # ── Qwen ──────────────────────────────────────────────────────────────────
    {
        # medcalc requires long step-by-step calculations before JSON
        "id": "qwen/qwen3.5-flash-02-23",
        "label": "qwen3-5-flash_base",
        "reasoning_config": _OR_REASON_OFF,
        "extra_params": {"temperature": 0, "max_tokens": 2048},
        "concurrency": _STD,
    },
    {
        # max_tokens set 2026-08-06 to restore the May 2026 run condition — see the
        # seed-2-mini_reasoning entry above for the full rationale. In May this endpoint
        # spent a p95 of 6,780 (max 20,333) reasoning tokens against a 512-token request
        # and truncated 0.0% of 10,036 zero-shot rows; its CoT visible output ran to a
        # median of 2,527 tokens against a 1,024-token request, also without truncation.
        # max_tokens plainly did not bind. The same request now truncates 99.8% of rows.
        # 32,000 is non-binding against every May observation and restores that condition.
        "id": "qwen/qwen3.5-flash-02-23",
        "label": "qwen3-5-flash_reasoning",
        "reasoning_config": _OR_REASON_ON,
        "extra_params": {"max_tokens": 32000},
        "concurrency": _RES,
    },
    # ── Aion Labs ─────────────────────────────────────────────────────────────
    {
        "id": "aion-labs/aion-2.0",
        "label": "aion-2-0_base",
        "reasoning_config": _OR_REASON_OFF,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    {
        # concurrency dropped from 30 — provider rate-limits aggressively at higher values
        #
        # max_tokens 2600, set 2026-08-06. This model needs a DIFFERENT value from the other
        # reasoning models because its two arms were constrained differently in May.
        #
        #   ZERO-SHOT: May truncated 0.0% of 10,040 rows while reasoning ran to a p95 of
        #   1,717 (max 5,911) against a 512-token request — the cap was not binding, because
        #   reasoning was not charged against it. Reasoning IS charged now, so {} would bind
        #   (5.5% truncation, 6.4% parse failure vs May's 0.0%/0.3%). A large non-binding cap
        #   reproduces May here.
        #
        #   CoT: May truncated 8.2%, median visible output 199 tokens. The 1,024 cap DID
        #   bind, and the resulting 33.3% parse-failure rate is a real property of the May
        #   experiment, not an artifact. A large cap therefore does NOT restore May — it
        #   removes a constraint that genuinely existed. Run at 32,000 the five re-run
        #   benchmarks fell to 0.8% parse failure against 33.3% on the nineteen unchanged
        #   ones; aion was the only model in either arm whose five-vs-nineteen difference ran
        #   in that direction (every other model parse-fails MORE on those five, which are
        #   harder), which is what identified the cap as the cause.
        #
        # The two arms therefore need DIFFERENT values, and a single literal cannot serve
        # both — 2600 would bind on zero-shot (May reasoning max 5,911) and 32000 removes
        # the CoT constraint. Hence the env override:
        #
        #   zero-shot : 32000  (default) — non-binding, reproduces May
        #   CoT       : 2600           — 1024 May visible budget + 1517 May max reasoning,
        #                                so visible output gets the same room it had in May
        #                                while reasoning is paid for separately
        #
        # Set it explicitly when running the CoT arm:
        #   AION_MAX_TOKENS=2600 python3 run_cot_experiment.py --models aion-2-0_reasoning ...
        #
        # See qc/benchmark_qc/_results_integrity/NOTES.md before changing either value.
        "id": "aion-labs/aion-2.0",
        "label": "aion-2-0_reasoning",
        "reasoning_config": _OR_REASON_ON,
        "extra_params": {"max_tokens": int(os.getenv("AION_MAX_TOKENS", "32000"))},
        "concurrency": 12,
    },
    # ── TheDrummer ────────────────────────────────────────────────────────────
    {
        "id": "thedrummer/cydonia-24b-v4.1",
        "label": "cydonia-24b",
        "reasoning_config": None,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    # ── Z-AI GLM ──────────────────────────────────────────────────────────────
    {
        "id": "z-ai/glm-4.7-flash",
        "label": "glm-4-7-flash_base",
        "reasoning_config": _OR_REASON_OFF,
        "extra_params": {"temperature": 0},
        "concurrency": _STD,
    },
    {
        # glm reasoning outputs thinking before JSON; needs more tokens.
        #
        # max_tokens is env-overridable, default 4096, as of 2026-08-07.
        #
        # 4096 is the value this model has carried since May and it BINDS: unlike
        # seed/qwen/aion — which truncated 0.0% in May because reasoning was not
        # charged against max_tokens — glm truncated at 4096 in May too
        # (medcalc_metacognition 54.2%, truthfulqa_ethics 15.9% on 05-19). So there
        # is no non-binding May condition to restore here, and a partial retry
        # would mix two conditions inside one experiment.
        #
        # It is also asymmetric: every other reasoning config is non-binding
        # (seed/qwen/aion 32000, gemini 16000, all 0.0% truncated) while glm
        # truncates 10.9% of zero-shot rows with zero visible output. Because the
        # dropped rows are the ones needing the longest reasoning, they are
        # systematically harder — on medcalc_verified the 153 truncated items are
        # 7.3 points harder for the other 21 models — so dropping them from the
        # denominator biases glm's accuracy UP.
        #
        # The fix is therefore a FULL re-run of the affected experiment at 32000,
        # not a retry of the truncated rows. Scoped per-run via the env var so
        # zeroshot_v1 and cot_v1 can be brought to the non-binding condition
        # without re-running mdagents / medrag / d1-d4 / open_ended, which stay at
        # 4096 by default:
        #
        #   GLM_MAX_TOKENS=32000 python run_experiment.py --models glm-4-7-flash_reasoning --force
        #
        # Anything left at 4096 must be reported as such — see
        # curated/results/summary_*.csv, column n_unscored.
        "id": "z-ai/glm-4.7-flash",
        "label": "glm-4-7-flash_reasoning",
        "reasoning_config": _OR_REASON_ON,
        "extra_params": {"max_tokens": int(os.getenv("GLM_MAX_TOKENS", "4096"))},
        "concurrency": _RES,
    },
]
