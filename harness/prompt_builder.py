"""
prompt_builder.py — Construct zero-shot prompts with deterministic option shuffling.

Shuffling rationale
───────────────────
MCQ benchmarks exhibit strong position bias: models prefer options in early
positions (A, B) regardless of content. To avoid conflating model accuracy with
positional preference we shuffle the options for each (question, model) pair
using a seed derived from the question_id and model_label. This makes shuffles:
  - Deterministic: the same run always produces the same shuffle.
  - Model-specific: different models see different orderings, so any position
    effect averages out across the model comparison rather than benefiting a
    particular option across all models.
  - Recoverable: we store the permutation so we can always map the model's
    selected letter back to the original option index.

ShuffledQuestion fields
───────────────────────
    original_options   list[str]   options in their original order
    shuffled_options   list[str]   options after shuffling
    permutation        list[int]   permutation[i] = original index of position i
                                   e.g. [2, 0, 1] means:
                                     new A = original index 2
                                     new B = original index 0
                                     new C = original index 1
    correct_letter     str         letter that corresponds to the correct answer
                                   after shuffling
    prompt_messages    list[dict]  OpenAI-style messages list ready for the API
"""

import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional

from data_loader import option_letter

# ---------------------------------------------------------------------------
# System prompt — shared across all zero-shot queries
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert evaluator. Your task is to answer multiple-choice questions "
    "accurately. Always respond with a single JSON object and nothing else. "
    "Do not add markdown fences, commentary, or extra text outside the JSON."
)

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------
_USER_TEMPLATE = """\
Answer the following multiple-choice question.

Question:
{question}

Options:
{options_block}

Respond with ONLY this JSON object (no markdown, no extra text):
{{
  "selected_answer": "<letter>",
  "explanation": "<one to two sentence justification for your choice>",
  "confidence": <integer 0-100 reflecting how confident you are>
}}"""


@dataclass
class ShuffledQuestion:
    benchmark: str
    question_id: str
    original_options: list[str]
    shuffled_options: list[str]
    permutation: list[int]          # permutation[new_idx] = original_idx
    correct_letter: str             # correct answer letter in shuffled order
    prompt_messages: list[dict] = field(default_factory=list)


def _make_seed(question_id: str, model_label: str, global_seed: int) -> int:
    """Derive a reproducible integer seed from question + model identifiers."""
    raw = f"{global_seed}:{question_id}:{model_label}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return int(digest[:16], 16)  # first 64 bits → int


def build_shuffled_question(
    item: dict,
    model_label: str,
    shuffle_seed: int,
) -> ShuffledQuestion:
    """
    Shuffle answer options for a given (item, model) pair and build the prompt.

    Parameters
    ----------
    item         : BenchmarkItem dict from data_loader
    model_label  : model label string from config (used to diversify shuffles)
    shuffle_seed : global seed from config.SHUFFLE_SEED
    """
    options = item["options"]
    n = len(options)

    rng = random.Random(_make_seed(item["id"], model_label, shuffle_seed))
    permutation = list(range(n))
    rng.shuffle(permutation)

    shuffled_options = [options[permutation[i]] for i in range(n)]

    # Find which new position holds the originally correct answer
    correct_new_idx = permutation.index(item["target"])
    correct_letter = option_letter(correct_new_idx)

    # Build the options block for the prompt
    options_block = "\n".join(
        f"{option_letter(i)}. {shuffled_options[i]}" for i in range(n)
    )

    user_content = _USER_TEMPLATE.format(
        question=item["question"],
        options_block=options_block,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return ShuffledQuestion(
        benchmark=item["benchmark"],
        question_id=item["id"],
        original_options=options,
        shuffled_options=shuffled_options,
        permutation=permutation,
        correct_letter=correct_letter,
        prompt_messages=messages,
    )


def resolve_selected_original_index(
    sq: ShuffledQuestion, selected_letter: Optional[str]
) -> Optional[int]:
    """
    Map the model's selected letter back to the original option index.
    Returns None if the letter is invalid or out of range.
    """
    if selected_letter is None:
        return None
    # Must be exactly one A–Z character. str.index on the alphabet is a substring
    # search, so "" would resolve to 0 and silently score as "A"; likewise "AB"→0,
    # "CD"→2. Anything that is not a single letter is unscored.
    letter = selected_letter.strip().upper()
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        return None
    new_idx = ord(letter) - ord("A")
    if new_idx >= len(sq.permutation):
        return None
    return sq.permutation[new_idx]
