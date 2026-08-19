"""
output_parser.py — Parse model responses into structured result records.

Strategy (in order):
  1. Try to parse the whole response as JSON.
  2. Try to extract a JSON object from within the text (handles models that
     wrap JSON in markdown fences or add preamble).
  3. Fall back to regex extraction of each field individually.
  4. If all else fails, mark the record as parse_failed=True.

Output record schema
────────────────────
{
  # Identification
  "experiment_id":           str,
  "benchmark":               str,
  "question_id":             str,
  "kind":                    str | None,

  # Model
  "model_id":                str,
  "model_label":             str,
  "reasoning_enabled":       bool,

  # Shuffling
  "permutation":             list[int],
  "correct_letter":          str,

  # Model outputs
  "selected_letter":         str | None,
  "selected_original_index": int | None,
  "is_correct":              bool | None,
  "explanation":             str | None,
  "confidence":              int | None,    # 0–100
  "raw_content":             str,
  "parse_failed":            bool,

  # Logprobs (None when model doesn't support them)
  "logprob_selected_token":  str | None,
  "logprob_selected_value":  float | None,
  "logprob_top_tokens":      list[dict] | None,

  # Reasoning / chain-of-thought
  "reasoning_content":       str | None,
  "cot_reasoning":           str | None,   # explicitly prompted CoT (cot_v* experiments)

  # Usage metrics
  "prompt_tokens":           int,
  "completion_tokens":       int,
  "total_tokens":            int,
  "reasoning_tokens":        int | None,   # usage.completion_tokens_details.reasoning_tokens
  "cost_usd":                float | None,
  "latency_ms":              float,
  "finish_reason":           str | None,

  # Error
  "error":                   str | None,
}
"""

import json
import logging
import re
from typing import Any, Optional

from api_client import ApiResponse
from prompt_builder import ShuffledQuestion, resolve_selected_original_index

logger = logging.getLogger(__name__)

_LETTER_RE = re.compile(r'\b([A-Z])\b')
_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*(\d+)')
_EXPLANATION_RE = re.compile(r'"explanation"\s*:\s*"([^"]*)"', re.DOTALL)
_ANSWER_RE = re.compile(r'"selected_answer"\s*:\s*"([A-Za-z])"')

# Matches ```json ... ``` or ``` ... ```
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _sanitize_json_strings(text: str) -> str:
    """
    Escape literal control characters (newline, carriage return, tab) that appear
    inside JSON string values but were not escaped by the model.

    Some models (e.g. ministral-8b) output well-structured JSON but embed literal
    newlines directly in string values, which json.loads rejects. This function
    walks the text character-by-character and replaces unescaped control characters
    inside strings with their JSON escape sequences.
    """
    result = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == "\\":
            result.append(ch)
            escaped = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
    return "".join(result)


def _try_json(text: str) -> Optional[dict]:
    text = text.strip()
    # First attempt: strict parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Second attempt: sanitize unescaped control chars inside strings, then retry
    try:
        obj = json.loads(_sanitize_json_strings(text))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _try_extract_json(text: str) -> Optional[dict]:
    # Strip markdown fences first
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        result = _try_json(candidate)
        if result is not None:
            return result

    # Find the first '{' and last '}' and try that substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        result = _try_json(text[start : end + 1])
        if result is not None:
            return result

    return _try_last_json_object(text)


def _try_last_json_object(text: str) -> Optional[dict]:
    """Scan backwards for the last self-contained JSON object carrying an answer.

    Added 2026-08-06. Some providers stopped splitting a model's thinking into the
    response's separate `reasoning` field and now return it inline in `content`,
    followed by the answer JSON. The first-brace/last-brace span above then fails
    whenever the prose contains a stray '{' — which it does in ~98% of such rows.

    Deliberately conservative: only objects that actually contain "selected_answer"
    are accepted, so this cannot promote an unrelated dict found in the reasoning.
    Re-parsing the full May 2026 corpus with this fallback moved 356 of 495,997
    rows (0.07%); it recovered 57/57 affected qwen3-5-flash_reasoning rows.
    """
    for candidate in (text, _sanitize_json_strings(text)):
        decoder = json.JSONDecoder()
        idx = candidate.rfind("{")
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(candidate, idx)
            except ValueError:
                obj = None
            if isinstance(obj, dict) and "selected_answer" in obj:
                return obj
            idx = candidate.rfind("{", 0, idx)
    return None


def _regex_fallback(text: str) -> dict:
    answer_match = _ANSWER_RE.search(text)
    selected_letter = answer_match.group(1).upper() if answer_match else None

    # If no explicit key, look for a lone capital letter near "answer"
    if selected_letter is None:
        letters = _LETTER_RE.findall(text)
        if letters:
            selected_letter = letters[0]

    conf_match = _CONFIDENCE_RE.search(text)
    confidence = int(conf_match.group(1)) if conf_match else None

    expl_match = _EXPLANATION_RE.search(text)
    explanation = expl_match.group(1) if expl_match else None

    return {
        "selected_answer": selected_letter,
        "confidence": confidence,
        "explanation": explanation,
    }


def _extract_fields(text: str) -> tuple[dict, bool]:
    """
    Returns (fields_dict, parse_failed).
    fields_dict always has keys: selected_answer, explanation, confidence.
    """
    parsed = _try_json(text)
    if parsed is None:
        parsed = _try_extract_json(text)
    if parsed is None:
        parsed = _regex_fallback(text)
        return parsed, True
    return parsed, False


def _clamp_confidence(raw: Any) -> Optional[int]:
    try:
        v = int(raw)
        return max(0, min(100, v))
    except (TypeError, ValueError):
        return None


def build_result_record(
    item: dict,
    sq: ShuffledQuestion,
    api_resp: ApiResponse,
    model_cfg: dict,
    experiment_id: str,
) -> dict:
    """
    Combine item metadata, shuffle info, API response, and parsed fields
    into a single flat record ready for JSONL output.
    """
    # If the API call itself failed, return an error record
    if api_resp.error:
        return {
            "experiment_id": experiment_id,
            "benchmark": item["benchmark"],
            "question_id": item["id"],
            "kind": item.get("kind"),
            "model_id": api_resp.model_id,
            "model_label": api_resp.model_label,
            "reasoning_enabled": bool(model_cfg.get("reasoning_config")),
            "permutation": sq.permutation,
            "correct_letter": sq.correct_letter,
            "selected_letter": None,
            "selected_original_index": None,
            "is_correct": None,
            "explanation": None,
            "confidence": None,
            "raw_content": "",
            "parse_failed": True,
            "logprob_selected_token": None,
            "logprob_selected_value": None,
            "logprob_top_tokens": None,
            "reasoning_content": None,
            "cot_reasoning": None,
            "prompt_tokens": api_resp.prompt_tokens,
            "completion_tokens": api_resp.completion_tokens,
            "total_tokens": api_resp.total_tokens,
            "reasoning_tokens": api_resp.reasoning_tokens,
            "cost_usd": api_resp.cost_usd,
            "latency_ms": api_resp.latency_ms,
            "finish_reason": api_resp.finish_reason,
            "error": api_resp.error,
        }

    fields, parse_failed = _extract_fields(api_resp.raw_content)

    raw_letter = fields.get("selected_answer")
    if isinstance(raw_letter, str):
        stripped = raw_letter.strip()
        # Handle "D. Legal Compliance", "D) something", "D: something" → "D"
        m = re.match(r'^([A-Za-z])[.):\s]', stripped)
        selected_letter = (m.group(1) if m else stripped).upper()
    else:
        selected_letter = None
    selected_original_index = resolve_selected_original_index(sq, selected_letter)

    is_correct: Optional[bool] = None
    if selected_original_index is not None:
        is_correct = selected_original_index == item["target"]

    explanation = fields.get("explanation")
    if isinstance(explanation, str):
        explanation = explanation.strip() or None

    cot_reasoning = fields.get("reasoning")
    if isinstance(cot_reasoning, str):
        cot_reasoning = cot_reasoning.strip() or None

    confidence = _clamp_confidence(fields.get("confidence"))

    # Logprobs
    lp = api_resp.logprobs
    logprob_selected_token = lp.selected_token if lp else None
    logprob_selected_value = lp.selected_logprob if lp else None
    logprob_top_tokens = lp.top_tokens if lp else None

    if parse_failed:
        logger.debug(
            "Parse fallback for %s / %s / %s",
            api_resp.model_label,
            item["benchmark"],
            item["id"],
        )

    return {
        "experiment_id": experiment_id,
        "benchmark": item["benchmark"],
        "question_id": item["id"],
        "kind": item.get("kind"),
        "model_id": api_resp.model_id,
        "model_label": api_resp.model_label,
        "reasoning_enabled": bool(model_cfg.get("reasoning_config")),
        "permutation": sq.permutation,
        "correct_letter": sq.correct_letter,
        "selected_letter": selected_letter,
        "selected_original_index": selected_original_index,
        "is_correct": is_correct,
        "explanation": explanation,
        "confidence": confidence,
        "raw_content": api_resp.raw_content,
        "parse_failed": parse_failed,
        "logprob_selected_token": logprob_selected_token,
        "logprob_selected_value": logprob_selected_value,
        "logprob_top_tokens": logprob_top_tokens,
        "reasoning_content": api_resp.reasoning_content,
        "cot_reasoning": cot_reasoning,
        "prompt_tokens": api_resp.prompt_tokens,
        "completion_tokens": api_resp.completion_tokens,
        "total_tokens": api_resp.total_tokens,
        "reasoning_tokens": api_resp.reasoning_tokens,
        "cost_usd": api_resp.cost_usd,
        "latency_ms": api_resp.latency_ms,
        "finish_reason": api_resp.finish_reason,
        "error": None,
    }
