"""
api_client.py — Async OpenRouter client with retry, metrics, and logprob capture.

Design
──────
- Uses the `openai` async client pointed at OpenRouter's base URL.
- Wraps every call in tenacity retry logic (exponential back-off) so transient
  rate-limit / server errors don't abort the whole run.
- Returns a raw ApiResponse dataclass that captures everything we care about
  before the caller interprets it (tokens, cost, latency, logprobs).
- Logprobs: requested when config.REQUEST_LOGPROBS is True. Not all models
  return them; we gracefully store None when absent.
- Cost: OpenRouter returns usage.cost in the response when available; we
  surface it directly. Fallback to None if the field is missing.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from openai import (
    AsyncOpenAI,
    RateLimitError,
    APIStatusError,
    APIConnectionError,
    APITimeoutError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

import config

logger = logging.getLogger(__name__)


# Substrings that mark a transient network failure surfacing as some other
# exception type (e.g. a mid-stream drop that bypasses the openai SDK's typed
# exceptions). Kept narrow so we don't accidentally retry non-transient errors —
# notably NOT matching the SSE JSON-parse error ("Expecting value"), which must
# flow to the stream-fallback in _dispatch_call instead of being retried here.
_TRANSIENT_MSG_HINTS = (
    "connection lost",
    "connection reset",
    "connection aborted",
    "connection error",
    "network",
    "server disconnected",
    "peer closed",
    "reset by peer",
    "incomplete read",
    "incomplete chunked read",
    "broken pipe",
    "eof occurred",
)


class EmptyResponseError(Exception):
    """A 200 response that carries no usable choice.

    Some providers return a well-formed envelope with `choices` null or empty rather than
    an error status — observed on Featherless under concurrency, 2026-08-07. Before this
    was handled, `response.choices[0]` raised a bare TypeError ("'NoneType' object is not
    subscriptable"), which the generic handler turned into a permanent error row, so the
    item was lost instead of retried.
    """


def _is_retryable(exc: BaseException) -> bool:
    """
    Return True for exceptions that warrant a retry.
    Covers:
      - RateLimitError / APIConnectionError / APITimeoutError (openai SDK typed)
      - APIStatusError with status 429 (how OpenRouter surfaces rate limits)
      - APIStatusError with status 5xx (transient server errors)
      - EmptyResponseError (200 with no choices — transient, provider-side)
      - httpx transport / stream errors (can surface mid-stream and bypass the
        openai SDK's typed exceptions — e.g. "Network connection lost")
      - any other exception whose message looks like a transient network drop
    """
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError,
                        EmptyResponseError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    if isinstance(exc, (httpx.TransportError, httpx.StreamError)):
        return True
    msg = str(exc).lower()
    return any(hint in msg for hint in _TRANSIENT_MSG_HINTS)


async def _wait_for_retry_after(exc: BaseException) -> None:
    """If the response included a Retry-After header, honour it."""
    if isinstance(exc, APIStatusError):
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            try:
                wait = float(retry_after)
                logger.warning("Rate limited — honouring Retry-After: %.1fs", wait)
                await asyncio.sleep(wait)
            except ValueError:
                pass


@dataclass
class LogprobInfo:
    """Top-N token log-probabilities at the first completion token position."""
    top_tokens: list[dict]   # [{"token": "A", "logprob": -0.12}, ...]
    selected_token: Optional[str] = None
    selected_logprob: Optional[float] = None


@dataclass
class ApiResponse:
    model_id: str
    model_label: str
    raw_content: str                    # full text returned by model
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: Optional[int] = None  # from usage.completion_tokens_details.reasoning_tokens
    cost_usd: Optional[float] = None    # from usage.cost if provided
    latency_ms: float = 0.0
    logprobs: Optional[LogprobInfo] = None
    error: Optional[str] = None         # set if the call failed after retries
    finish_reason: Optional[str] = None
    reasoning_content: Optional[str] = None  # chain-of-thought when available
    raw_response: Optional[dict] = None      # full provider response (incl. extras)


def _build_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": config.HTTP_REFERER,
            "X-Title": config.APP_TITLE,
        },
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )


# Module-level client — created once, reused across all coroutines
_CLIENT: Optional[AsyncOpenAI] = None


def get_client() -> AsyncOpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _build_client()
    return _CLIENT


def _extract_logprobs(choice) -> Optional[LogprobInfo]:
    """Pull logprob data out of a chat completion choice object."""
    try:
        lp = choice.logprobs
        if lp is None:
            return None
        content = lp.content  # list of ChatCompletionTokenLogprob
        if not content:
            return None

        first = content[0]
        top_tokens = []
        if first.top_logprobs:
            for entry in first.top_logprobs:
                top_tokens.append({"token": entry.token, "logprob": entry.logprob})

        return LogprobInfo(
            top_tokens=top_tokens,
            selected_token=first.token,
            selected_logprob=first.logprob,
        )
    except Exception:
        return None


def _extract_reasoning(choice) -> Optional[str]:
    """Extract chain-of-thought / reasoning content when present.

    Different providers use different field names on `message`:
      - OpenRouter unified schema: `reasoning`               ← most common
      - DeepSeek / some MoE models: `reasoning_content`
      - Anthropic (when wrapped):   `thinking`
    Some providers also return a list of content blocks (e.g. [{"text": "..."}]),
    so we handle both string and list-of-blocks forms.
    """
    candidates = ("reasoning", "reasoning_content", "thinking")
    try:
        msg = choice.message
        # First check declared attributes
        for field in candidates:
            val = getattr(msg, field, None)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, list) and val:
                joined = "".join(
                    (b.get("text", "") if isinstance(b, dict) else str(b))
                    for b in val
                ).strip()
                if joined:
                    return joined
        # Then check Pydantic v2 model_extra (where OpenAI SDK stashes unknown fields)
        extra = getattr(msg, "model_extra", None) or {}
        for field in candidates:
            val = extra.get(field)
            if isinstance(val, str) and val.strip():
                return val
            if isinstance(val, list) and val:
                joined = "".join(
                    (b.get("text", "") if isinstance(b, dict) else str(b))
                    for b in val
                ).strip()
                if joined:
                    return joined
    except Exception:
        pass
    return None


def _dump_raw_response(response) -> Optional[dict]:
    """Serialize the API response to a dict, including provider-specific extras.

    OpenAI SDK uses Pydantic v2; `model_dump()` covers declared fields but
    misses provider extras stashed in `model_extra` (e.g. `reasoning` from
    OpenRouter). We merge those in per-choice/message.

    Drops `logprobs` from the dump to keep io.jsonl files manageable
    (logprobs are already captured separately into the parsed record).
    """
    try:
        base = response.model_dump(mode="json")
    except Exception:
        return None
    try:
        for i, choice in enumerate(response.choices):
            # Merge choice-level extras
            ch_extra = getattr(choice, "model_extra", None) or {}
            base["choices"][i].update(ch_extra)
            # Merge message-level extras (where `reasoning` lives)
            msg_extra = getattr(choice.message, "model_extra", None) or {}
            base["choices"][i].setdefault("message", {}).update(msg_extra)
            # Drop logprobs to keep file size down
            base["choices"][i].pop("logprobs", None)
        # Merge top-level extras (e.g. provider name, generation_id)
        top_extra = getattr(response, "model_extra", None) or {}
        for k, v in top_extra.items():
            base.setdefault(k, v)
    except Exception:
        pass
    return base


def _extract_usage(usage) -> tuple[int, int, int, Optional[float], Optional[int]]:
    """Pull (prompt, completion, total, cost_usd, reasoning_tokens) from a usage object.

    Shared by the streaming and non-streaming paths so they can't drift.
    """
    if usage is None:
        return 0, 0, 0, None, None

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    cost_usd = None
    try:
        cost_usd = float(usage.cost)  # type: ignore[attr-defined]
    except Exception:
        pass

    reasoning_tokens: Optional[int] = None
    try:
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            rt = getattr(details, "reasoning_tokens", None)
            if rt is None:
                extra = getattr(details, "model_extra", None) or {}
                rt = extra.get("reasoning_tokens")
            if rt is not None:
                reasoning_tokens = int(rt)
        if reasoning_tokens is None:
            usage_extra = getattr(usage, "model_extra", None) or {}
            details_dict = usage_extra.get("completion_tokens_details") or {}
            if isinstance(details_dict, dict) and details_dict.get("reasoning_tokens") is not None:
                reasoning_tokens = int(details_dict["reasoning_tokens"])
    except Exception:
        pass

    return prompt_tokens, completion_tokens, total_tokens, cost_usd, reasoning_tokens


def _reasoning_enabled(model_cfg: dict) -> bool:
    """True only when reasoning is actually turned ON for this model.

    Distinguishes the reasoning-OFF variants (e.g. {"reasoning": {"enabled": False}})
    from genuine reasoning configs (OpenRouter {"reasoning": {"enabled": True}},
    Google/Anthropic thinking budgets, etc.). Reasoning-ON models are streamed
    proactively (they are slow → trigger OpenRouter's SSE keep-alive comments);
    everyone else uses the non-streaming path and still gets the reactive
    stream-fallback if a parse error shows up.
    """
    rc = model_cfg.get("reasoning_config")
    if not rc:
        return False
    r = rc.get("reasoning")
    if isinstance(r, dict) and r.get("enabled") is False:
        return False
    return True


def _is_sse_parse_error(exc: BaseException) -> bool:
    """True if an exception looks like a failed JSON parse of an SSE-comment body.

    OpenRouter injects SSE keep-alive comments (": OPENROUTER PROCESSING") into the
    response body for slow requests. The non-streaming client then fails to json.loads
    the body, raising a JSONDecodeError ("Expecting value: line N column 1"). The fix
    is to re-issue the request as a stream, which parses SSE correctly.
    """
    if isinstance(exc, json.JSONDecodeError):
        return True
    msg = str(exc)
    return "Expecting value" in msg or "Expecting ',' delimiter" in msg


def _build_request_kwargs(model_cfg: dict, messages: list[dict]) -> dict[str, Any]:
    """Assemble the create() kwargs shared by streaming and non-streaming paths."""
    extra_body: dict[str, Any] = {}
    if model_cfg.get("reasoning_config"):
        extra_body.update(model_cfg["reasoning_config"])
    extra_params: dict[str, Any] = dict(model_cfg.get("extra_params") or {})

    request_kwargs: dict[str, Any] = {
        "model": model_cfg["id"],
        "messages": messages,
        "max_tokens": config.MAX_COMPLETION_TOKENS,
        **extra_params,
    }
    if config.REQUEST_LOGPROBS:
        request_kwargs["logprobs"] = True
        request_kwargs["top_logprobs"] = config.TOP_LOGPROBS
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    return request_kwargs


async def _call_once(
    client: AsyncOpenAI,
    model_cfg: dict,
    messages: list[dict],
) -> ApiResponse:
    """Make a single (non-retried) API call and return an ApiResponse."""
    request_kwargs = _build_request_kwargs(model_cfg, messages)

    t0 = time.perf_counter()
    response = await client.chat.completions.create(**request_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Guard the same condition the streaming path already handles with
    # `if not chunk.choices: continue`. Raised rather than returned so the retry
    # decorator sees it — see EmptyResponseError.
    if not getattr(response, "choices", None):
        raise EmptyResponseError(
            f"provider returned no choices (id={getattr(response, 'id', None)}, "
            f"finish=n/a, usage={'yes' if getattr(response, 'usage', None) else 'no'})"
        )

    choice = response.choices[0]
    content = (getattr(choice, "message", None) and choice.message.content) or ""

    prompt_tokens, completion_tokens, total_tokens, cost_usd, reasoning_tokens = \
        _extract_usage(response.usage)

    return ApiResponse(
        model_id=model_cfg["id"],
        model_label=model_cfg["label"],
        raw_content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=cost_usd,
        latency_ms=round(latency_ms, 1),
        logprobs=_extract_logprobs(choice),
        finish_reason=choice.finish_reason,
        reasoning_content=_extract_reasoning(choice),
        raw_response=_dump_raw_response(response),
    )


async def _call_once_stream(
    client: AsyncOpenAI,
    model_cfg: dict,
    messages: list[dict],
) -> ApiResponse:
    """Streaming variant of _call_once.

    Used for reasoning models (and as a fallback when a non-streaming call hits
    an SSE-comment parse error). Streaming lets the SDK parse SSE events properly
    and skip OpenRouter's ": OPENROUTER PROCESSING" keep-alive comments, which the
    non-streaming JSON parser cannot handle. Reassembles content / reasoning /
    usage / first-token logprobs from the chunk deltas.
    """
    request_kwargs = _build_request_kwargs(model_cfg, messages)
    request_kwargs["stream"] = True
    request_kwargs["stream_options"] = {"include_usage": True}

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: Optional[str] = None
    usage = None
    logprob_info: Optional[LogprobInfo] = None
    resp_id: Optional[str] = None

    t0 = time.perf_counter()
    stream = await client.chat.completions.create(**request_kwargs)
    async for chunk in stream:
        if resp_id is None:
            resp_id = getattr(chunk, "id", None)
        # Usage arrives in a trailing chunk (often with empty choices).
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        delta = ch.delta
        if getattr(delta, "content", None):
            content_parts.append(delta.content)
        # Reasoning deltas: declared attr or OpenRouter's model_extra "reasoning".
        r = getattr(delta, "reasoning", None)
        if r is None:
            r = (getattr(delta, "model_extra", None) or {}).get("reasoning")
        if isinstance(r, str) and r:
            reasoning_parts.append(r)
        if ch.finish_reason:
            finish_reason = ch.finish_reason
        # First-token logprobs, best effort (many providers omit them when streaming).
        if logprob_info is None and getattr(ch, "logprobs", None):
            li = _extract_logprobs(ch)
            if li is not None:
                logprob_info = li
    latency_ms = (time.perf_counter() - t0) * 1000

    content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts).strip() or None
    prompt_tokens, completion_tokens, total_tokens, cost_usd, reasoning_tokens = \
        _extract_usage(usage)

    usage_dump = None
    if usage is not None:
        try:
            usage_dump = usage.model_dump(mode="json")
        except Exception:
            usage_dump = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
    raw_response = {
        "id": resp_id,
        "model": model_cfg["id"],
        "streamed": True,
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content, "reasoning": reasoning_content},
        }],
        "usage": usage_dump,
    }

    return ApiResponse(
        model_id=model_cfg["id"],
        model_label=model_cfg["label"],
        raw_content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=cost_usd,
        latency_ms=round(latency_ms, 1),
        logprobs=logprob_info,
        finish_reason=finish_reason,
        reasoning_content=reasoning_content,
        raw_response=raw_response,
    )


async def _dispatch_call(
    client: AsyncOpenAI,
    model_cfg: dict,
    messages: list[dict],
) -> ApiResponse:
    """Pick streaming vs non-streaming, with a reactive stream fallback.

    Reasoning models stream proactively (they are slow → trigger SSE keep-alive
    comments). Everyone else uses the non-streaming path, but if that path hits an
    SSE-comment parse error we retry once as a stream rather than failing.
    """
    if _reasoning_enabled(model_cfg):
        return await _call_once_stream(client, model_cfg, messages)
    try:
        return await _call_once(client, model_cfg, messages)
    except Exception as e:
        if _is_sse_parse_error(e):
            logger.warning(
                "Non-stream parse failed for %s (SSE keep-alive?); retrying as stream",
                model_cfg["label"],
            )
            return await _call_once_stream(client, model_cfg, messages)
        raise


async def call_model(
    model_cfg: dict,
    messages: list[dict],
) -> ApiResponse:
    """
    Call OpenRouter with retry on transient errors.
    Returns an ApiResponse; on permanent failure, returns one with error set.
    """
    client = get_client()

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _with_retry():
        try:
            return await _dispatch_call(client, model_cfg, messages)
        except APIStatusError as e:
            if e.status_code == 429:
                await _wait_for_retry_after(e)
            raise

    try:
        return await _with_retry()
    except APIStatusError as e:
        logger.error("API error %s for model %s: %s", e.status_code, model_cfg["label"], e.message)
        return ApiResponse(
            model_id=model_cfg["id"],
            model_label=model_cfg["label"],
            raw_content="",
            error=f"APIStatusError {e.status_code}: {e.message}",
        )
    except Exception as e:
        logger.error("Unexpected error for model %s: %s", model_cfg["label"], str(e))
        return ApiResponse(
            model_id=model_cfg["id"],
            model_label=model_cfg["label"],
            raw_content="",
            error=str(e),
        )
