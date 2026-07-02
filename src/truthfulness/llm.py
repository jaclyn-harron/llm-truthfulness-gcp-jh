"""Thin Vertex AI (google-genai) wrapper: concurrency, retries, on-disk cache.

Every LLM call in the package goes through this module. The client targets
Vertex AI in your GCP project using Application Default Credentials — no API
keys. The cache keys on (model, messages, params) so re-running the evaluation
does not re-pay for identical calls; this is what lets the eval entrypoint be
both reproducible and cheap on a second run.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

from .config import Config, get_config

# HTTP status codes worth retrying: rate limits, transient server errors.
_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None) in _RETRYABLE_CODES
    # Connection-level failures surface as httpx/requests errors.
    return exc.__class__.__name__ in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
    }


_client: genai.Client | None = None
_client_lock = threading.Lock()


def client() -> genai.Client:
    """Return a process-wide Vertex AI client (ADC auth)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                cfg = get_config()
                _client = genai.Client(
                    vertexai=True,
                    project=cfg.require_project(),
                    location=cfg.location,
                    http_options=genai_types.HttpOptions(
                        timeout=cfg.request_timeout * 1000  # milliseconds
                    ),
                )
    return _client


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class _Cache:
    def __init__(self, cfg: Config):
        self.enabled = cfg.use_cache
        self.dir = cfg.cache_dir / "chat"
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        f = self.dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        return None

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        f = self.dir / f"{key}.json"
        with self._lock:
            f.write_text(json.dumps(value))


def _split_messages(messages: list[dict]) -> tuple[str | None, list[genai_types.Content]]:
    """Convert chat-style [{role, content}] into (system_instruction, contents)."""
    system: str | None = None
    contents: list[genai_types.Content] = []
    for m in messages:
        role, text = m["role"], m["content"]
        if role == "system":
            system = text if system is None else f"{system}\n\n{text}"
        else:
            contents.append(
                genai_types.Content(
                    role="model" if role in ("assistant", "model") else "user",
                    parts=[genai_types.Part.from_text(text=text)],
                )
            )
    return system, contents


def _thinking_config(model: str, budget: int) -> genai_types.ThinkingConfig | None:
    """Thinking budget for Gemini 2.5 flash-class models and endpoints tuned
    from them (`projects/…/endpoints/…`). budget<0 leaves the model default.
    Other models are left alone: 2.5-pro cannot disable thinking, and
    Gemini 3.x uses a different control (thinking_level)."""
    if budget < 0:
        return None
    if "2.5-flash" in model or "/endpoints/" in model:
        return genai_types.ThinkingConfig(thinking_budget=budget)
    return None


def _normalise_logprobs(candidate) -> list[dict]:
    """Flatten the first generation step's top candidates into
    [{"token","logprob"}, ...] (same shape the predictors expect)."""
    lr = getattr(candidate, "logprobs_result", None)
    if not lr or not getattr(lr, "top_candidates", None):
        return []
    first = lr.top_candidates[0]
    out = []
    for c in first.candidates or []:
        out.append({"token": c.token or "", "logprob": float(c.log_probability or 0.0)})
    return out


def chat_json(
    *,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    json_mode: bool = False,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    cache: _Cache | None = None,
) -> dict:
    """Single chat completion against Vertex AI. Returns a normalised dict:
    {"content": str, "top_logprobs": [...optional...]}.
    Retries transient errors with exponential backoff.
    """
    cfg = get_config()
    cache = cache if cache is not None else _Cache(cfg)
    payload = {
        "provider": "vertex",
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "thinking_budget": cfg.thinking_budget,
    }
    key = _cache_key(payload)
    hit = cache.get(key)
    if hit is not None:
        return hit

    system, contents = _split_messages(messages)
    gen_cfg = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system,
        response_mime_type="application/json" if json_mode else None,
        response_logprobs=True if logprobs else None,
        logprobs=(top_logprobs or 5) if logprobs else None,
        thinking_config=_thinking_config(model, cfg.thinking_budget),
    )

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_random_exponential(min=0.5, max=6),
        stop=stop_after_attempt(cfg.max_retries),
        reraise=True,
    )
    def _call():
        return client().models.generate_content(
            model=model, contents=contents, config=gen_cfg
        )

    resp = _call()
    candidate = resp.candidates[0] if resp.candidates else None
    result: dict[str, Any] = {"content": (resp.text or "") if candidate else ""}
    if logprobs and candidate is not None:
        tops = _normalise_logprobs(candidate)
        if tops:
            result["top_logprobs"] = tops
    cache.put(key, result)
    return result


def map_concurrent(
    fn: Callable[[int], Any],
    n: int,
    *,
    max_workers: int,
    desc: str = "",
    show_progress: bool = True,
) -> list[Any]:
    """Apply `fn(i)` for i in range(n) across a thread pool, preserving order."""
    from concurrent.futures import as_completed

    results: list[Any] = [None] * n
    if n == 0:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, i): i for i in range(n)}
        for fut in tqdm(
            as_completed(futures), total=n, desc=desc, disable=not show_progress
        ):
            i = futures[fut]
            results[i] = fut.result()
    return results


def new_cache() -> _Cache:
    return _Cache(get_config())
