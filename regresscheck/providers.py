"""LLM providers: any OpenAI-compatible endpoint, plus a deterministic built-in mock.

The mock is not a language model. It is a small rule-based simulator that reads the
system prompt the way an instruction-following model would (category definitions,
few-shot examples, tie-break order, default category, summary length, person/voice)
so that prompt edits change behaviour deterministically. That makes the whole pipeline
(eval, diffing, thresholds, drift, reports, CI gate) runnable for $0 with no keys.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .contract import CATEGORIES

Message = dict[str, str]


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class Provider(Protocol):
    name: str
    model: str

    async def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> Completion: ...


class ProviderError(RuntimeError):
    pass


# --------------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatibleProvider:
    """Chat Completions over HTTP. Works with OpenAI, Groq, OpenRouter, Ollama, vLLM, llama.cpp."""

    name = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        extra_body: dict[str, object] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set (use --provider mock to run without a key)")
        raw_extra = os.environ.get("OPENAI_EXTRA_BODY", "") or "{}"
        try:
            extra = json.loads(raw_extra)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"OPENAI_EXTRA_BODY is not valid JSON: {exc.msg}") from exc
        if not isinstance(extra, dict):
            raise ProviderError("OPENAI_EXTRA_BODY must be a JSON object")
        try:
            retries = int(os.environ.get("OPENAI_MAX_RETRIES", "") or 3)
        except ValueError as exc:
            raise ProviderError("OPENAI_MAX_RETRIES must be an integer") from exc
        return cls(
            base_url=os.environ.get("OPENAI_BASE_URL", "") or "https://api.openai.com/v1",
            api_key=key,
            model=os.environ.get("MODEL", "") or "gpt-4o-mini",
            max_retries=retries,
            extra_body=extra,
        )

    async def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> Completion:
        payload = {
            **self.extra_body,
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error = "no attempt made"
        for attempt in range(self.max_retries):
            delay = 0.5 * 2**attempt
            start = time.perf_counter()
            try:
                resp = await self._client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                latency = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    return _parse_chat_response(resp.json(), latency)
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in (408, 429) and resp.status_code < 500:
                    break
                delay = max(delay, retry_after_seconds(resp.headers.get("retry-after"), default=delay))
            if attempt + 1 < self.max_retries:
                await asyncio.sleep(delay)
        raise ProviderError(last_error)

    async def aclose(self) -> None:
        await self._client.aclose()


def retry_after_seconds(header: str | None, *, default: float, cap: float = 60.0) -> float:
    """Seconds to wait from a Retry-After header (numeric form only), capped; default if absent or unparseable."""
    if header is None:
        return default
    try:
        value = float(header)
    except ValueError:
        return default
    return min(cap, max(0.0, value))


def _parse_chat_response(data: dict[str, object], latency_ms: float) -> Completion:
    try:
        choices = data["choices"]
        assert isinstance(choices, list)
        text = choices[0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError, AssertionError) as exc:
        raise ProviderError(f"malformed chat response: {exc!r}") from exc
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return Completion(
        text=str(text),
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        latency_ms=latency_ms,
    )


# --------------------------------------------------------------------------- deterministic mock

# Deliberately small: the obvious words a model would know without being told. Everything
# else has to come from the prompt (definitions + few-shot examples), so prompt quality matters.
BASE_LEXICON: dict[str, frozenset[str]] = {
    "billing": frozenset({"invoice", "refund", "charge", "payment", "bill"}),
    "technical": frozenset({"error", "crash", "bug", "broken"}),
    "account": frozenset({"password", "login", "username"}),
    "general": frozenset({"question", "feedback", "hour"}),
}

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "too",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "any",
        "all",
        "also",
        "am",
        "get",
        "got",
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank",
        "please",
        "regards",
        "dear",
        "team",
        "still",
        "very",
        "really",
        "one",
        "some",
        "like",
        "via",
        "each",
        "other",
        "only",
        "e",
        "g",
        "eg",
        "etc",
        "such",
        "used",
        "use",
        "using",
        "don",
        "won",
        "couldn",
        "didn",
        "isn",
        "wasn",
        "aren",
        "doesn",
        "can",
        "ll",
        "ve",
        "re",
    ]
)

_DEF_RE = re.compile(r"^\s*[-*]?\s*(billing|technical|account|general)\s*[:\-\u2013]\s*(.+)$", re.I | re.M)
_ORDER_RE = re.compile(r"prefer(?:ence)?(?: the)? order\s*:?\s*([a-z ,>]+)", re.I)
_DEFAULT_RE = re.compile(
    r"(?:if unsure|when in doubt|if none apply|otherwise)[^.\n]*?\b(billing|technical|account|general)\b", re.I
)
_MAXWORDS_RE = re.compile(r"(?:at most|no more than|under|maximum of)\s+(\d+)\s+words", re.I)
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_GREETING_RE = re.compile(r"^(?:hi|hello|hey|dear|good (?:morning|afternoon|evening)|to whom)[^,.!\n]*[,.!]?\s*", re.I)
_PRONOUNS = (
    (r"\bI'm\b", "the customer is"),
    (r"\bI've\b", "the customer has"),
    (r"\bI\b", "the customer"),
    (r"\bmy\b", "their"),
    (r"\bme\b", "them"),
    (r"\bwe\b", "the customer"),
    (r"\bour\b", "their"),
)


def stem(word: str) -> str:
    w = re.sub(r"[\W_]", "", word.lower())
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            w = w[: -len(suffix)]
            break
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def content_stems(text: str) -> list[str]:
    return [s for s in (stem(t) for t in re.findall(r"[^\W_]+", text)) if len(s) > 1 and s not in STOPWORDS]


def within_one_edit(a: str, b: str) -> bool:
    """True if a and b differ by at most one insertion, deletion, substitution or adjacent swap."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        if len(diffs) != 2 or diffs[1] != diffs[0] + 1:
            return False
        i, j = diffs
        return a[i] == b[j] and a[j] == b[i]
    short, long_ = (a, b) if la < lb else (b, a)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


@dataclass(frozen=True)
class MockPolicy:
    """What the mock 'understood' from a system prompt + few-shot messages."""

    categories: tuple[str, ...]
    keywords: dict[str, dict[str, int]]
    order: tuple[str, ...]
    default: str
    max_words: int
    third_person: bool
    typo_tolerant: bool


def parse_policy(messages: list[Message]) -> MockPolicy:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    defs: dict[str, set[str]] = {}
    for name, text in _DEF_RE.findall(system):
        defs.setdefault(name.lower(), set()).update(content_stems(text))
    if defs:
        categories = tuple(defs)
    else:
        lowered = system.lower()
        categories = tuple(c for c in CATEGORIES if re.search(rf"\b{c}\b", lowered)) or CATEGORIES

    # Words the prompt states outright (and the model's own priors) weigh 2; words only seen
    # in few-shot examples weigh 1, the way a model generalises less from an example than a rule.
    keywords: dict[str, dict[str, int]] = {
        c: dict.fromkeys({stem(w) for w in BASE_LEXICON[c]} | defs.get(c, set()), 2) for c in categories
    }

    # Few-shot: a user turn followed by an assistant JSON turn labels that email.
    convo = [m for m in messages if m["role"] in ("user", "assistant")]
    for user, assistant in zip(convo[:-1], convo[1:]):
        if user["role"] != "user" or assistant["role"] != "assistant":
            continue
        try:
            label = str(json.loads(assistant["content"])["category"]).lower()
        except (ValueError, KeyError, TypeError):
            continue
        if label in keywords:
            for word in content_stems(user["content"]):
                keywords[label].setdefault(word, 1)

    # A word that several categories claim carries no signal.
    counts: dict[str, int] = {}
    for words in keywords.values():
        for w in words:
            counts[w] = counts.get(w, 0) + 1
    frozen = {c: {w: wt for w, wt in ws.items() if counts[w] == 1} for c, ws in keywords.items()}

    order = categories
    if m := _ORDER_RE.search(system):
        listed = [c for c in re.findall(r"[a-z]+", m.group(1).lower()) if c in categories]
        order = tuple(dict.fromkeys(listed + list(categories)))

    default = categories[0]
    if m := _DEFAULT_RE.search(system):
        default = m.group(1).lower() if m.group(1).lower() in categories else default

    max_words = int(m.group(1)) if (m := _MAXWORDS_RE.search(system)) else 30
    return MockPolicy(
        categories=categories,
        keywords=frozen,
        order=order,
        default=default,
        max_words=max(1, max_words),
        third_person=bool(re.search(r"third[- ]person", system, re.I)),
        typo_tolerant=bool(re.search(r"\btypos?\b|misspell", system, re.I)),
    )


def _hits(tokens: list[str], keywords: dict[str, int], typo_tolerant: bool) -> int:
    score = 0
    for tok in tokens:
        if tok in keywords:
            score += keywords[tok]
        elif typo_tolerant and len(tok) >= 5:
            score += max((wt for k, wt in keywords.items() if len(k) >= 5 and within_one_edit(tok, k)), default=0)
    return score


def mock_classify(email: str, policy: MockPolicy) -> tuple[str, str]:
    tokens = content_stems(email)
    scores = {c: _hits(tokens, policy.keywords[c], policy.typo_tolerant) for c in policy.categories}
    best = max(scores.values())
    category = policy.default if best == 0 else next(c for c in policy.order if scores[c] == best)
    return category, mock_summarise(email, category, policy)


def mock_summarise(email: str, category: str, policy: MockPolicy) -> str:
    body = _GREETING_RE.sub("", email.strip())
    sentences = [s.strip() for s in _SENT_RE.split(body) if len(s.strip().split()) >= 2] or [body.strip() or email]
    keys = policy.keywords.get(category, {})
    ranked = max(sentences, key=lambda s: (_hits(content_stems(s), keys, False), -sentences.index(s)))
    text = ranked
    if policy.third_person:
        for pattern, repl in _PRONOUNS:
            text = re.sub(pattern, repl, text, flags=re.I)
        text = text[0].upper() + text[1:] if text else text
    words = text.split()
    text = " ".join(words[: policy.max_words]).rstrip(",;:")
    return text if text.endswith((".", "!", "?")) else text + "."


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


class MockProvider:
    """Deterministic, offline, free. Latency is simulated from token counts (no sleeping)."""

    name = "mock"

    def __init__(self, model: str = "mock-classifier-v1") -> None:
        self.model = model

    async def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> Completion:
        policy = parse_policy(messages)
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        email = user_turns[-1] if user_turns else ""
        category, summary = mock_classify(email, policy)
        text = json.dumps({"category": category, "summary": summary})
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        completion_tokens = min(max_tokens, estimate_tokens(text))
        latency = 120.0 + 0.05 * prompt_tokens + 9.0 * completion_tokens
        await asyncio.sleep(0)
        return Completion(text, prompt_tokens, completion_tokens, round(latency, 1))


def build_provider(name: str) -> Provider:
    if name == "mock":
        return MockProvider()
    if name == "openai":
        return OpenAICompatibleProvider.from_env()
    raise ProviderError(f"unknown provider {name!r} (expected 'mock' or 'openai')")
