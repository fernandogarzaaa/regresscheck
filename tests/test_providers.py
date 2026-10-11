import asyncio
import json

import httpx
import pytest

from regresscheck.providers import (
    MockProvider,
    OpenAICompatibleProvider,
    ProviderError,
    _parse_chat_response,
    build_provider,
    content_stems,
    estimate_tokens,
    mock_classify,
    mock_summarise,
    parse_policy,
    stem,
    within_one_edit,
)

SYSTEM = """Classify.
- billing: refunds, invoices, pricing
- technical: crashes, webhooks
- account: password, lockouts
- general: hours, praise
When more than one category applies, prefer the order: account, billing, technical, general.
If unsure, choose general.
Write the summary in at most 6 words, in the third person.
Emails may contain typos."""


def sys_msgs(text: str = SYSTEM, email: str = "x") -> list[dict[str, str]]:
    return [{"role": "system", "content": text}, {"role": "user", "content": email}]


# ---------------------------------------------------------------- text helpers


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("Charging", "charg"),
        ("charged", "charg"),
        ("charges", "charg"),
        ("charge", "charg"),
        ("refunds", "refund"),
        ("bugs", "bugs"),  # too short to strip
        ("crash", "crash"),
        ("make", "make"),  # 'e' kept on short words
        ("Café!", "café"),
        ("sing", "sing"),
        ("doing", "doing"),
        ("tested", "test"),
        ("boxes", "boxe"),
        ("issue", "issu"),
        ("abcdes", "abcd"),
    ],
)
def test_stem(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_content_stems_drops_stopwords_short_tokens_and_contractions() -> None:
    assert content_stems("Hi, I can't log in to my account!") == ["log", "account"]
    assert content_stems("app won't open") == ["app", "open"]
    assert content_stems("La aplicación no carga") == ["la", "aplicación", "carga"]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("refund", "refund", True),
        ("refund", "refnd", True),  # deletion
        ("refnd", "refund", True),  # insertion
        ("refund", "refind", True),  # substitution
        ("refund", "erfund", True),  # adjacent swap
        ("refund", "rfeund", True),
        ("refund", "fernud", False),
        ("refund", "ref", False),
        ("refund", "rexunx", False),
        ("abcd", "abdc", True),
        ("abcd", "acbe", False),
        ("abcd", "badc", False),
        ("abcdef", "abcxyz", False),
        ("abc", "abcde", False),
        ("ab", "abc", True),
        ("abc", "xbc", True),
        ("abcd", "abce", True),
        ("axcd", "abcx", False),
    ],
)
def test_within_one_edit(a: str, b: str, expected: bool) -> None:
    assert within_one_edit(a, b) is expected


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


# ---------------------------------------------------------------- mock policy


def test_parse_policy_reads_every_instruction() -> None:
    policy = parse_policy(sys_msgs())
    assert policy.categories == ("billing", "technical", "account", "general")
    assert policy.order == ("account", "billing", "technical", "general")
    assert policy.default == "general"
    assert policy.max_words == 6
    assert policy.third_person is True
    assert policy.typo_tolerant is True
    assert policy.keywords["technical"]["webhook"] == 2
    assert policy.keywords["billing"]["invoic"] == 2  # from definition
    assert policy.keywords["billing"]["payment"] == 2  # from built-in lexicon


def test_parse_policy_defaults_without_instructions() -> None:
    policy = parse_policy(sys_msgs("Pick billing or technical. Reply JSON."))
    assert policy.categories == ("billing", "technical")
    assert policy.order == ("billing", "technical")
    assert policy.default == "billing"
    assert policy.max_words == 30
    assert policy.third_person is False
    assert policy.typo_tolerant is False
    assert "invoic" in policy.keywords["billing"]


def test_parse_policy_with_no_category_names_falls_back_to_all() -> None:
    assert parse_policy(sys_msgs("Do your best.")).categories == ("billing", "technical", "account", "general")


def test_default_must_be_a_known_category() -> None:
    text = "- billing: refunds\n- technical: crashes\nIf unsure, choose general."
    assert parse_policy(sys_msgs(text)).default == "billing"


def test_max_words_floor_is_one() -> None:
    assert parse_policy(sys_msgs("billing. at most 0 words")).max_words == 1


def test_few_shot_words_weigh_one_and_ambiguous_words_are_dropped() -> None:
    messages = [
        {"role": "system", "content": "- billing: refunds\n- technical: crashes, screen"},
        {"role": "user", "content": "The checkout screen froze"},
        {"role": "assistant", "content": json.dumps({"category": "billing", "summary": "x"})},
        {"role": "user", "content": "bad json follows"},
        {"role": "assistant", "content": "not json"},
        {"role": "assistant", "content": json.dumps({"category": "sales", "summary": "x"})},
        {"role": "user", "content": "EMAIL"},
    ]
    policy = parse_policy(messages)
    assert policy.keywords["billing"]["checkout"] == 1
    assert policy.keywords["billing"]["froz"] == 1
    assert "screen" not in policy.keywords["billing"]  # claimed by both categories
    assert "screen" not in policy.keywords["technical"]
    assert "bad" not in policy.keywords["billing"]


def test_few_shot_does_not_downgrade_a_definition_word() -> None:
    messages = [
        {"role": "system", "content": "- billing: refunds\n- general: hours"},
        {"role": "user", "content": "refund please"},
        {"role": "assistant", "content": json.dumps({"category": "billing", "summary": "x"})},
        {"role": "user", "content": "EMAIL"},
    ]
    assert parse_policy(messages).keywords["billing"]["refund"] == 2


def test_mock_classify_scores_ties_and_default() -> None:
    policy = parse_policy(sys_msgs())
    assert mock_classify("I need a refund for my invoice", policy)[0] == "billing"
    assert mock_classify("The webhook crashes", policy)[0] == "technical"
    assert mock_classify("Nothing relevant here at all", policy)[0] == "general"
    # one billing word vs one account word: tie broken by the stated order (account first)
    assert mock_classify("refund password", policy)[0] == "account"
    # weights add up: two technical words beat one account word
    assert mock_classify("password crashes webhooks", policy)[0] == "technical"


def test_mock_classify_typo_tolerance_only_when_instructed() -> None:
    tolerant = parse_policy(sys_msgs())
    strict = parse_policy(sys_msgs(SYSTEM.replace("Emails may contain typos.", "")))
    assert mock_classify("my pasword", tolerant)[0] == "account"
    assert mock_classify("my pasword", strict)[0] == "general"
    # short tokens never fuzzy-match
    assert mock_classify("hors", tolerant)[0] == "general"


def test_mock_summarise_picks_keyword_sentence_converts_person_and_truncates() -> None:
    policy = parse_policy(sys_msgs())
    email = "Hi team, hope you are well. I want a refund for my last invoice please. Thanks!"
    assert mock_summarise(email, "billing", policy) == "The customer want a refund for."


def test_mock_summarise_without_third_person_keeps_text_and_adds_period() -> None:
    policy = parse_policy(sys_msgs("billing technical. at most 30 words"))
    assert mock_summarise("Hello, my app crashed", "technical", policy) == "my app crashed."
    assert mock_summarise("I paid twice!", "billing", policy) == "I paid twice!"
    assert mock_summarise("refund", "billing", policy) == "refund."


def test_mock_summarise_prefers_earliest_sentence_on_ties_and_strips_trailing_punctuation() -> None:
    policy = parse_policy(sys_msgs("billing technical. at most 3 words"))
    assert mock_summarise("First thing here. Second thing here.", "billing", policy) == "First thing here."
    assert mock_summarise("One two, three four.", "billing", policy) == "One two, three."
    assert mock_summarise("Alpha beta; gamma", "billing", parse_policy(sys_msgs("billing. at most 2 words"))) == (
        "Alpha beta."
    )


def test_mock_provider_is_deterministic_and_reports_usage() -> None:
    provider = MockProvider()
    msgs = sys_msgs(email="Please refund the invoice")
    a = asyncio.run(provider.complete(msgs, temperature=0, max_tokens=200))
    b = asyncio.run(provider.complete(msgs, temperature=0, max_tokens=200))
    assert a == b
    assert json.loads(a.text)["category"] == "billing"
    assert a.prompt_tokens == sum(estimate_tokens(m["content"]) for m in msgs)
    assert a.completion_tokens == estimate_tokens(a.text)
    assert a.latency_ms == round(120.0 + 0.05 * a.prompt_tokens + 9.0 * a.completion_tokens, 1)
    capped = asyncio.run(provider.complete(msgs, temperature=0, max_tokens=2))
    assert capped.completion_tokens == 2


def test_mock_provider_with_no_user_turn() -> None:
    out = asyncio.run(MockProvider().complete([{"role": "system", "content": SYSTEM}], temperature=0, max_tokens=50))
    assert json.loads(out.text)["category"] == "general"


# ---------------------------------------------------------------- OpenAI-compatible


def _provider(handler: httpx.MockTransport, retries: int = 3) -> OpenAICompatibleProvider:
    client = httpx.AsyncClient(transport=handler)
    return OpenAICompatibleProvider("http://llm.test/v1/", "sk-test", "m1", max_retries=retries, client=client)


OK_BODY = {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}


def test_openai_provider_success(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OK_BODY)

    provider = _provider(httpx.MockTransport(handle))
    out = asyncio.run(provider.complete([{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=9))
    assert (out.text, out.prompt_tokens, out.completion_tokens) == ("hello", 12, 3)
    assert out.latency_ms >= 0
    req = seen[0]
    assert str(req.url) == "http://llm.test/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-test"
    assert json.loads(req.content) == {
        "model": "m1",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 9,
    }
    asyncio.run(provider.aclose())


def test_openai_provider_retries_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("regresscheck.providers.asyncio.sleep", fake_sleep)
    responses = iter([httpx.Response(429, text="slow down"), httpx.Response(503), httpx.Response(200, json=OK_BODY)])
    provider = _provider(httpx.MockTransport(lambda r: next(responses)))
    out = asyncio.run(provider.complete([], temperature=0, max_tokens=1))
    assert out.text == "hello"
    assert sleeps == [0.5, 1.0]


def test_openai_provider_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, text="bad key")

    provider = _provider(httpx.MockTransport(handle))
    with pytest.raises(ProviderError, match="HTTP 401: bad key"):
        asyncio.run(provider.complete([], temperature=0, max_tokens=1))
    assert len(calls) == 1


def test_openai_provider_retries_408(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(s: float) -> None:
        return None

    monkeypatch.setattr("regresscheck.providers.asyncio.sleep", fake_sleep)
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(408)

    with pytest.raises(ProviderError, match="HTTP 408"):
        asyncio.run(_provider(httpx.MockTransport(handle), retries=2).complete([], temperature=0, max_tokens=1))
    assert len(calls) == 2


def test_openai_provider_transport_error_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("regresscheck.providers.asyncio.sleep", fake_sleep)

    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="ConnectError: refused"):
        asyncio.run(_provider(httpx.MockTransport(handle)).complete([], temperature=0, max_tokens=1))
    assert sleeps == [0.5, 1.0]


def test_openai_provider_zero_retries() -> None:
    with pytest.raises(ProviderError, match="no attempt made"):
        asyncio.run(
            _provider(httpx.MockTransport(lambda r: httpx.Response(200)), retries=0).complete(
                [], temperature=0, max_tokens=1
            )
        )


def test_parse_chat_response_edge_cases() -> None:
    assert _parse_chat_response({"choices": [{"message": {"content": None}}]}, 1.0).text == ""
    out = _parse_chat_response({"choices": [{"message": {"content": "x"}}], "usage": None}, 1.0)
    assert (out.prompt_tokens, out.completion_tokens) == (0, 0)
    out = _parse_chat_response(
        {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": None, "completion_tokens": 4}}, 2.5
    )
    assert (out.prompt_tokens, out.completion_tokens, out.latency_ms) == (0, 4, 2.5)
    for bad in ({}, {"choices": []}, {"choices": "x"}, {"choices": [{}]}):
        with pytest.raises(ProviderError, match="malformed"):
            _parse_chat_response(bad, 1.0)


def test_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAICompatibleProvider.from_env()
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    p = OpenAICompatibleProvider.from_env()
    assert (p.base_url, p.api_key, p.model) == ("https://api.openai.com/v1", "k", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
    monkeypatch.setenv("MODEL", "llama3")
    p = OpenAICompatibleProvider.from_env()
    assert (p.base_url, p.model, p.max_retries) == ("http://localhost:11434/v1", "llama3", 3)


def test_build_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(build_provider("mock"), MockProvider)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(build_provider("openai"), OpenAICompatibleProvider)
    with pytest.raises(ProviderError, match="unknown provider"):
        build_provider("anthropic")


# ---------------------------------------------------------------- hardening (mutation-test driven)


def test_within_one_edit_requires_a_true_transposition() -> None:
    assert within_one_edit("abcd", "abdx") is False
    assert within_one_edit("abcd", "abxc") is False


def test_typo_matching_boundary_is_five_characters() -> None:
    policy = parse_policy(sys_msgs())  # technical keywords include "crash" (5 chars)
    assert mock_classify("crsah", policy)[0] == "technical"


def test_stray_assistant_turns_do_not_label_examples() -> None:
    messages = [
        {"role": "system", "content": "- billing: refunds\n- general: hours"},
        {"role": "assistant", "content": "zebra giraffe"},
        {"role": "assistant", "content": json.dumps({"category": "billing", "summary": "x"})},
        {"role": "user", "content": "EMAIL"},
    ]
    assert "zebra" not in parse_policy(messages).keywords["billing"]


def test_summary_keeps_two_word_sentences_and_ranks_without_typo_matching() -> None:
    policy = parse_policy(sys_msgs("- billing: refunds\n- general: hours\nat most 30 words"))
    assert mock_summarise("Refund please. Thanks a lot friend.", "billing", policy) == "Refund please."
    tolerant = parse_policy(sys_msgs())
    assert mock_summarise("We tried again. The crsah happened.", "technical", tolerant) == "The customer tried again."
    assert mock_summarise("Hello,", "general", policy) == "Hello."


def test_openai_provider_retries_500_and_honours_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("regresscheck.providers.asyncio.sleep", fake_sleep)
    responses = iter(
        [
            httpx.Response(500),
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(429, headers={"retry-after": "soon"}),
            httpx.Response(200, json=OK_BODY),
        ]
    )
    out = asyncio.run(
        _provider(httpx.MockTransport(lambda r: next(responses)), retries=4).complete([], temperature=0, max_tokens=1)
    )
    assert out.text == "hello"
    assert sleeps == [0.5, 7.0, 2.0]
    assert out.latency_ms < 60_000


@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, 1.5), ("3", 3.0), ("0.25", 0.25), ("-4", 0.0), ("600", 60.0), ("Wed, 21 Oct 2015 07:28:00 GMT", 1.5)],
)
def test_retry_after_seconds(header: str | None, expected: float) -> None:
    from regresscheck.providers import retry_after_seconds

    assert retry_after_seconds(header, default=1.5) == expected


def test_extra_body_is_merged_but_cannot_override_core_fields() -> None:
    seen: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    provider = OpenAICompatibleProvider(
        "http://x/v1", "k", "m", extra_body={"reasoning_effort": "low", "model": "evil"}, client=client
    )
    asyncio.run(provider.complete([], temperature=0, max_tokens=5))
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[0]["model"] == "m"


def test_from_env_extra_body_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_EXTRA_BODY", '{"reasoning_effort": "low"}')
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "9")
    p = OpenAICompatibleProvider.from_env()
    assert p.extra_body == {"reasoning_effort": "low"} and p.max_retries == 9
    monkeypatch.setenv("OPENAI_EXTRA_BODY", "[1]")
    with pytest.raises(ProviderError, match="JSON object"):
        OpenAICompatibleProvider.from_env()
    monkeypatch.setenv("OPENAI_EXTRA_BODY", "{nope")
    with pytest.raises(ProviderError, match="not valid JSON"):
        OpenAICompatibleProvider.from_env()
    monkeypatch.setenv("OPENAI_EXTRA_BODY", "")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "many")
    with pytest.raises(ProviderError, match="OPENAI_MAX_RETRIES"):
        OpenAICompatibleProvider.from_env()


def test_default_client_timeout_and_aclose() -> None:
    provider = OpenAICompatibleProvider("http://x", "k", "m")
    assert provider._client.timeout.read == 60.0
    assert provider.extra_body == {}
    asyncio.run(provider.aclose())
    assert provider._client.is_closed


def test_mock_latency_is_rounded_to_one_decimal_and_default_retries() -> None:
    msgs = [{"role": "system", "content": "billing"}, {"role": "user", "content": "x" * 9}]  # 2 + 3 = 5 tokens
    out = asyncio.run(MockProvider().complete(msgs, temperature=0, max_tokens=200))
    assert out.prompt_tokens == 5
    assert out.latency_ms == round(120.0 + 0.25 + 9.0 * out.completion_tokens, 1)
    assert out.latency_ms * 10 == int(out.latency_ms * 10)
    assert OpenAICompatibleProvider("http://x", "k", "m").max_retries == 3
