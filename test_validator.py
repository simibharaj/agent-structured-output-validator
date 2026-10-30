import io
import json
from urllib.error import HTTPError

import pytest
from pydantic import ValidationError

import llm_provider_example
from validator import (
    MeetingNotes,
    extract_json,
    extract_with_retries,
    parse_output,
    parse_profile,
)

GOOD = '{"client_id":"u1","target_retirement_age":62,"monthly_income_needed":4500,"risk_assessment":"Moderate"}'

def test_extracts_from_markdown_fence():
    assert extract_json("```json\n" + GOOD + "\n```")["client_id"] == "u1"

def test_extracts_from_chatty_text():
    assert extract_json("Sure! Here you go: " + GOOD + " Hope that helps.")["client_id"] == "u1"

def test_no_json_raises():
    with pytest.raises(ValueError):
        extract_json("no braces here")

def test_rejects_bad_risk_value():
    bad = GOOD.replace("Moderate", "Reckless")
    with pytest.raises(Exception):
        parse_profile(bad)

def test_rejects_out_of_range_age():
    with pytest.raises(Exception):
        parse_profile(GOOD.replace("62", "20"))

def test_retry_recovers_after_bad_answer():
    answers = iter(["not json", GOOD])
    prompts = []
    def fake(p):
        prompts.append(p)
        return next(answers)
    profile = extract_with_retries(fake, "extract please")
    assert profile.client_id == "u1"
    assert "previous answer was invalid" in prompts[1]

def test_gives_up_after_max_attempts():
    with pytest.raises(RuntimeError):
        extract_with_retries(lambda p: "never valid", "x", max_attempts=2)


NESTED = '{"outer": {"inner": {"k": 1}}, "list": [{"a": 1}, {"b": 2}]}'

def test_extracts_nested_json_from_fence():
    assert extract_json("```json\n" + NESTED + "\n```")["outer"]["inner"]["k"] == 1

def test_extracts_nested_json_from_chatty_text():
    assert extract_json("Result: " + NESTED + " done")["list"][1]["b"] == 2

def test_fence_without_language_tag():
    assert extract_json("```\n" + GOOD + "\n```")["client_id"] == "u1"

def test_fence_with_surrounding_text():
    raw = "Here is the answer:\n```json\n" + GOOD + "\n```\nLet me know if you need more."
    assert extract_json(raw)["client_id"] == "u1"

def test_uppercase_fence_tag():
    assert extract_json("```JSON\n" + GOOD + "\n```")["client_id"] == "u1"

def test_braces_in_chatter_before_fence():
    raw = "I used the {schema} you gave.\n```json\n" + GOOD + "\n```"
    assert extract_json(raw)["client_id"] == "u1"

def test_malformed_json_raises_value_error():
    with pytest.raises(ValueError):
        extract_json('{"client_id": "u1",}')


MEETING_NOTES = '''{
  "title": "Planning session",
  "attendees": ["Alex Example"],
  "decisions": ["Keep the demo provider-neutral"],
  "action_items": [{"task": "Add a sample", "owner": "Alex Example"}]
}'''


def test_parse_output_uses_requested_schema():
    notes = parse_output("```json\n" + MEETING_NOTES + "\n```", MeetingNotes)
    assert isinstance(notes, MeetingNotes)
    assert notes.action_items[0].task == "Add a sample"


def test_parse_output_rejects_invalid_requested_schema():
    with pytest.raises(ValidationError):
        parse_output('{"title":"","action_items":[{"task":""}]}', MeetingNotes)


def test_retry_uses_requested_schema_and_returns_typed_result():
    answers = iter(['{"title":"Planning session","action_items":[{"task":""}]}', MEETING_NOTES])
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return next(answers)

    notes = extract_with_retries(fake, "summarize the planning session", model=MeetingNotes)
    assert isinstance(notes, MeetingNotes)
    assert notes.title == "Planning session"
    assert "previous answer was invalid" in prompts[1]
    assert "task" in prompts[1]


def _set_provider_environment(monkeypatch, base_url="https://provider.example/v1"):
    monkeypatch.setenv("LLM_API_BASE_URL", base_url)
    monkeypatch.setenv("LLM_API_KEY", "test-only-not-a-secret")
    monkeypatch.setenv("LLM_MODEL", "example-model")


def test_provider_adapter_plugs_into_retry_loop_without_network(monkeypatch):
    _set_provider_environment(monkeypatch)
    content = (
        '{"title":"Planning session","attendees":["Alex Example"],'
        '"decisions":["Keep the demo provider-neutral"],'
        '"action_items":[{"task":"Add a sample","owner":"Alex Example"}]}'
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(llm_provider_example, "urlopen", fake_urlopen)
    notes = extract_with_retries(
        llm_provider_example.call_llm,
        "Summarize the fictional planning session.",
        model=MeetingNotes,
    )

    request = captured["request"]
    assert isinstance(notes, MeetingNotes)
    assert notes.action_items[0].task == "Add a sample"
    assert request.full_url == "https://provider.example/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == "Bearer test-only-not-a-secret"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {
        "model": "example-model",
        "messages": [{"role": "user", "content": "Summarize the fictional planning session."}],
        "temperature": 0,
    }
    assert captured["timeout"] == 30


def test_provider_adapter_requires_environment_configuration(monkeypatch):
    for name in llm_provider_example._REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="LLM_API_BASE_URL"):
        llm_provider_example.call_llm("a prompt")


def test_provider_adapter_rejects_remote_plain_http(monkeypatch):
    _set_provider_environment(monkeypatch, "http://provider.example/v1")

    with pytest.raises(ValueError, match="HTTPS"):
        llm_provider_example.call_llm("a prompt")


def test_provider_adapter_allows_loopback_http_for_local_development(monkeypatch):
    _set_provider_environment(monkeypatch, "http://127.0.0.1:1234/v1")
    body = json.dumps({"choices": [{"message": {"content": "local reply"}}]}).encode()
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        return io.BytesIO(body)

    monkeypatch.setattr(llm_provider_example, "urlopen", fake_urlopen)
    assert llm_provider_example.call_llm("a prompt") == "local reply"
    assert captured["url"] == "http://127.0.0.1:1234/v1/chat/completions"


def test_provider_adapter_does_not_echo_key_on_http_error(monkeypatch):
    _set_provider_environment(monkeypatch)

    def unauthorized(request, timeout):
        raise HTTPError(
            request.full_url,
            401,
            "unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"test-only-not-a-secret"),
        )

    monkeypatch.setattr(llm_provider_example, "urlopen", unauthorized)
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        llm_provider_example.call_llm("a prompt")
    assert "test-only-not-a-secret" not in str(error.value)
