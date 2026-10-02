import pytest
from validator import extract_json, parse_profile, extract_with_retries

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
