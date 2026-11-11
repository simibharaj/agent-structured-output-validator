"""
LLM structured-output validation with automatic retries.

Takes free-text model output, extracts a JSON object, validates it against a
typed schema (Pydantic v2), and, if it fails, asks the model again with the
exact validation errors so it can correct itself.
"""

import json
import re
from typing import Callable, List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)


class DiscoveryProfile(BaseModel):
    client_id: str = Field(..., min_length=1, description="Anonymized internal id")
    target_retirement_age: int = Field(..., ge=45, le=85)
    monthly_income_needed: float = Field(..., gt=0)
    existing_accounts: List[str] = Field(default_factory=list)
    risk_assessment: Literal["Conservative", "Moderate", "Aggressive"]
    notes: Optional[str] = None


class MeetingActionItem(BaseModel):
    task: str = Field(..., min_length=1)
    owner: Optional[str] = None


class MeetingNotes(BaseModel):
    title: str = Field(..., min_length=1)
    attendees: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    action_items: List[MeetingActionItem] = Field(default_factory=list)


class DuplicateJSONKeyError(ValueError):
    """Raised when an object repeats a member name."""


def _reject_duplicate_keys(pairs):
    """Build JSON objects while rejecting repeated member names."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _matching_object_end(text: str, start: int) -> Optional[int]:
    """Return the end of a brace-balanced object, ignoring braces in strings."""
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1

    return None


def _extract_json_object(text: str) -> dict:
    """Find the first complete JSON object in text that may include chatter."""
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys)
    last_decode_error: json.JSONDecodeError | None = None
    start = 0

    while True:
        start = text.find("{", start)
        if start == -1:
            break

        try:
            value, end = decoder.raw_decode(text, start)
        except DuplicateJSONKeyError:
            raise
        except json.JSONDecodeError as error:
            if last_decode_error is None:
                last_decode_error = error
            candidate_end = _matching_object_end(text, start)
            if candidate_end is None:
                break
            start = candidate_end
            continue

        if isinstance(value, dict):
            return value
        start = end

    if last_decode_error is not None:
        raise last_decode_error
    raise ValueError("no JSON object found in model output")


def extract_json(raw: str) -> dict:
    """Pulls the first JSON object out of text that may include markdown fences or chatter."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1), object_pairs_hook=_reject_duplicate_keys)
    return _extract_json_object(text)


def parse_output(raw: str, model: Type[ModelT]) -> ModelT:
    """Extract JSON from model output and validate it against the requested Pydantic model."""
    return model(**extract_json(raw))


def parse_profile(raw: str) -> DiscoveryProfile:
    """Backward-compatible parser for the original discovery-profile example."""
    return parse_output(raw, DiscoveryProfile)


def extract_with_retries(
    call_llm: Callable[[str], str],
    prompt: str,
    max_attempts: int = 3,
    model: Type[ModelT] = DiscoveryProfile,
) -> ModelT:
    """
    call_llm: function taking a prompt string and returning the model's text.
    model: Pydantic model to validate; defaults to DiscoveryProfile for compatibility.
    On failure, re-prompts with the specific errors. Raises after max_attempts.
    """
    current = prompt
    last_error = ""
    for _ in range(max_attempts):
        raw = call_llm(current)
        try:
            return parse_output(raw, model)
        except (ValueError, ValidationError) as err:  # JSONDecodeError is a ValueError
            last_error = str(err)
            current = (
                f"{prompt}\n\nYour previous answer was invalid:\n{last_error}\n"
                "Return ONLY a corrected JSON object matching the schema."
            )
    raise RuntimeError(f"failed after {max_attempts} attempts: {last_error}")


if __name__ == "__main__":
    good = '```json\n{"client_id":"usr_demo","target_retirement_age":62,"monthly_income_needed":4500,' \
           '"existing_accounts":["Traditional IRA"],"risk_assessment":"Moderate"}\n```'
    print(parse_profile(good).model_dump_json(indent=2))
