"""
LLM structured-output validation with automatic retries.

Takes free-text model output, extracts a JSON object, validates it against a
typed schema (Pydantic v2), and, if it fails, asks the model again with the
exact validation errors so it can correct itself.
"""

import json
import re
from typing import Callable, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError


class DiscoveryProfile(BaseModel):
    client_id: str = Field(..., min_length=1, description="Anonymized internal id")
    target_retirement_age: int = Field(..., ge=45, le=85)
    monthly_income_needed: float = Field(..., gt=0)
    existing_accounts: List[str] = Field(default_factory=list)
    risk_assessment: Literal["Conservative", "Moderate", "Aggressive"]
    notes: Optional[str] = None


def extract_json(raw: str) -> dict:
    """Pulls the first JSON object out of text that may include markdown fences or chatter."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output")
        text = text[start : end + 1]
    return json.loads(text)


def parse_profile(raw: str) -> DiscoveryProfile:
    return DiscoveryProfile(**extract_json(raw))


def extract_with_retries(
    call_llm: Callable[[str], str],
    prompt: str,
    max_attempts: int = 3,
) -> DiscoveryProfile:
    """
    call_llm: function taking a prompt string and returning the model's text.
    On failure, re-prompts with the specific errors. Raises after max_attempts.
    """
    current = prompt
    last_error = ""
    for _ in range(max_attempts):
        raw = call_llm(current)
        try:
            return parse_profile(raw)
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
