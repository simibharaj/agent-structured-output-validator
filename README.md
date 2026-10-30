# Agent Structured Output Validator (demo)

Middleware pattern for putting an LLM in front of a business system safely: never trust the model's text directly. Extract the JSON, validate it against a typed schema, and if it fails, re-prompt the model with the exact errors.

## What it does
- Pulls a JSON object out of messy output (markdown fences, extra chatter)
- Validates with Pydantic v2 (typed fields, ranges, allowed values)
- `extract_with_retries()` feeds validation errors back to the model and tries again, then fails loudly after N attempts
- The model call is a plain function you pass in, so it works with any provider and is easy to test with a fake
- `parse_output()` and `extract_with_retries()` accept any Pydantic v2 model; the included meeting-notes schema shows reuse beyond the original discovery-profile example

## Run it
```
pip install -r requirements.txt -r requirements-dev.txt
python validator.py
pytest
```

## Reuse with another schema

The retry loop can validate a different Pydantic model without changing the parser:

```python
from validator import MeetingNotes, extract_with_retries

notes = extract_with_retries(
    call_llm,  # your provider adapter: prompt in, model text out
    "Summarize the planning session as JSON.",
    model=MeetingNotes,
)
```

`MeetingNotes` includes nested action items and is a fictional project-planning example. The original `parse_profile()` helper remains available for the discovery-profile example.

## Optional live provider example

`llm_provider_example.py` shows how to connect the retry loop to a real chat-completions provider. It supports endpoints that accept the OpenAI-compatible `POST /chat/completions` request and response shape; other providers can use their own adapter with the same `call_llm(prompt) -> str` interface.

Set `LLM_API_BASE_URL` to the provider's HTTPS API base URL (for example, ending in `/v1`), `LLM_MODEL` to its model name, and `LLM_API_KEY` in your shell or secret manager. The example never stores the key. Remote URLs must use HTTPS; plain HTTP is accepted only for loopback development. Then run:

```sh
python llm_provider_example.py
```

The example sends only a fictional planning note. Its tests mock the HTTP request and do not call a provider.

## Notes
The examples and sample names are fictional. No real client data is included.
