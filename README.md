# Agent Structured Output Validator (demo)

Middleware pattern for putting an LLM in front of a business system safely: never trust the model's text directly. Extract the JSON, validate it against a typed schema, and if it fails, re-prompt the model with the exact errors.

## What it does
- Pulls a JSON object out of messy output (markdown fences, extra chatter)
- Validates with Pydantic v2 (typed fields, ranges, allowed values)
- `extract_with_retries()` feeds validation errors back to the model and tries again, then fails loudly after N attempts
- The model call is a plain function you pass in, so it works with any provider and is easy to test with a fake

## Run it
```
pip install -r requirements.txt -r requirements-dev.txt
python validator.py
pytest
```

## Notes
The example schema is a made-up client intake profile with anonymized ids. No real client data is included.
