"""Optional adapter for OpenAI-compatible chat-completions APIs.

The adapter reads configuration from the environment and can be passed directly
to validator.extract_with_retries. It never stores credentials in the project.
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from validator import MeetingNotes, extract_with_retries


_REQUIRED_ENV = ("LLM_API_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


def call_llm(prompt: str) -> str:
    """Send a prompt to an OpenAI-compatible endpoint and return its text reply."""
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Set these environment variables before use: " + ", ".join(missing))

    base_url = os.environ["LLM_API_BASE_URL"].strip()
    api_key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"].strip()
    parsed = urlsplit(base_url)
    local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not parsed.netloc
        or (parsed.scheme != "https" and not local_http)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "LLM_API_BASE_URL must be an HTTPS base URL (HTTP is allowed only for loopback) "
            "without embedded credentials, a query, or a fragment."
        )

    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"Provider request failed with HTTP {error.code}.") from None
    except URLError:
        raise RuntimeError("Could not reach the configured provider.") from None

    try:
        content = payload["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError):
        raise RuntimeError("Provider response did not include choices[0].message.content.") from None
    if not isinstance(content, str):
        raise RuntimeError("Provider response content was not text.")
    return content


def main() -> None:
    notes = extract_with_retries(
        call_llm,
        "Summarize this fictional planning note as JSON meeting notes: "
        "Alex Example and Sam Sample agreed to keep the demo provider-neutral; "
        "Alex will add a small usage example.",
        model=MeetingNotes,
    )
    print(notes.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
