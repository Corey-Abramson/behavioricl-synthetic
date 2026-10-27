"""
Shared OpenAI client and call utilities for all inference scripts.

call_model(user_prompt, prediction_key, prediction_range) is the generic entry point.
It handles JSON parsing, post-processing, and retries for any trait.

For trust (two-output), use call_model_trust() which returns {trust_send, trust_return}.

Offline mode (added for this synthetic export): setting BEHAVIORICL_MOCK=1 routes
both entry points to mock_backend, a deterministic arithmetic heuristic, so the
pipeline runs end to end with no API key and no spend. Results produced in that
mode are a plumbing test only — see the warning in mock_backend.py.
"""

import json
import os
import re
import time
from openai import OpenAI, RateLimitError, APIError

import mock_backend

MODEL = os.environ.get("BEHAVIORICL_MODEL", "gpt-4o")  # e.g. "gpt-4o-mini" for cheaper tests

# Optional sampling settings; unset means the provider default.
TEMPERATURE = os.environ.get("BEHAVIORICL_TEMPERATURE")
SEED = os.environ.get("BEHAVIORICL_SEED")


def sampling_kwargs() -> dict:
    kw = {}
    if TEMPERATURE is not None:
        kw["temperature"] = float(TEMPERATURE)
    if SEED is not None:
        kw["seed"] = int(SEED)
    return kw


def run_settings() -> dict:
    """Model settings written into every output row, so each prediction records how it was made."""
    if mock_enabled():
        return {"model": "mock", "temperature": None, "seed": None}
    kw = sampling_kwargs()
    return {"model": MODEL, "temperature": kw.get("temperature"), "seed": kw.get("seed")}


def mock_enabled() -> bool:
    return os.environ.get("BEHAVIORICL_MOCK", "").strip() not in ("", "0", "false", "False")

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client


def _make_system_prompt(prediction_key: str, prediction_range: str) -> str:
    return (
        "You are a behavioral economist predicting how a student will behave in an economic game. "
        "Respond ONLY with a JSON object — no explanation, no markdown fences — in exactly this format:\n"
        f'{{"{prediction_key}": <number>}}\n'
        f"The value must be {prediction_range}."
    )


def _make_trust_system_prompt() -> str:
    return (
        "You are a behavioral economist predicting how a student will behave in an economic game. "
        "Respond ONLY with a JSON object — no explanation, no markdown fences — in exactly this format:\n"
        '{"trust_send": <number>, "trust_return": <number>}\n'
        "trust_send must be a multiple of 0.5 between 0 and 5. "
        "trust_return must be a non-negative number."
    )


def _strip_fences(raw: str) -> str:
    return re.sub(r"^```[a-z]*\n?|```$", "", raw.strip())


def call_model(
    user_prompt: str,
    prediction_key: str,
    prediction_range: str,
    retries: int = 3,
    backoff: float = 5.0,
) -> dict:
    """
    Call the model for a single-output trait.
    Returns {prediction_key: value} where value is a float.
    """
    if mock_enabled():
        return mock_backend.mock_call_model(user_prompt, prediction_key, prediction_range)

    client = get_client()
    system_prompt = _make_system_prompt(prediction_key, prediction_range)
    raw = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=64,
                **sampling_kwargs(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = _strip_fences(raw)
            result = json.loads(raw)
            value = float(result[prediction_key])
            return {prediction_key: value}
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to parse model output after {retries} attempts: {e}\nRaw: {raw!r}")
            time.sleep(backoff * (attempt + 1))
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except APIError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)


def call_model_trust(
    user_prompt: str,
    retries: int = 3,
    backoff: float = 5.0,
) -> dict:
    """
    Call the model for the trust trait (two outputs: trust_send + trust_return).
    Returns {"trust_send": float, "trust_return": float}.
    """
    if mock_enabled():
        return mock_backend.mock_call_model_trust(user_prompt)

    client = get_client()
    system_prompt = _make_trust_system_prompt()
    raw = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=64,
                **sampling_kwargs(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            raw = _strip_fences(raw)
            result = json.loads(raw)
            trust_send = float(result["trust_send"])
            trust_return = float(result["trust_return"])
            # round trust_send to nearest 0.5, clip to [0, 5]
            trust_send = round(round(trust_send * 2) / 2, 1)
            trust_send = max(0.0, min(5.0, trust_send))
            trust_return = max(0.0, trust_return)
            return {"trust_send": trust_send, "trust_return": trust_return}
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to parse model output after {retries} attempts: {e}\nRaw: {raw!r}")
            time.sleep(backoff * (attempt + 1))
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except APIError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
