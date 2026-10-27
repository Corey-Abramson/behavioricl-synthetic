"""
Offline mock backend — plumbing test only, NOT a model
=====================================================

Added for this synthetic export so the whole pipeline can be run end to end
with no API key and no spend. Enable it with:

    export BEHAVIORICL_MOCK=1

WHAT THIS IS
    A crude deterministic arithmetic heuristic. It scans the behavioural-history
    part of the prompt for numbers, squashes their mean through a tanh, and maps
    the result onto the trait's legal output range. A zero-shot prompt has no
    history section, so the mock falls back to the midpoint of the range.

WHAT THIS IS NOT
    An LLM, or any kind of approximation of one. Every statistic that comes out
    of `3_Eval` while BEHAVIORICL_MOCK=1 is a property of this heuristic and of
    the synthetic data generator. Those numbers are a smoke test that the files,
    the hold-out logic and the metrics line up. They say nothing whatsoever
    about GPT-4o or about human behaviour. Do not cite, plot or compare them.

To get real predictions, unset BEHAVIORICL_MOCK and set OPENAI_API_KEY.
"""

import hashlib
import math
import re

# Legal output range per trait, keyed by the exact `prediction_range` string in
# 1_Prompts/trait_configs.py. Keyed this way (rather than inferred from the text)
# so that a config edit fails loudly instead of silently picking wrong bounds.
BOUNDS = {
    "0 (chose safe/fixed pay) or 1 (chose competitive pay)": (0.0, 1.0, "int"),
    "mean send across 4 rounds, 0.0 to 1.0": (0.0, 1.0, "float"),
    "proportion of rounds chose B: 0, 0.25, 0.5, 0.75, or 1.0": (0.0, 1.0, "quarter"),
    "round 1 guess, 0 to 100": (0.0, 100.0, "int"),
    "credits given to Participant A, integer 1 to 9 (B receives 10 minus that)": (1.0, 9.0, "int"),
    "number of boxes opened, 0 to 10": (0.0, 10.0, "int"),
    "credits given to another student, 0 to 10": (0.0, 10.0, "int"),
    "number of UNICEF donations out of 5 decisions, 0 to 5": (0.0, 5.0, "int"),
}

# Lines that report a decision rather than describe the game. Pulling numbers
# only from these keeps the slider count (48), the guess ceiling (100) and the
# payoff rates (0.20, 0.50) printed in the game blurbs out of the arithmetic.
_DECISION_LINE = re.compile(
    r"(?:sent|chose|guessed|gave|allocated|opened|reported|donated|kept|returned|"
    r"Choice:|Performance:|Decision:|proposal:|revised to:|group decision:)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

# Internal call counter. Zero-shot prompts are byte-identical across
# participants for most traits, so a purely prompt-derived prediction would be
# constant and its Pearson r undefined. Mixing the call index into the jitter
# gives the mock a non-degenerate spread. Deterministic for a fresh run from an
# empty output directory; resuming from a checkpoint shifts the sequence.
_call_index = 0


def _history_numbers(prompt: str) -> list[float]:
    """Decision-line numbers before the '---' separating history from the question."""
    history = prompt.split("\n---\n")[0] if "\n---\n" in prompt else ""
    nums = []
    for line in history.splitlines():
        if _DECISION_LINE.search(line):
            nums.extend(float(m.group()) for m in _NUMBER.finditer(line))
    # keep values on a decision-like scale; round labels and ids are dropped
    return [x for x in nums if 0.0 <= x <= 10.0]


def _jitter(prompt: str) -> float:
    """Deterministic value in [-0.5, 0.5), varying per call."""
    global _call_index
    _call_index += 1
    h = hashlib.sha256(f"{_call_index}:{prompt}".encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32 - 0.5


def _score(prompt: str) -> float:
    """Crude [0, 1] score from the history; 0.5 when there is no history."""
    nums = _history_numbers(prompt)
    if not nums:
        return 0.5
    nums.sort()
    median = nums[len(nums) // 2]
    return 0.5 * (1.0 + math.tanh((median - 3.0) / 3.0))


def _quantise(value: float, lo: float, hi: float, kind: str) -> float:
    value = max(lo, min(hi, value))
    if kind == "int":
        return float(round(value))
    if kind == "quarter":
        return round(round(value * 4) / 4, 2)
    return round(value, 2)


def _amplitude(prompt: str) -> float:
    """Jitter as a fraction of the output range.

    Wider when the prompt carries no behavioural history: the score is then a
    constant 0.5, and a narrow jitter would be erased by the rounding in
    _quantise, leaving a constant predictor whose Pearson r is undefined.
    """
    return 0.08 if _history_numbers(prompt) else 0.22


def mock_call_model(prompt: str, prediction_key: str, prediction_range: str) -> dict:
    if prediction_range not in BOUNDS:
        raise KeyError(
            f"mock_backend has no bounds for prediction_range {prediction_range!r}. "
            "Add an entry to BOUNDS, or run against the real API."
        )
    lo, hi, kind = BOUNDS[prediction_range]
    span = hi - lo
    raw = lo + span * _score(prompt) + _amplitude(prompt) * span * _jitter(prompt)
    return {prediction_key: _quantise(raw, lo, hi, kind)}


def mock_call_model_trust(prompt: str) -> dict:
    s = _score(prompt)
    send = round(round((5.0 * s + 5.0 * _amplitude(prompt) * _jitter(prompt)) * 2) / 2, 1)
    send = max(0.0, min(5.0, send))
    # the real receiver cap is 3 x the partner's send, which the prompt states but
    # this heuristic does not parse; returning a fraction of own send stays small
    ret = round(max(0.0, 1.5 * send * (0.4 + 0.3 * s)), 1)
    return {"trust_send": send, "trust_return": ret}
