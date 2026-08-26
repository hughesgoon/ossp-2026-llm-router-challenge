from __future__ import annotations

MODEL_IDS = ("ax31-light", "ax31", "axk1-think")
TIERS = ("fast", "balanced", "premium")

TIER_BUDGET = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
TIER_WEIGHT = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

RATE_INPUT = {"ax31-light": 1.0, "ax31": 2.127, "axk1-think": 6.565}
RATE_OUTPUT = {"ax31-light": 4.0, "ax31": 8.509, "axk1-think": 26.260}

CHARS_PER_INPUT_TOKEN = {
    "ax31-light": 1.2535,
    "ax31": 1.2283,
    "axk1-think": 1.2059,
}


def episode_cost(model_id: str, input_tokens: float, output_tokens: float) -> float:
    return (
        input_tokens * RATE_INPUT[model_id] + output_tokens * RATE_OUTPUT[model_id]
    ) / 1_000_000.0
