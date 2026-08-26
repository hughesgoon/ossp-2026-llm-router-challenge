from __future__ import annotations

import json

from router.allocate import allocate
from router.artifact import load
from router.constants import MODEL_IDS, TIER_BUDGET
from router.predict import predict_batch
from train.paths import ARTIFACTS, DATA


def decide(artifact, episodes, tier):
    texts = [e["prompt"] for e in episodes]
    scores, costs, _ = predict_batch(artifact, texts)
    selection = allocate(scores, costs, tier, TIER_BUDGET[tier])
    return {t: MODEL_IDS[selection[i]] for i, t in enumerate(texts)}


def main() -> int:
    artifact = load(ARTIFACTS / "router-v2.json")
    batch = json.loads((DATA / "dev" / "inputs-base.json").read_text("utf-8"))
    original = batch["episodes"]
    shuffled = [
        {"episode_id": f"changed-{i}", "prompt": e["prompt"]}
        for i, e in enumerate(reversed(original))
    ]
    failures = 0
    for tier in TIER_BUDGET:
        first = decide(artifact, original, tier)
        second = decide(artifact, shuffled, tier)
        mismatched = [k for k in first if first[k] != second.get(k)]
        failures += len(mismatched)
        print(f"{tier:9s} {'OK' if not mismatched else f'MISMATCH {len(mismatched)}'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
