from __future__ import annotations

import json

from router.allocate import allocate
from router.artifact import load
from router.constants import MODEL_IDS, TIER_BUDGET, TIER_WEIGHT, episode_cost
from router.predict import predict_batch
from train.paths import ARTIFACTS, DATA


def load_truth(split):
    inputs = json.loads((DATA / split / "inputs-base.json").read_text("utf-8"))
    outcomes = json.loads((DATA / split / "outcomes.json").read_text("utf-8"))
    prompts = {e["episode_id"]: e.get("prompt", "") for e in inputs["episodes"]}
    models = {e["episode_id"]: e["models"] for e in outcomes["episodes"]}
    ids = sorted(i for i in prompts if i in models)
    texts = [prompts[i] for i in ids]
    scores = [[float(models[i][m]["score"]) for m in MODEL_IDS] for i in ids]
    costs = [
        [
            episode_cost(m, models[i][m]["input_tokens"], models[i][m]["output_tokens"])
            for m in MODEL_IDS
        ]
        for i in ids
    ]
    return texts, scores, costs


def main() -> int:
    artifact = load(ARTIFACTS / "router-v2.json")
    for split in ("dev",):
        texts, true_scores, true_costs = load_truth(split)
        est_scores, est_costs, _ = predict_batch(artifact, texts)
        base = sum(row[0] for row in true_costs)
        total = 0.0
        print(f"[{split}] n={len(texts)}")
        for tier, multiplier in TIER_BUDGET.items():
            selection = allocate(est_scores, est_costs, tier, multiplier)
            cost = sum(true_costs[i][selection[i]] for i in range(len(selection)))
            score = sum(true_scores[i][selection[i]] for i in range(len(selection)))
            ratio = cost / base
            passed = ratio <= multiplier
            mean_score = score / len(selection)
            total += TIER_WEIGHT[tier] * (mean_score if passed else 0.0)
            print(
                f"  {tier:9s} ratio={ratio:.3f}/{multiplier} "
                f"{'OK' if passed else 'OVER'} score={mean_score:.4f} "
                f"dist={[selection.count(k) for k in range(3)]}"
            )
        print(f"  total = {total:.4f}   baseline 0.6954")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
