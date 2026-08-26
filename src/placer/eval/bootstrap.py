from __future__ import annotations

import numpy as np

from router.allocate import allocate
from router.artifact import load
from router.constants import TIER_BUDGET, TIER_WEIGHT
from router.predict import predict_batch
from train.paths import ARTIFACTS
from eval.simulate import load_truth

CACHE = ARTIFACTS / "dev_pred.npz"


def cached_predictions():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["scores"], d["costs"], d["true_scores"], d["true_costs"]
    artifact = load(ARTIFACTS / "router-v2.json")
    texts, ts, tc = load_truth("dev")
    es, ec, _ = predict_batch(artifact, texts)
    np.savez(CACHE, scores=np.array(es), costs=np.array(ec),
             true_scores=np.array(ts), true_costs=np.array(tc))
    return np.array(es), np.array(ec), np.array(ts), np.array(tc)


def main() -> int:
    es, ec, ts, tc = cached_predictions()
    n = len(es)
    over = {t: 0 for t in TIER_BUDGET}
    totals = []
    for seed in [1, 2, 3, 101, 202, 303, 90909, 123456]:
        rng = np.random.default_rng(seed)
        for r in range(40):
            idx = np.arange(n) if r == 0 else rng.choice(n, int(n * 0.7), replace=False)
            total = 0.0
            for tier, mult in TIER_BUDGET.items():
                sel = allocate(es[idx].tolist(), ec[idx].tolist(), tier, mult)
                sel = np.array(sel)
                sub = tc[idx]
                ratio = sub[np.arange(len(idx)), sel].sum() / sub[:, 0].sum()
                score = ts[idx][np.arange(len(idx)), sel].mean()
                ok = ratio <= mult
                if not ok:
                    over[tier] += 1
                total += TIER_WEIGHT[tier] * (score if ok else 0.0)
            totals.append(total)
    print(f"320회: 초과={over}")
    print(f"  기대={np.mean(totals):.4f} p5={np.percentile(totals,5):.4f} 최악={min(totals):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
