from __future__ import annotations

import json

import numpy as np

from router.constants import MODEL_IDS
from router.features import FEATURE_VERSION
from train.paths import ARTIFACTS


def main() -> int:
    heads = np.load(ARTIFACTS / "heads.npz")
    curves = json.loads((ARTIFACTS / "cost_curves.json").read_text("utf-8"))
    rules = json.loads((ARTIFACTS / "bpe_rules.json").read_text("utf-8"))
    train = np.load(ARTIFACTS / "dataset_train.npz")

    payload = {
        "artifact_type": "ossp-bpe-retrieval-v2",
        "feature_version": FEATURE_VERSION,
        "model_ids": list(MODEL_IDS),
        "bpe_rules": rules,
        "normalize": {"mean": heads["mean"].tolist(), "std": heads["std"].tolist()},
        "metric": np.round(heads["metric"], 6).tolist(),
        "projection": np.round(heads["projection"], 6).tolist(),
        "database": {
            "vectors": np.round(heads["database"], 5).tolist(),
            "scores": train["scores"].tolist(),
        },
        "retrieval_normalize": {
            "mean": heads["retr_mean"].tolist(),
            "std": heads["retr_std"].tolist(),
        },
        "score_heads": [
            {
                "ngram": heads[f"score_ng_{i}_coef"].tolist(),
                "retrieval": heads[f"score_rt_{i}_coef"].tolist(),
                "bias": float(heads[f"score_{i}_bias"]),
                "ngram_bias": float(heads[f"score_ng_{i}_bias"]),
            }
            for i in range(3)
        ],
        "logout_heads": [
            {
                "weights": heads[f"logout_{i}_coef"].tolist(),
                "bias": float(heads[f"logout_{i}_bias"]),
            }
            for i in range(3)
        ],
        "gate_head": {
            "weights": heads["gate_coef"].tolist(),
            "bias": float(heads["gate_bias"]),
        },
        "cost_curves": curves,
    }
    path = ARTIFACTS / "router-v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
