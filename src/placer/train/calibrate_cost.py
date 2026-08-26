from __future__ import annotations

import json

import numpy as np

from router.constants import MODEL_IDS
from train.paths import ARTIFACTS

GRID = 64


def main() -> int:
    train = np.load(ARTIFACTS / "dataset_train.npz")
    heads = np.load(ARTIFACTS / "heads.npz")
    x = train["features"].astype(np.float64)
    z = (x - heads["mean"]) / heads["std"]

    curves = []
    for index, model_id in enumerate(MODEL_IDS):
        predicted = z @ heads[f"logout_{index}_coef"] + heads[f"logout_{index}_bias"]
        order = np.argsort(predicted)
        sorted_actual = np.sort(train["costs"][:, index])
        sorted_pred = np.sort(predicted)
        knots = np.linspace(0, len(order) - 1, GRID).astype(int)
        curves.append(
            {
                "pred_knots": [float(sorted_pred[k]) for k in knots],
                "cost_knots": [float(sorted_actual[k]) for k in knots],
            }
        )
        print(f"{model_id:12s} pred range {sorted_pred[0]:.2f}..{sorted_pred[-1]:.2f}")

    (ARTIFACTS / "cost_curves.json").write_text(json.dumps(curves), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
