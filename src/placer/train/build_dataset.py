from __future__ import annotations

import json

import numpy as np

from router.constants import MODEL_IDS, episode_cost
from router.features import extract, load_tokenizer
from train.paths import ARTIFACTS, DATA


def load_split(split: str):
    inputs = json.loads((DATA / split / "inputs-base.json").read_text("utf-8"))
    outcomes = json.loads((DATA / split / "outcomes.json").read_text("utf-8"))
    prompts = {e["episode_id"]: e.get("prompt", "") for e in inputs["episodes"]}
    models = {e["episode_id"]: e["models"] for e in outcomes["episodes"]}
    ids = sorted(i for i in prompts if i in models)
    texts = [prompts[i] for i in ids]
    scores = np.array([[float(models[i][m]["score"]) for m in MODEL_IDS] for i in ids])
    output_tokens = np.array(
        [[models[i][m]["output_tokens"] for m in MODEL_IDS] for i in ids], dtype=float
    )
    costs = np.array(
        [
            [
                episode_cost(m, models[i][m]["input_tokens"], models[i][m]["output_tokens"])
                for m in MODEL_IDS
            ]
            for i in ids
        ]
    )
    return texts, scores, output_tokens, costs


def main() -> int:
    tokenizer = load_tokenizer(ARTIFACTS / "bpe_rules.json")
    for split in ("train", "dev"):
        texts, scores, output_tokens, costs = load_split(split)
        features = np.array([extract(t, tokenizer) for t in texts], dtype=np.float32)
        np.savez(
            ARTIFACTS / f"dataset_{split}.npz",
            features=features,
            scores=scores,
            output_tokens=output_tokens,
            costs=costs,
            chars=np.array([len(t) for t in texts], dtype=float),
        )
        print(f"{split}: {features.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
