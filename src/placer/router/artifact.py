from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from router.constants import MODEL_IDS
from router.features import FEATURE_DIM, FEATURE_VERSION, Tokenizer


class Artifact:
    def __init__(self, payload: dict):
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"feature_version mismatch: {payload.get('feature_version')} != {FEATURE_VERSION}"
            )
        if tuple(payload["model_ids"]) != MODEL_IDS:
            raise ValueError("model_ids mismatch")

        self.tokenizer = Tokenizer([tuple(pair) for pair in payload["bpe_rules"]])
        self.mean = payload["normalize"]["mean"]
        self.std = payload["normalize"]["std"]
        if len(self.mean) != FEATURE_DIM:
            raise ValueError(f"normalize length must be {FEATURE_DIM}")

        self.mean = np.array(self.mean)
        self.std = np.array(self.std)
        self.metric = np.array(payload["metric"])
        self.projection = np.array(payload["projection"])
        self.database = np.array(payload["database"]["vectors"])
        self.database_scores = np.array(payload["database"]["scores"])
        self.retrieval_mean = np.array(payload["retrieval_normalize"]["mean"])
        self.retrieval_std = np.array(payload["retrieval_normalize"]["std"])
        self.score_heads = [
            {"ngram": np.array(h["ngram"]), "retrieval": np.array(h["retrieval"]),
             "bias": h["bias"], "ngram_bias": h.get("ngram_bias", h["bias"])}
            for h in payload["score_heads"]
        ]
        self.logout_heads = [
            {"weights": np.array(h["weights"]), "bias": h["bias"]}
            for h in payload["logout_heads"]
        ]
        self.gate_head = {
            "weights": np.array(payload["gate_head"]["weights"]),
            "bias": payload["gate_head"]["bias"],
        }
        self.cost_curves = payload["cost_curves"]

    def calibrate(self, index: int, predicted):
        curve = self.cost_curves[index]
        return np.interp(predicted, curve["pred_knots"], curve["cost_knots"])


def load(path: Path) -> Artifact:
    with open(path, encoding="utf-8") as handle:
        return Artifact(json.load(handle))
