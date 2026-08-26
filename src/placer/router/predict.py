from __future__ import annotations

import numpy as np

from router.artifact import Artifact
from router.features import extract

NEIGHBOUR_K = (5, 20, 80)
NEIGHBOUR_T = (0.05, 0.2)


def _retrieval_features(similarity: np.ndarray, targets: np.ndarray) -> np.ndarray:
    order = np.argsort(-similarity, axis=1)
    columns = []
    for k in NEIGHBOUR_K:
        index = order[:, :k]
        sims = np.take_along_axis(similarity, index, 1)
        for t in NEIGHBOUR_T:
            weights = np.exp((sims - sims.max(1, keepdims=True)) / t)
            weights /= weights.sum(1, keepdims=True)
            columns.append((targets[index] * weights[:, :, None]).sum(1))
        columns.append(targets[index].std(1))
        columns.append(sims.mean(1, keepdims=True))
        columns.append(sims[:, :1])
    return np.hstack(columns)


def predict_batch(artifact: Artifact, prompts):
    raw = np.array(
        [extract(p, artifact.tokenizer) for p in prompts], dtype=np.float64
    )
    z = (raw - artifact.mean) / artifact.std

    tokens = raw[:, :4096]
    weighted = tokens * artifact.metric
    weighted = weighted / np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-9)
    query = weighted @ artifact.projection
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-9)

    similarity = query @ artifact.database.T
    block = _retrieval_features(similarity, artifact.database_scores)
    r = (block - artifact.retrieval_mean) / artifact.retrieval_std

    scores = np.zeros((len(prompts), 3))
    for i, head in enumerate(artifact.score_heads):
        ngram = z @ head["ngram"] + head["ngram_bias"]
        retrieval = r @ head["retrieval"] + head["bias"]
        scores[:, i] = np.clip(0.8 * retrieval + 0.2 * ngram, 0.0, 1.0)

    costs = np.zeros((len(prompts), 3))
    for i, head in enumerate(artifact.logout_heads):
        predicted = z @ head["weights"] + head["bias"]
        costs[:, i] = artifact.calibrate(i, predicted)

    gate = z @ artifact.gate_head["weights"] + artifact.gate_head["bias"]
    return scores.tolist(), costs.tolist(), gate.tolist()
