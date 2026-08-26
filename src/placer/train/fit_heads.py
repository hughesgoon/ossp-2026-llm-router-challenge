from __future__ import annotations

import json

import numpy as np

from router.constants import MODEL_IDS
from train.paths import ARTIFACTS

ALPHAS = (1000.0, 3000.0, 10000.0, 30000.0)
RETRIEVAL_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
NEIGHBOUR_K = (5, 20, 80)
NEIGHBOUR_T = (0.05, 0.2)
LABEL_SMOOTH = 0.30
LABEL_SMOOTH_K = 10


def r2(actual, predicted):
    return 1.0 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


def solve(gram, rhs, alpha):
    return np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)


def retrieval_features(sim, targets):
    order = np.argsort(-sim, axis=1)
    columns = []
    for k in NEIGHBOUR_K:
        index = order[:, :k]
        similarity = np.take_along_axis(sim, index, 1)
        for t in NEIGHBOUR_T:
            weight = np.exp((similarity - similarity.max(1, keepdims=True)) / t)
            weight /= weight.sum(1, keepdims=True)
            columns.append((targets[index] * weight[:, :, None]).sum(1))
        columns.append(targets[index].std(1))
        columns.append(similarity.mean(1, keepdims=True))
        columns.append(similarity[:, :1])
    return np.hstack(columns)


def main() -> int:
    train = np.load(ARTIFACTS / "dataset_train.npz")
    dev = np.load(ARTIFACTS / "dataset_dev.npz")
    xtr, xde = train["features"].astype(np.float64), dev["features"].astype(np.float64)
    mean, std = xtr.mean(0), xtr.std(0)
    std = np.where(std > 1e-9, std, 1.0)
    ztr, zde = (xtr - mean) / std, (xde - mean) / std
    gram = ztr.T @ ztr

    scores_tr, scores_de = train["scores"], dev["scores"]
    logout_tr = np.log(train["costs"] + 1e-9)
    logout_de = np.log(dev["costs"] + 1e-9)

    heads = {"mean": mean, "std": std}
    report = {}

    for index, model_id in enumerate(MODEL_IDS):
        target = logout_tr[:, index]
        best = max(
            (
                (r2(logout_de[:, index], zde @ solve(gram, ztr.T @ (target - target.mean()), a) + target.mean()), a)
                for a in ALPHAS
            )
        )
        coef = solve(gram, ztr.T @ (target - target.mean()), best[1])
        heads[f"logout_{index}_coef"] = coef
        heads[f"logout_{index}_bias"] = np.array(target.mean())
        report[f"logout_{model_id}"] = round(best[0], 4)

    vtr = xtr[:, :4096].copy()
    vde = xde[:, :4096].copy()
    weight_target = scores_tr.mean(1)
    centered = weight_target - weight_target.mean()
    numerator = (vtr * centered[:, None]).sum(0)
    denominator = np.sqrt((vtr ** 2).sum(0) * (centered ** 2).sum()) + 1e-9
    metric = (np.abs(numerator / denominator) / np.abs(numerator / denominator).max()) ** 0.5
    heads["metric"] = metric

    def weighted(matrix):
        scaled = matrix * metric
        return scaled / np.maximum(np.linalg.norm(scaled, axis=1, keepdims=True), 1e-9)

    atr, ade = weighted(vtr), weighted(vde)
    _, _, vt = np.linalg.svd(atr, full_matrices=False)
    projection = vt[:512].T
    heads["projection"] = projection
    atr = atr @ projection
    ade = ade @ projection
    atr = atr / np.maximum(np.linalg.norm(atr, axis=1, keepdims=True), 1e-9)
    ade = ade / np.maximum(np.linalg.norm(ade, axis=1, keepdims=True), 1e-9)
    heads["database"] = atr
    sim_tr = atr @ atr.T
    np.fill_diagonal(sim_tr, -9.0)
    sim_de = ade @ atr.T
    ftr = retrieval_features(sim_tr, scores_tr)
    fde = retrieval_features(sim_de, scores_tr)
    fmean, fstd = ftr.mean(0), ftr.std(0)
    fstd = np.where(fstd > 1e-9, fstd, 1.0)
    rtr, rde = (ftr - fmean) / fstd, (fde - fmean) / fstd
    rgram = rtr.T @ rtr
    heads["retr_mean"], heads["retr_std"] = fmean, fstd

    neighbour = np.argsort(-sim_tr, axis=1)[:, :LABEL_SMOOTH_K]
    for index, model_id in enumerate(MODEL_IDS):
        smoothed = (1.0 - LABEL_SMOOTH) * scores_tr[:, index] + LABEL_SMOOTH * scores_tr[
            neighbour, index
        ].mean(1)
        target = scores_tr[:, index]
        ng = max(
            (
                (r2(scores_de[:, index], zde @ solve(gram, ztr.T @ (smoothed - smoothed.mean()), a) + smoothed.mean()), a)
                for a in ALPHAS
            )
        )
        rt = max(
            (
                (r2(scores_de[:, index], rde @ solve(rgram, rtr.T @ (target - target.mean()), a) + target.mean()), a)
                for a in RETRIEVAL_ALPHAS
            )
        )
        ngc = solve(gram, ztr.T @ (smoothed - smoothed.mean()), ng[1])
        rtc = solve(rgram, rtr.T @ (target - target.mean()), rt[1])
        heads[f"score_ng_{index}_coef"] = ngc
        heads[f"score_rt_{index}_coef"] = rtc
        heads[f"score_ng_{index}_bias"] = np.array(smoothed.mean())
        heads[f"score_{index}_bias"] = np.array(target.mean())
        blended = 0.8 * (rde @ rtc + target.mean()) + 0.2 * (zde @ ngc + smoothed.mean())
        report[f"score_{model_id}"] = round(r2(scores_de[:, index], blended), 4)

    advantage = scores_tr[:, 2] - np.maximum(scores_tr[:, 0], scores_tr[:, 1])
    gate_target = (advantage > 0.01).astype(float)
    gbest = None
    for a in ALPHAS:
        coef = solve(gram, ztr.T @ (gate_target - gate_target.mean()), a)
        prediction = zde @ coef + gate_target.mean()
        actual = (scores_de[:, 2] - np.maximum(scores_de[:, 0], scores_de[:, 1])) > 0.01
        pos, neg = prediction[actual], prediction[~actual]
        auc = (pos[:, None] > neg[None, :]).mean()
        if gbest is None or auc > gbest[0]:
            gbest = (auc, coef)
    heads["gate_coef"] = gbest[1]
    heads["gate_bias"] = np.array(gate_target.mean())
    report["gate_auc"] = round(float(gbest[0]), 4)

    np.savez(ARTIFACTS / "heads.npz", **heads)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
