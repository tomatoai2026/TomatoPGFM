from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def metrics_binary(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

    y_pred = (y_score >= 0.5).astype(int)
    out = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if len(set(y_true.tolist())) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


def grouped_splits(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x_dummy = np.zeros((len(y), 1))
    return list(splitter.split(x_dummy, y, groups))


def paired_bootstrap_delta(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, repeats: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y_true)
    for _ in range(repeats):
        idx = rng.integers(0, n, size=n)
        ma = metrics_binary(y_true[idx], score_a[idx])["mcc"]
        mb = metrics_binary(y_true[idx], score_b[idx])["mcc"]
        deltas.append(ma - mb)
    arr = np.array(deltas)
    return {"delta_mean": float(arr.mean()), "ci95_low": float(np.quantile(arr, 0.025)), "ci95_high": float(np.quantile(arr, 0.975))}


def bh_fdr(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        p = p_values[idx]
        val = min(prev, p * m / (m - rank + 1))
        q[idx] = val
        prev = val
    return q.tolist()


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

