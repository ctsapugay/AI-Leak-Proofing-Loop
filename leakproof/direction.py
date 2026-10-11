"""Steering direction v̂ per layer: the normalized difference of means, leak minus safe.

    .venv/bin/python -m leakproof.direction runs/collect/<default run> runs/collect/<strict run>

For every layer it reports, under the same template-grouped 5-fold CV as the probe:
  - how well projecting onto v̂ alone (fit on the train folds) separates leak vs safe
    (AUROC), next to random unit directions scored the same way (the control)
  - cosine(v̂, probe w), with w the probe's weights mapped back to activation units

It also fixes the held-out split shared by M3-M4 (two folds, ~40% of templates, saved
to split.json) and saves v̂ computed on the remaining train templates only.

Writes runs/steer/: vhat.npy [layers, hidden] (train templates), split.json, direction.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from leakproof.data import ROOT
from leakproof.probe import load_runs, make_probe, splits, template_of

STEER_DIR = ROOT / "runs" / "steer"
N_RANDOM = 20


def diff_of_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit vector mean(X[y == 1]) - mean(X[y == 0]); works on [n, hidden] or [n, layers, hidden]."""
    v = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def projection_auroc(X: np.ndarray, y: np.ndarray, v: np.ndarray) -> float:
    return float(roc_auc_score(y, X @ v))


def random_units(dim: int, n: int = N_RANDOM, seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed).normal(size=(n, dim))
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def holdout_split(y: np.ndarray, groups: np.ndarray, n_folds: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the test templates of the first `n_folds` CV folds (~40% of templates)."""
    test = np.concatenate([te for _, te in splits(y, groups)[:n_folds]])
    train = np.setdiff1d(np.arange(len(y)), test)
    return train, np.sort(test)


def probe_direction(X: np.ndarray, y: np.ndarray, C: float) -> np.ndarray:
    """Probe weights in activation units (undo the scaler), unit length."""
    probe = make_probe(C).fit(X, y)
    w = probe[1].coef_[0] / probe[0].scale_
    return w / np.linalg.norm(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--C", type=float, default=1.0, help="probe C (from M1 summary.json)")
    args = ap.parse_args()

    acts, index = load_runs(args.runs)
    kind = np.array([r["kind"] for r in index])
    label = np.array([r["label"] for r in index])
    rows = (kind == "attack") & np.isin(label, ["full", "none"])
    A = acts["answer_mean"][rows].astype(np.float32)  # [n, layers, hidden]
    y = (label[rows] == "full").astype(int)
    groups = np.array([template_of(r["id"]) for r in index])[rows]

    train, test = holdout_split(y, groups)
    test_templates = sorted(str(g) for g in set(groups[test]))
    vhat = diff_of_means(A[train], y[train])  # [layers, hidden]
    rand = random_units(A.shape[2])
    folds = splits(y, groups)

    lines = ["layer,cv_auroc_mean,cv_auroc_std,random_cv_mean,random_cv_p95,cos_vhat_probe"]
    for layer in range(A.shape[1]):
        X = A[:, layer]
        cv, rnd = [], []
        for tr, te in folds:
            if len(set(y[te])) < 2:
                continue
            cv.append(projection_auroc(X[te], y[te], diff_of_means(X[tr], y[tr])))
            rnd += [projection_auroc(X[te], y[te], u) for u in rand]
        cos = float(vhat[layer] @ probe_direction(X[train], y[train], args.C)) if layer else float("nan")
        lines.append(f"{layer},{np.mean(cv):.4f},{np.std(cv):.4f},{np.mean(rnd):.4f},"
                     f"{np.percentile(rnd, 95):.4f},{cos:.4f}")
        print(f"  layer {layer:>2}: v̂ CV AUROC {np.mean(cv):.3f} ± {np.std(cv):.3f} | random CV mean "
              f"{np.mean(rnd):.3f} p95 {np.percentile(rnd, 95):.3f} | cos(v̂, w) {cos:.3f}", flush=True)

    STEER_DIR.mkdir(parents=True, exist_ok=True)
    np.save(STEER_DIR / "vhat.npy", vhat.astype(np.float32))
    (STEER_DIR / "direction.csv").write_text("\n".join(lines) + "\n")
    (STEER_DIR / "split.json").write_text(json.dumps({
        "runs": [str(r) for r in args.runs], "test_templates": test_templates,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_test_full": int(y[test].sum()), "n_test_none": int((1 - y[test]).sum()),
    }, indent=2) + "\n")
    print(f"held-out templates: {test_templates}\nsaved to {STEER_DIR}")


if __name__ == "__main__":
    main()
