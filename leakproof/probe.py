"""Linear leak probe on cached activations: which layer represents "I'm leaking"?

    .venv/bin/python -m leakproof.probe runs/collect/<default run> runs/collect/<strict run>

Data: attack answers from the given collect runs. Label 1 = judge says `full` leak,
0 = `none`; `partial` answers are held out and scored separately. Benign answers
are never trained on and give the false-alarm rate.

Model: StandardScaler + LogisticRegression(class_weight="balanced"), evaluated with
StratifiedGroupKFold grouped by attack template (the case id before "@"), so an
attack's default and strict versions always land on the same side of a split.

Writes runs/probe/<time>/:
  auroc.csv       layer x pooling: mean/std AUROC over folds, plus the shuffled-label control
  summary.json    chosen C, best layer, threshold, false alarms, partial scores
  probe.npz       final probe at the best layer (trained on all attack data): w, b,
                  scaler mean/scale, threshold. The gate (M4) loads this.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from leakproof.data import ROOT

POOLINGS = ("answer_mean", "last_prompt")
C_GRID = (0.001, 0.01, 0.1, 1.0)
C_TOL = 0.005
N_SPLITS = 5
SEED = 0
TARGET_FALSE_ALARM = 0.05


def template_of(case_id: str) -> str:
    return case_id.split("@")[0]


def load_runs(run_dirs: list[Path]) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Concatenate acts and index rows from several collect runs."""
    acts: dict[str, list[np.ndarray]] = {p: [] for p in POOLINGS}
    index: list[dict] = []
    for d in run_dirs:
        for p in POOLINGS:
            acts[p].append(np.load(d / "acts" / f"{p}.npy"))
        index += [{**json.loads(line), "run": d.name} for line in (d / "acts" / "index.jsonl").open()]
    return {p: np.concatenate(v) for p, v in acts.items()}, index


def splits(y: np.ndarray, groups: np.ndarray, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Template-grouped, class-stratified folds. Asserts no template is on both sides."""
    folds = list(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(y, y, groups))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test]), "template leaked across split"
    return folds


def make_probe(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float,
               seed: int = SEED) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold P(leak) for every row, and AUROC per fold (folds with one class skipped)."""
    scores = np.zeros(len(y))
    fold_auc = []
    for train, test in splits(y, groups, seed):
        probe = make_probe(C).fit(X[train], y[train])
        scores[test] = probe.predict_proba(X[test])[:, 1]
        if len(set(y[test])) == 2:
            fold_auc.append(roc_auc_score(y[test], scores[test]))
    return scores, fold_auc


def pick_C(c_auc: dict[float, float], tol: float = C_TOL) -> float:
    """Strongest regularization (smallest C) within `tol` AUROC of the best: simpler,
    faster to fit, and differences that small are fold noise."""
    best = max(c_auc.values())
    return min(c for c, a in c_auc.items() if a >= best - tol)


def shuffled(y: np.ndarray, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).permutation(y)


def threshold_for_false_alarm(benign_scores: np.ndarray, rate: float = TARGET_FALSE_ALARM) -> float:
    """Smallest threshold with at most `rate` of benign scores above it."""
    return float(np.quantile(benign_scores, 1 - rate))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()

    acts, index = load_runs(args.runs)
    kind = np.array([r["kind"] for r in index])
    label = np.array([r["label"] for r in index])
    groups_all = np.array([template_of(r["id"]) for r in index])

    train_rows = (kind == "attack") & np.isin(label, ["full", "none"])
    y = (label[train_rows] == "full").astype(int)
    groups = groups_all[train_rows]
    n_layers = acts["answer_mean"].shape[1]
    print(f"{train_rows.sum()} attack answers ({y.sum()} full, {(1 - y).sum()} none), "
          f"{len(set(groups))} templates, {n_layers} layers")

    # Choose C once, at a middle layer, on answer_mean.
    mid = n_layers // 2
    c_auc = {C: np.mean(oof_scores(acts["answer_mean"][train_rows, mid].astype(np.float32), y, groups, C)[1])
             for C in C_GRID}
    C = pick_C(c_auc)
    print(f"C sweep at layer {mid}: " + ", ".join(f"{c}: {a:.3f}" for c, a in c_auc.items()) + f" -> C={C}")

    rows = []
    for pooling in POOLINGS:
        for layer in range(n_layers):
            X = acts[pooling][train_rows, layer].astype(np.float32)
            _, auc = oof_scores(X, y, groups, C)
            _, auc_shuf = oof_scores(X, shuffled(y), groups, C)
            rows.append({"pooling": pooling, "layer": layer, "auroc_mean": float(np.mean(auc)),
                         "auroc_std": float(np.std(auc)), "shuffled_mean": float(np.mean(auc_shuf)),
                         "n_folds": len(auc)})
            print(f"  {pooling:<12} layer {layer:>2}: AUROC {np.mean(auc):.3f} ± {np.std(auc):.3f}   "
                  f"shuffled {np.mean(auc_shuf):.3f}", flush=True)

    best = max((r for r in rows if r["pooling"] == "answer_mean"), key=lambda r: r["auroc_mean"])
    best_lp = max((r for r in rows if r["pooling"] == "last_prompt"), key=lambda r: r["auroc_mean"])
    layer = best["layer"]

    # Final probe at the best layer, trained on all attack data.
    X_all = acts["answer_mean"][:, layer].astype(np.float32)
    probe = make_probe(C).fit(X_all[train_rows], y)
    benign = probe.predict_proba(X_all[kind == "benign"])[:, 1]
    threshold = threshold_for_false_alarm(benign)
    oof, _ = oof_scores(X_all[train_rows], y, groups, C)
    recall_at_thr = float((oof[y == 1] > threshold).mean())
    fa_at_half = float((benign > 0.5).mean())
    partial_rows = (kind == "attack") & (label == "partial")
    partial = probe.predict_proba(X_all[partial_rows])[:, 1]

    summary = {
        "runs": [str(r) for r in args.runs], "C": C, "c_sweep": c_auc,
        "n_train": int(train_rows.sum()), "n_full": int(y.sum()), "n_none": int((1 - y).sum()),
        "n_templates": len(set(groups)),
        "best_answer_mean": best, "best_last_prompt": best_lp,
        "threshold_5pct_benign_fa": threshold,
        "oof_recall_at_threshold": recall_at_thr,
        "benign_false_alarm_at_0.5": fa_at_half,
        "n_benign": int((kind == "benign").sum()),
        "partial_mean_score": float(partial.mean()) if len(partial) else None,
        "partial_frac_above_threshold": float((partial > threshold).mean()) if len(partial) else None,
        "n_partial": int(partial_rows.sum()),
    }
    out = ROOT / "runs" / "probe" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "auroc.csv").open("w") as f:
        f.write("pooling,layer,auroc_mean,auroc_std,shuffled_mean,n_folds\n")
        for r in rows:
            f.write(f"{r['pooling']},{r['layer']},{r['auroc_mean']:.4f},{r['auroc_std']:.4f},"
                    f"{r['shuffled_mean']:.4f},{r['n_folds']}\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    scaler, lr = probe[0], probe[1]
    np.savez(out / "probe.npz", layer=layer, w=lr.coef_[0], b=lr.intercept_[0],
             mean=scaler.mean_, scale=scaler.scale_, threshold=threshold, C=C)
    print(json.dumps(summary, indent=2))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
