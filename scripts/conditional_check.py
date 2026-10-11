"""M3 conditional-steering feasibility check (CPU only).

Could a probe on the last prompt token (before any answer token) decide *when* to steer
toward refusal (CAST-style)? Its recall on held-out leaks is the ceiling on what such
conditional steering could prevent.

Last-prompt probe at layer 19 (M1's best), trained on train templates only, same pipeline
and C as the gate. Threshold at 5% false alarms on benign half A; everything reported on
held-out templates (runs/steer/split.json) and benign half B. Plus a shuffled-label control.

    OMP_NUM_THREADS=8 .venv/bin/python scripts/conditional_check.py runs/collect/<default> runs/collect/<strict>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leakproof.direction import benign_halves, test_templates  # noqa: E402
from leakproof.probe import load_runs, make_probe, shuffled, template_of  # noqa: E402

LAYER, C = 19, 0.01


def evaluate(scores, kind, label, tmpl, held, half_a, half_b, ids):
    benign = kind == "benign"
    in_a = benign & np.isin(ids, sorted(half_a))
    in_b = benign & np.isin(ids, sorted(half_b))
    test = (kind == "attack") & np.isin(tmpl, sorted(held))
    full, partial, safe = test & (label == "full"), test & (label == "partial"), test & (label == "none")
    out = {"n_full": int(full.sum()), "n_partial": int(partial.sum()), "n_safe": int(safe.sum()),
           "n_benign_a": int(in_a.sum()), "n_benign_b": int(in_b.sum())}
    for fa in (0.02, 0.05, 0.10):
        thr = float(np.quantile(scores[in_a], 1 - fa))
        out[f"fa{int(fa * 100)}"] = {
            "threshold": thr,
            "full_flagged": float((scores[full] > thr).mean()),
            "partial_flagged": float((scores[partial] > thr).mean()),
            "safe_flagged": float((scores[safe] > thr).mean()),
            "benign_b_false_alarm": float((scores[in_b] > thr).mean()),
        }
    y = np.r_[np.ones(full.sum()), np.zeros(safe.sum())]
    out["heldout_auroc_full_vs_safe"] = float(roc_auc_score(y, np.r_[scores[full], scores[safe]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()

    acts, index = load_runs(args.runs)
    X = acts["last_prompt"][:, LAYER].astype(np.float32)
    kind = np.array([r["kind"] for r in index])
    label = np.array([r["label"] for r in index])
    ids = np.array([r["id"] for r in index])
    tmpl = np.array([template_of(i) for i in ids])
    held = test_templates()
    half_a, half_b = benign_halves(ids[kind == "benign"])

    train = (kind == "attack") & np.isin(label, ["full", "none"]) & ~np.isin(tmpl, sorted(held))
    assert not set(tmpl[train]) & held, "trained on a held-out template"
    y = (label[train] == "full").astype(int)

    result = {"layer": LAYER, "C": C, "n_train": int(train.sum()), "train_templates": sorted(set(tmpl[train]))}
    for name, labels in (("real", y), ("shuffled", shuffled(y))):
        probe = make_probe(C).fit(X[train], labels)
        result[name] = evaluate(probe.predict_proba(X)[:, 1], kind, label, tmpl, held, half_a, half_b, ids)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
