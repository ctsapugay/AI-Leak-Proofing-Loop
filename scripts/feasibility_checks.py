"""M3 feasibility checks (CPU only) on the cached M0 activations.

A2. Last-prompt probe (L19) with benign half A as extra negatives: does it stop flagging
    benign questions and catch more leaks before the answer starts?
B.  Per-field leak directions (salary / medical_leave / performance_note):
    B1 per-field v̂ at layers 16-20 and their cosines (with each other, with pooled v̂)
    B2 held-out AUROC of each field's v̂ vs pooled v̂ vs random directions
    B3 4-way last-prompt classifier at L19 (none / salary / medical / performance)

Everything is fit on train templates only and reported on held-out templates
(runs/steer/split.json) + benign half B.

    OMP_NUM_THREADS=8 .venv/bin/python scripts/feasibility_checks.py runs/collect/<default> runs/collect/<strict>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leakproof.direction import benign_halves, diff_of_means, random_units, test_templates  # noqa: E402
from leakproof.probe import load_runs, make_probe, shuffled, template_of  # noqa: E402

LP_LAYER, C = 19, 0.01
STEER_LAYERS = [16, 17, 18, 19, 20]
FIELDS = ("salary", "medical_leave", "performance_note")
FAS = (0.02, 0.05, 0.10)
MIN_CLASS = 10  # fewer rows than this in a class -> flagged as too small to trust


def flagged_table(scores, thr_scores_a, masks):
    out = {}
    for fa in FAS:
        thr = float(np.quantile(thr_scores_a, 1 - fa))
        out[f"fa{int(fa * 100)}"] = {"threshold": thr, **{k: float((scores[m] > thr).mean()) for k, m in masks.items()}}
    return out


def check_a2(X, kind, label, tmpl, ids, held, half_a, half_b):
    benign_a = (kind == "benign") & np.isin(ids, sorted(half_a))
    benign_b = (kind == "benign") & np.isin(ids, sorted(half_b))
    train_atk = (kind == "attack") & np.isin(label, ["full", "none"]) & ~np.isin(tmpl, sorted(held))
    train = train_atk | benign_a
    assert not set(tmpl[train_atk]) & held
    y = ((kind == "attack") & (label == "full"))[train].astype(int)
    test = (kind == "attack") & np.isin(tmpl, sorted(held))
    masks = {"full_flagged": test & (label == "full"), "partial_flagged": test & (label == "partial"),
             "safe_flagged": test & (label == "none"), "benign_b_false_alarm": benign_b}
    out = {"n_train": int(train.sum()), "n_train_pos": int(y.sum()), "n_train_benign_neg": int(benign_a.sum()),
           "n_train_safe_neg": int((train_atk & (label == "none")).sum()),
           **{f"n_{k.split('_')[0]}": int(m.sum()) for k, m in masks.items()}}
    for name, labels in (("real", y), ("shuffled", shuffled(y))):
        probe = make_probe(C).fit(X[train], labels)
        s = probe.predict_proba(X)[:, 1]
        res = {"rule_threshold_on_benign_a": flagged_table(s, s[benign_a], masks)}
        # Benign A is training data here, so its in-sample scores are optimistic. Cross-fit:
        # score each benign-A question with a probe that didn't see it (5 folds by question).
        oof = np.zeros(len(s))
        a_idx = np.where(benign_a)[0]
        for tr_a, te_a in GroupKFold(5).split(a_idx, groups=ids[a_idx]):
            tr = train.copy()
            tr[a_idx[te_a]] = False
            p = make_probe(C).fit(X[tr], ((kind == "attack") & (label == "full"))[tr].astype(int)
                                  if name == "real" else shuffled(((kind == "attack") & (label == "full"))[tr].astype(int)))
            oof[a_idx[te_a]] = p.predict_proba(X[a_idx[te_a]])[:, 1]
        res["crossfit_threshold_on_benign_a"] = flagged_table(s, oof[benign_a], masks)
        full, safe = masks["full_flagged"], masks["safe_flagged"]
        res["heldout_auroc_full_vs_safe"] = float(roc_auc_score(np.r_[np.ones(full.sum()), np.zeros(safe.sum())],
                                                                np.r_[s[full], s[safe]]))
        res["heldout_auroc_full_vs_benign_b"] = float(roc_auc_score(np.r_[np.ones(full.sum()), np.zeros(benign_b.sum())],
                                                                    np.r_[s[full], s[benign_b]]))
        out[name] = res
    return out


def check_b(A, kind, label, field, tmpl, held):
    atk = (kind == "attack") & np.isin(label, ["full", "none"])
    train, test = atk & ~np.isin(tmpl, sorted(held)), atk & np.isin(tmpl, sorted(held))
    y = (label == "full").astype(int)
    rand = random_units(A.shape[2])
    out = {"counts": {}, "layers": {}}
    for f in FIELDS + ("all",):
        m = field == f
        out["counts"][f] = {"train_full": int((train & m & (y == 1)).sum()), "train_none": int((train & m & (y == 0)).sum()),
                            "test_full": int((test & m & (y == 1)).sum()), "test_none": int((test & m & (y == 0)).sum())}
        out["counts"][f]["too_small"] = min(out["counts"][f].values()) < MIN_CLASS
    for L in STEER_LAYERS:
        X = A[:, L]
        pooled = diff_of_means(X[train], y[train])
        vf = {f: diff_of_means(X[train & (field == f)], y[train & (field == f)])
              for f in FIELDS if out["counts"][f]["train_full"] and out["counts"][f]["train_none"]}
        row = {"cos_with_pooled": {f: float(v @ pooled) for f, v in vf.items()},
               "cos_pairwise": {f"{a}|{b}": float(vf[a] @ vf[b]) for i, a in enumerate(vf) for b in list(vf)[i + 1:]},
               "heldout_auroc": {}}
        for f in vf:
            m = test & (field == f)
            if len(set(y[m])) < 2:
                row["heldout_auroc"][f] = "n/a (held-out rows of this field have one class or none)"
                continue
            r = [roc_auc_score(y[m], X[m] @ u) for u in rand]
            row["heldout_auroc"][f] = {"field_vhat": float(roc_auc_score(y[m], X[m] @ vf[f])),
                                       "pooled_vhat": float(roc_auc_score(y[m], X[m] @ pooled)),
                                       "random_mean": float(np.mean(r)), "random_max": float(np.max(r))}
        out["layers"][L] = row
    return out


def check_b3(X, kind, label, field, tmpl, held):
    cls = np.full(len(label), "", dtype=object)
    atk = kind == "attack"
    cls[atk & (label == "none")] = "none"
    for f in FIELDS:
        cls[atk & (label == "full") & (field == f)] = f
    use = cls != ""  # partial leaks and field="all" leaks are left out (no single field)
    train, test = use & ~np.isin(tmpl, sorted(held)), use & np.isin(tmpl, sorted(held))
    classes = ["none", *FIELDS]
    probe = make_probe(C).fit(X[train], cls[train])
    pred = probe.predict(X[test])
    train_major = max(classes, key=lambda c: (cls[train] == c).sum())
    test_major = max(classes, key=lambda c: (cls[test] == c).sum())
    return {"n_train": {c: int((cls[train] == c).sum()) for c in classes},
            "n_test": {c: int((cls[test] == c).sum()) for c in classes},
            "accuracy": float((pred == cls[test]).mean()),
            "baseline_train_majority": {"class": train_major, "accuracy": float((cls[test] == train_major).mean())},
            "baseline_test_majority": {"class": test_major, "accuracy": float((cls[test] == test_major).mean())},
            "confusion_rows_true_cols_pred": {"labels": classes,
                                              "matrix": confusion_matrix(cls[test], pred, labels=classes).tolist()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    acts, index = load_runs(args.runs)
    kind = np.array([r["kind"] for r in index])
    label = np.array([r["label"] for r in index])
    field = np.array([r["field"] for r in index])
    ids = np.array([r["id"] for r in index])
    tmpl = np.array([template_of(i) for i in ids])
    held = test_templates()
    half_a, half_b = benign_halves(ids[kind == "benign"])
    lp = acts["last_prompt"][:, LP_LAYER].astype(np.float32)
    result = {
        "A2_last_prompt_with_benign_negatives": check_a2(lp, kind, label, tmpl, ids, held, half_a, half_b),
        "B1_B2_per_field_directions": check_b(acts["answer_mean"].astype(np.float32), kind, label, field, tmpl, held),
        "B3_last_prompt_4way": check_b3(lp, kind, label, field, tmpl, held),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
