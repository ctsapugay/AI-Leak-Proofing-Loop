"""Exploratory salary-only control test (teacher-forced, M3). Doesn't change any milestone.

Directions at L16-20 (train side of runs/steer/split_byfield.json, answer_mean):
  salary_all      mean(salary full) - mean(salary safe)
  salary_matched  within-template differences, averaged over salary train templates that have
                  both leaks and refusals (answer style cancels inside a template)
Controls:
  random          one seeded random unit direction per layer
  varmatch{0,1,2} principal components of the activations whose projection variance is closest
                  to salary_all's (|cos| with salary_all < 0.3): "big" directions like v̂

Measure (forward passes only): mean log-prob of the salary tokens in held-out salary full-leak
answers, and mean per-token log-prob of benign half-B answers, for alpha in {0.5, 1, 2}.

    .venv/bin/python scripts/salary_control.py runs/collect/<default> runs/collect/<strict>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from leakproof.data import load_employees  # noqa: E402
from leakproof.direction import benign_halves  # noqa: E402
from leakproof.probe import load_runs, template_of  # noqa: E402
from perfield_control import LAYERS, SPLIT_PATH, multi_project, secret_span  # noqa: E402

ALPHAS = (0.5, 1.0, 2.0)


def unit(v):
    return v / np.linalg.norm(v)


def directions(runs):
    acts, index = load_runs(runs)
    kind = np.array([r["kind"] for r in index]); label = np.array([r["label"] for r in index])
    field = np.array([r["field"] for r in index]); tmpl = np.array([template_of(r["id"]) for r in index])
    split = json.loads(SPLIT_PATH.read_text())
    train_t = split["by_field"]["salary"]["train_templates"]
    sal = (kind == "attack") & (field == "salary") & np.isin(label, ["full", "none"]) & np.isin(tmpl, train_t)
    A = acts["answer_mean"].astype(np.float32)
    matched_t = [t for t in train_t if ((sal & (tmpl == t) & (label == "full")).any()
                                       and (sal & (tmpl == t) & (label == "none")).any())]
    info = {"train_templates": train_t, "matched_templates": {
        t: {"full": int((sal & (tmpl == t) & (label == "full")).sum()),
            "none": int((sal & (tmpl == t) & (label == "none")).sum())} for t in matched_t}}
    out = defaultdict(dict)
    rng = np.random.default_rng(0)
    for L in LAYERS:
        X = A[:, L]
        d_all = unit(X[sal & (label == "full")].mean(0) - X[sal & (label == "none")].mean(0))
        diffs = [X[sal & (tmpl == t) & (label == "full")].mean(0) - X[sal & (tmpl == t) & (label == "none")].mean(0)
                 for t in matched_t]
        d_matched = unit(np.mean(diffs, axis=0))
        # Activity-matched controls: PCs (over all answers) with variance closest to d_all's.
        Xc = X - X.mean(0)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pc_var = S ** 2 / len(X)
        target = float(np.var(Xc @ d_all))
        ok = [i for i in range(len(pc_var)) if abs(Vt[i] @ d_all) < 0.3 and abs(Vt[i] @ d_matched) < 0.3]
        pick = sorted(ok, key=lambda i: abs(pc_var[i] - target))[:3]
        out[L] = {"salary_all": d_all, "salary_matched": d_matched, "random": unit(rng.normal(size=X.shape[1])),
                  **{f"varmatch{j}": Vt[i] for j, i in enumerate(pick)}}
        info.setdefault("per_layer", {})[L] = {
            "cos_all_matched": float(d_all @ d_matched), "var_salary_all": target,
            "var_salary_matched": float(np.var(Xc @ d_matched)), "var_random": float(np.var(Xc @ out[L]["random"])),
            "varmatch_pc_index": pick, "varmatch_var": [float(pc_var[i]) for i in pick]}
    return out, info


def main():
    import torch
    from leakproof.hf_target import HFBackend
    from leakproof.target import build_system_prompt

    runs = [Path(p) for p in sys.argv[1:]]
    dirs, info = directions(runs)
    split = json.loads(SPLIT_PATH.read_text())
    test_t = set(split["by_field"]["salary"]["test_templates"])
    by_id = {e["id"]: e for e in load_employees()}
    rows = [json.loads(l) for r in runs for l in open(r / "answers.jsonl")]
    _, half_b = benign_halves([r["id"] for r in rows if r["kind"] == "benign"])
    leaks = [r for r in rows if r["kind"] == "attack" and r["label"] == "full" and r["field"] == "salary"
             and template_of(r["id"]) in test_t]
    benign = [r for r in rows if r["kind"] == "benign" and r["id"] in half_b]

    b = HFBackend()
    b.capture_acts = False
    tok, model = b.tok, b.model

    def example(r):
        msgs = [{"role": "system", "content": build_system_prompt([by_id[i] for i in r["context_ids"]], r["strictness"])}]
        for t, (turn, reply) in enumerate(zip(r["turns"], r["replies"])):
            prompt = tok.apply_chat_template(msgs + [{"role": "user", "content": turn}], add_generation_prompt=True,
                                             tokenize=False)
            if r["kind"] == "benign":
                if t == len(r["turns"]) - 1:
                    return prompt + reply, (len(prompt), len(prompt) + len(reply))
            else:
                span = secret_span(reply, by_id[r["employee_id"]], "salary")
                if span:
                    return prompt + reply[:span[1]], (len(prompt) + span[0], len(prompt) + span[1])
            msgs += [{"role": "user", "content": turn}, {"role": "assistant", "content": reply}]
        return None

    @torch.no_grad()
    def mean_logprob(text, span):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offs = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to("cuda")
        pos = [i for i, (s, e) in enumerate(offs) if e > span[0] and s < span[1] and i > 0]
        logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
        return float(np.mean([logp[i - 1, ids[0, i]].item() for i in pos])) if pos else None

    names = ["salary_all", "salary_matched", "random", "varmatch0", "varmatch1", "varmatch2"]
    conds = {"none": None}
    for name in names:
        for a in ALPHAS:
            conds[f"{name}_a{a:g}"] = ({L: torch.from_numpy(np.asarray(dirs[L][name], dtype=np.float32)[None])
                                        for L in LAYERS}, a)
    res = {"leak": defaultdict(list), "benign": defaultdict(list)}
    skipped = 0
    for group, items in (("leak", leaks), ("benign", benign)):
        for r in items:
            ex = example(r)
            if ex is None:
                skipped += 1
                continue
            scores = {}
            for name, cfg in conds.items():
                handles = multi_project(model, *cfg) if cfg else []
                try:
                    scores[name] = mean_logprob(*ex)
                finally:
                    for h in handles:
                        h.remove()
            if any(v is None for v in scores.values()):
                skipped += 1
                continue
            for name, v in scores.items():
                res[group][name].append(v)
    summary = {"info": info, "skipped": skipped, "n_leak": len(res["leak"]["none"]), "n_benign": len(res["benign"]["none"])}
    for name in conds:
        if name == "none":
            continue
        dl = np.array(res["leak"][name]) - np.array(res["leak"]["none"])
        db = np.array(res["benign"][name]) - np.array(res["benign"]["none"])
        summary[name] = {"secret_delta": float(dl.mean()), "secret_prob_factor": float(np.exp(dl.mean())),
                         "benign_delta_per_token": float(db.mean()), "benign_prob_factor_per_token": float(np.exp(db.mean())),
                         "selectivity": float(dl.mean() - db.mean())}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
