"""Exploratory per-field control test (M3). Doesn't change any milestone.

  split    (CPU) template split stratified by field -> runs/steer/split_byfield.json; per-field
           v̂ (full - safe, train side, answer_mean, L16-20), Gram-Schmidt per layer, pooled v̂
           on the same rows -> runs/steer/byfield_dirs.npz
  teacher  (GPU, forward passes only) on held-out full-leak answers: mean log-prob of the
           secret's tokens (first occurrence in the reply) with no hook / pooled v̂ removed /
           three field v̂ removed / three random orthonormal directions removed, at L16-20,
           alpha 1 and 2. Side effect: change in mean per-token log-prob of benign half-B answers.

    OMP_NUM_THREADS=8 .venv/bin/python scripts/perfield_control.py split runs/collect/<default> runs/collect/<strict>
    .venv/bin/python scripts/perfield_control.py teacher runs/collect/<default> runs/collect/<strict>
"""

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from leakproof.data import ROOT, load_employees  # noqa: E402
from leakproof.direction import benign_halves, diff_of_means  # noqa: E402
from leakproof.probe import load_runs, template_of  # noqa: E402

FIELDS = ("salary", "medical_leave", "performance_note")
LAYERS = [16, 17, 18, 19, 20]
SPLIT_PATH = ROOT / "runs" / "steer" / "split_byfield.json"
DIRS_PATH = ROOT / "runs" / "steer" / "byfield_dirs.npz"


# --- split + directions (CPU) ------------------------------------------------------

def choose_heldout(counts: dict[str, tuple[int, int]], frac: float = 0.4) -> list[str]:
    """Templates to hold out for one field: the subset of ~frac of them that maximizes the
    smallest of (train full, train safe, test full, test safe). Deterministic."""
    names = sorted(counts)
    k = max(1, round(frac * len(names)))
    best, best_key = None, None
    for test in itertools.combinations(names, k):
        tf = sum(counts[t][0] for t in test); tn = sum(counts[t][1] for t in test)
        af = sum(c[0] for c in counts.values()); an = sum(c[1] for c in counts.values())
        key = (min(af - tf, an - tn, tf, tn), min(tn, an - tn))  # scarcest cell, then safe balance
        if best_key is None or key > best_key:
            best, best_key = list(test), key
    return best


def gram_schmidt(V: np.ndarray) -> np.ndarray:
    """Rows of V -> orthonormal rows, in order."""
    out = []
    for v in V:
        w = v - sum((v @ u) * u for u in out)
        out.append(w / np.linalg.norm(w))
    return np.stack(out)


def do_split(runs):
    acts, index = load_runs(runs)
    kind = np.array([r["kind"] for r in index]); label = np.array([r["label"] for r in index])
    field = np.array([r["field"] for r in index]); tmpl = np.array([template_of(r["id"]) for r in index])
    atk = (kind == "attack") & np.isin(label, ["full", "none"]) & np.isin(field, FIELDS)
    held, counts_out = [], {}
    for f in FIELDS:
        m = atk & (field == f)
        counts = {t: (int((m & (tmpl == t) & (label == "full")).sum()), int((m & (tmpl == t) & (label == "none")).sum()))
                  for t in sorted(set(tmpl[m]))}
        test = choose_heldout(counts)
        held += test
        tr = [t for t in counts if t not in test]
        counts_out[f] = {"train_templates": tr, "test_templates": test,
                         "train_full": sum(counts[t][0] for t in tr), "train_safe": sum(counts[t][1] for t in tr),
                         "test_full": sum(counts[t][0] for t in test), "test_safe": sum(counts[t][1] for t in test)}
    train = atk & ~np.isin(tmpl, held)
    y = (label == "full").astype(int)
    A = acts["answer_mean"].astype(np.float32)
    raw = {f: diff_of_means(A[train & (field == f)][:, LAYERS], y[train & (field == f)]) for f in FIELDS}  # [5, H]
    pooled = diff_of_means(A[train][:, LAYERS], y[train])
    ortho = np.stack([gram_schmidt(np.stack([raw[f][i] for f in FIELDS])) for i in range(len(LAYERS))])  # [5, 3, H]
    cos = {i: {"pairwise_raw": {f"{a}|{b}": float(raw[a][i] @ raw[b][i]) for a, b in itertools.combinations(FIELDS, 2)},
               "with_pooled_raw": {f: float(raw[f][i] @ pooled[i]) for f in FIELDS}} for i in range(len(LAYERS))}
    SPLIT_PATH.write_text(json.dumps({"test_templates": sorted(held), "by_field": counts_out,
                                      "runs": [str(r) for r in runs]}, indent=2) + "\n")
    np.savez(DIRS_PATH, layers=LAYERS, fields=FIELDS, field_ortho=ortho, pooled=pooled,
             field_raw=np.stack([raw[f] for f in FIELDS], axis=1))
    print(json.dumps({"by_field": counts_out, "cosines_by_layer": {LAYERS[i]: c for i, c in cos.items()}}, indent=2))


# --- teacher-forced log-probs (GPU) ------------------------------------------------

def secret_span(reply: str, emp: dict, fld: str):
    """(start, end) of the first occurrence of this employee's secret for `fld` in reply."""
    if fld == "salary":
        s = emp["secrets"]["salary"]
        pats = [re.escape(f"${s:,}"), re.escape(f"{s:,}"), rf"(?<!\d){s}(?!\d)"]
    else:
        pats = [re.escape(t) for t in emp.get("leak_terms", {}).get(fld, [])]
    hits = [m for p in pats for m in [re.search(p, reply, re.I)] if m]
    if not hits:
        return None
    m = min(hits, key=lambda m: m.start())
    return m.start(), m.end()


def multi_project(model, dirs: dict[int, "torch.Tensor"], alpha: float):
    """Hooks removing alpha * projection onto the orthonormal rows of dirs[L] at hidden index L."""
    import torch
    handles = []
    for L, V in dirs.items():
        V = V.to(model.device, torch.float32)

        def hook(_m, _i, out, V=V):
            h = out[0] if isinstance(out, tuple) else out
            hf = h.float()
            new = (hf - alpha * (hf @ V.T) @ V).to(h.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        handles.append(model.model.layers[L - 1].register_forward_hook(hook, prepend=True))
    return handles


def do_teacher(runs):
    import torch
    from leakproof.hf_target import HFBackend
    from leakproof.target import build_system_prompt

    split = json.loads(SPLIT_PATH.read_text())
    held = set(split["test_templates"])
    d = np.load(DIRS_PATH)
    by_id = {e["id"]: e for e in load_employees()}
    rows = [json.loads(l) for r in runs for l in open(r / "answers.jsonl")]
    benign_ids = [r["id"] for r in rows if r["kind"] == "benign"]
    _, half_b = benign_halves(benign_ids)

    b = HFBackend()
    b.capture_acts = False
    tok, model = b.tok, b.model
    g = torch.Generator().manual_seed(0)
    rand = {L: torch.linalg.qr(torch.randn(d["pooled"].shape[1], 3, generator=g))[0].T for L in LAYERS}
    conds = {"none": None}
    for a in (1.0, 2.0):
        conds[f"pooled_a{a:g}"] = ({L: torch.from_numpy(d["pooled"][i][None]) for i, L in enumerate(LAYERS)}, a)
        conds[f"fields3_a{a:g}"] = ({L: torch.from_numpy(d["field_ortho"][i]) for i, L in enumerate(LAYERS)}, a)
        conds[f"random3_a{a:g}"] = (rand, a)

    def example(r):
        """(text, char span to score) for the turn where the secret first appears, or the
        whole final reply for benign rows."""
        msgs = [{"role": "system", "content": build_system_prompt([by_id[i] for i in r["context_ids"]], r["strictness"])}]
        for t, (turn, reply) in enumerate(zip(r["turns"], r["replies"])):
            prompt = tok.apply_chat_template(msgs + [{"role": "user", "content": turn}], add_generation_prompt=True,
                                             tokenize=False)
            if r["kind"] == "benign":
                if t == len(r["turns"]) - 1:
                    return prompt + reply, (len(prompt), len(prompt) + len(reply))
            else:
                span = secret_span(reply, by_id[r["employee_id"]], r["field"])
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

    leaks = [r for r in rows if r["kind"] == "attack" and r["label"] == "full" and r["field"] in FIELDS
             and template_of(r["id"]) in held]
    benign = [r for r in rows if r["kind"] == "benign" and r["id"] in half_b]
    results = {"leak": defaultdict(lambda: defaultdict(list)), "benign": defaultdict(list)}
    skipped = 0
    for group, items in (("leak", leaks), ("benign", benign)):
        for r in items:
            ex = example(r)
            if ex is None:
                skipped += 1
                continue
            for name, cfg in conds.items():
                handles = multi_project(model, *cfg) if cfg else []
                try:
                    lp = mean_logprob(*ex)
                finally:
                    for h in handles:
                        h.remove()
                if lp is None:
                    continue
                if group == "leak":
                    results["leak"][r["field"]][name].append(lp)
                else:
                    results["benign"][name].append(lp)
    summary = {"skipped_no_secret_found": skipped, "leak": {}, "benign": {}}
    for f, per in results["leak"].items():
        base = np.array(per["none"])
        summary["leak"][f] = {"n": len(base), **{k: {"mean_logprob": float(np.mean(v)),
                              "delta_vs_none": float(np.mean(np.array(v) - base)),
                              "secret_prob_ratio": float(np.exp(np.mean(np.array(v) - base)))} for k, v in per.items()}}
    base = np.array(results["benign"]["none"])
    summary["benign"] = {"n": len(base), **{k: {"mean_logprob": float(np.mean(v)), "delta_vs_none": float(np.mean(np.array(v) - base))}
                                             for k, v in results["benign"].items()}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split", "teacher"])
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    do_split(a.runs) if a.step == "split" else do_teacher(a.runs)
