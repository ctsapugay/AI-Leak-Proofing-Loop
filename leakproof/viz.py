"""Leak telemetry: one model call -> a `requests` row + `tokens` rows of db/schema.sql.

    from leakproof.hf_target import HFBackend
    from leakproof import viz
    backend = HFBackend.with_saved_defenses()          # gate probe + steering setting
    tr = viz.trace(backend, messages, defense="none" | "gate" | "steer")
    tr["request"]        # {model, defense, ..., prompt_risk, peak_risk, risk_cross_pos, ...}
    tr["tokens"]         # [{pos, token_id, text, char_start, char_end, risk,
                         #   contrib_base, attn_contrib, mlp_contrib, steer}, ...]
    viz.insert(conn, tr) # writes both into an open sqlite3 connection, returns the request id

On a laptop (no GPU, same keys and the same invariants):

    tr = viz.MockTracer().trace(messages, defense="steer")

What the numbers mean (docs/PLAN-viz.md, "How the numbers are defined"):

  - risk of token t = the gate's probe on the running mean of the answer's hidden states at
    probe_layer over tokens 0..t (the gate's own input), in risk units: the probe logit
    rescaled so the median train-template `none` answer = 0 and the median `full` = 1.
  - The probe logit is linear in the hidden state, and the hidden state is the embedding
    plus every layer's attention and MLP update, so
        contrib_base + sum(attn_contrib) + sum(mlp_contrib) - sum(steer[l] for l <= probe_layer) = risk
    attn_contrib / mlp_contrib have one value per decoder layer 0..probe_layer-1.
  - steer = {"<hidden_states index>": leak signal the steering hook removed there, in risk
    units}. Keys use the same numbering as defense_params.steer_layers; layers after
    probe_layer are listed too but don't change this token's risk. NULL with steering off.
  - response = the text the model actually emitted (with the gate, cut where it fired;
    the board shows the refusal when gate_fired_pos is set). "".join(token texts) == response.

JSON columns are Python objects here; insert() serializes them.

Fit the risk units and the last-prompt probe (CPU, cached activations; writes runs/viz/):

    .venv/bin/python -m leakproof.viz fit --src ~/clara/AI-Hackathon/runs --gate <gate_probe.npz>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from leakproof.data import ROOT

VIZ_DIR = ROOT / "runs" / "viz"
SCHEMA = ROOT / "db" / "schema.sql"
DEFENSES = ("none", "gate", "steer")  # lora / lora+steer: no adapter yet
JSON_COLUMNS = {"defense_params", "messages", "context", "tags", "judge_raw",
                "attn_contrib", "mlp_contrib", "steer"}


# --- probe + risk units ----------------------------------------------------------

@dataclass
class Probe:
    """A linear probe on one layer (the gate's GateProbe fields): logit = ((x - mean) / scale) · w + b."""
    layer: int  # hidden_states index
    w: np.ndarray
    b: float
    mean: np.ndarray
    scale: np.ndarray
    threshold: float = 0.5  # in probability, like the gate
    k: int = 2

    @classmethod
    def load(cls, path) -> "Probe":
        z = np.load(path)
        return cls(int(z["layer"]), z["w"], float(z["b"]), z["mean"], z["scale"],
                   float(z["threshold"]) if "threshold" in z else 0.5, int(z["k"]) if "k" in z else 2)

    @classmethod
    def of(cls, gp) -> "Probe":
        """From a leakproof.gate.GateProbe (or anything with the same fields)."""
        return cls(int(gp.layer), np.asarray(gp.w), float(gp.b), np.asarray(gp.mean),
                   np.asarray(gp.scale), float(gp.threshold), int(gp.k))

    def readout(self) -> tuple[np.ndarray, float]:
        """(w', b') in float64 with logit = x · w' + b' on raw hidden states."""
        w = self.w.astype(np.float64) / self.scale.astype(np.float64)
        return w, float(self.b - self.mean.astype(np.float64) @ w)

    def logit(self, X: np.ndarray) -> np.ndarray:
        w, b = self.readout()
        return np.asarray(X, dtype=np.float64) @ w + b

    def version(self) -> str:
        h = hashlib.sha1()
        for a in (self.w, self.mean, self.scale):
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        h.update(repr((self.layer, round(self.b, 9))).encode())
        return f"L{self.layer}-{h.hexdigest()[:8]}"


@dataclass
class Units:
    """risk = a * logit + c, with median safe -> 0 and median leak -> 1."""
    a: float
    c: float
    z_safe: float
    z_leak: float

    @classmethod
    def fit(cls, z_safe: np.ndarray, z_leak: np.ndarray) -> "Units":
        s, l = float(np.median(z_safe)), float(np.median(z_leak))
        if not l > s:
            raise ValueError(f"leak median {l:.3f} is not above safe median {s:.3f}")
        a = 1.0 / (l - s)
        return cls(a, -a * s, s, l)

    def __call__(self, z):
        return self.a * np.asarray(z, dtype=np.float64) + self.c

    def from_prob(self, p: float) -> float:
        """A probability threshold in risk units (same decisions: the map is monotone)."""
        return float(self(np.log(p) - np.log1p(-p)))

    def to_dict(self) -> dict:
        return {"a": self.a, "c": self.c, "z_safe": self.z_safe, "z_leak": self.z_leak}

    def version(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:6]


def load_fitted(path: Path = VIZ_DIR / "units.json") -> dict:
    """{'units': Units for the gate probe, 'prompt': (Probe, Units) or None, ...} from `fit`."""
    d = json.loads(Path(path).read_text())
    out = {**d, "units": Units(**d["units"]), "prompt": None}
    if d.get("prompt_probe"):
        out["prompt"] = (Probe.load(Path(path).parent / d["prompt_probe"]), Units(**d["prompt_units"]))
    return out


# --- decomposition ---------------------------------------------------------------

def running_mean(x: np.ndarray) -> np.ndarray:
    """Mean of x[..., 0..t] along the last axis, for every t."""
    x = np.asarray(x, dtype=np.float64)
    return np.cumsum(x, axis=-1) / np.arange(1, x.shape[-1] + 1)


def decompose(cap: dict, b: float, units: Units, probe_layer: int) -> dict:
    """Per-token risk and its breakdown, from readouts (x · w') of the residual stream.

    cap holds [n_layers_captured, T] arrays over the answer tokens: "in" (decoder layer i's
    input; in[0] is the embedding output), "mid" (after attention), "pre" (after the MLP,
    before any steering hook), "out" (the layer's final output). Returns float64 arrays:
    risk [T], base [T], attn [probe_layer, T], mlp [probe_layer, T], steer {L: [T]}."""
    rm = {k: running_mean(v) for k, v in cap.items()}
    L = probe_layer
    a = units.a
    attn = a * (rm["mid"][:L] - rm["in"][:L])
    mlp = a * (rm["pre"][:L] - rm["mid"][:L])
    removed = a * (rm["pre"] - rm["out"])
    steer = {i + 1: removed[i] for i in cap.get("steered", [])}
    return {
        "risk": units(rm["out"][L - 1] + b),  # read directly, not from the sum
        "base": units(rm["in"][0] + b),
        "attn": attn,
        "mlp": mlp,
        "steer": steer,
    }


def identity_residual(d: dict, probe_layer: int) -> np.ndarray:
    """base + sum(attn) + sum(mlp) - sum(steer at layers <= probe_layer) - risk (should be ~0)."""
    s = d["base"] + d["attn"].sum(axis=0) + d["mlp"].sum(axis=0)
    for L, v in d["steer"].items():
        if L <= probe_layer:
            s = s - v
    return s - d["risk"]


@contextmanager
def capturing(model, w, n_layers: int, positions=None, steered=(), keep_raw_at=None):
    """Hooks that record x · w for the residual stream around decoder layers 0..n_layers-1.

    Enter this *inside* any steering context: the "pre" hook is prepended after the steering
    hook was, so it runs first and sees the stream before the edit; "out" is appended and sees
    it after. keep_raw_at = (layer i, position): also keep that raw "out" vector (float64).
    Yields a dict filled by the next forward pass; values are float64 numpy [n_layers, seq]."""
    import torch
    layers = model.model.layers
    wt = torch.as_tensor(np.asarray(w, dtype=np.float64))
    rec: dict = {"in": {}, "mid": {}, "pre": {}, "out": {}, "steered": sorted(int(s) - 1 for s in steered)}
    handles = []

    def dot(h):
        h = h[0] if isinstance(h, tuple) else h
        x = h[0].double() @ wt.to(h.device)
        return x.cpu().numpy() if positions is None else x[positions].cpu().numpy()

    def first_arg(args, kwargs):
        return args[0] if args else kwargs["hidden_states"]

    for i in range(n_layers):
        layer = layers[i]
        handles.append(layer.register_forward_pre_hook(
            lambda m, args, kwargs, i=i: rec["in"].__setitem__(i, dot(first_arg(args, kwargs))) or None,
            with_kwargs=True))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda m, args, i=i: rec["mid"].__setitem__(i, dot(args[0])) or None))
        handles.append(layer.register_forward_hook(
            lambda m, a, out, i=i: rec["pre"].__setitem__(i, dot(out)) or None, prepend=True))

        def out_hook(m, a, out, i=i):
            rec["out"][i] = dot(out)
            if keep_raw_at is not None and keep_raw_at[0] == i:
                h = out[0] if isinstance(out, tuple) else out
                rec["raw"] = h[0, keep_raw_at[1]].double().cpu().numpy()
        handles.append(layer.register_forward_hook(out_hook))
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()


def stack(rec: dict) -> dict:
    """capturing()'s per-layer dicts -> [n_layers, T] arrays (the input of decompose)."""
    out = {k: np.stack([rec[k][i] for i in sorted(rec[k])]) for k in ("in", "mid", "pre", "out")}
    out["steered"] = rec["steered"]
    return out


# --- rows -------------------------------------------------------------------------

def char_spans(prefix_lengths: list[int]) -> list[tuple[int, int]]:
    """Token spans from len(decode(ids[:t+1])) for every t. A token that ends mid-character
    (byte-level BPE) gets an empty span; the next one carries the whole character."""
    spans, prev = [], 0
    for n in prefix_lengths:
        end = max(prev, n)
        spans.append((prev, end))
        prev = end
    return spans


def first_over(risks, threshold: float, k: int = 1) -> int | None:
    """Index of the k-th consecutive value > threshold (k=1: first crossing), or None."""
    run = 0
    for i, r in enumerate(risks):
        run = run + 1 if r > threshold else 0
        if run >= k:
            return i
    return None


def build_rows(*, model: str, defense: str, defense_params, probe_version: str, probe_layer: int,
               threshold: float, messages: list[dict], response: str, ids: list[int],
               spans: list[tuple[int, int]], d: dict | None, prompt_risk: float | None,
               gate_fired_pos: int | None, n_prompt_tokens: int, latency_ms: int,
               error: str | None = None) -> dict:
    T = len(ids)
    risk = d["risk"] if T else np.zeros(0)
    tokens = []
    for t in range(T):
        tokens.append({
            "pos": t,
            "token_id": int(ids[t]),
            "text": response[spans[t][0]:spans[t][1]],
            "char_start": spans[t][0],
            "char_end": spans[t][1],
            "risk": float(risk[t]),
            "contrib_base": float(d["base"][t]),
            "attn_contrib": [float(x) for x in d["attn"][:, t]],
            "mlp_contrib": [float(x) for x in d["mlp"][:, t]],
            "steer": {str(L): float(v[t]) for L, v in sorted(d["steer"].items())} or None,
        })
    request = {
        "model": model,
        "defense": defense,
        "defense_params": defense_params,
        "probe_version": probe_version,
        "probe_layer": probe_layer,
        "threshold": threshold,
        "messages": messages,
        "response": response,
        "prompt_risk": prompt_risk,
        "peak_risk": float(risk.max()) if T else None,
        "risk_cross_pos": first_over(risk, threshold),
        "gate_fired_pos": gate_fired_pos,
        "n_prompt_tokens": n_prompt_tokens,
        "n_tokens": T,
        "latency_ms": latency_ms,
        "error": error,
    }
    return {"request": request, "tokens": tokens}


def insert(conn: sqlite3.Connection, tr: dict) -> int:
    """Write a trace into the schema's tables; returns requests.id. Doesn't commit."""
    def val(k, v):
        return json.dumps(v) if k in JSON_COLUMNS and v is not None else v
    req = tr["request"]
    cols = list(req)
    cur = conn.execute(f"INSERT INTO requests ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                       [val(k, req[k]) for k in cols])
    rid = cur.lastrowid
    for tok in tr["tokens"]:
        tcols = ["request_id", *tok]
        conn.execute(f"INSERT INTO tokens ({', '.join(tcols)}) VALUES ({', '.join('?' * len(tcols))})",
                     [rid, *(val(k, v) for k, v in tok.items())])
    return rid


def connect_schema(path=":memory:") -> sqlite3.Connection:
    """A sqlite3 connection with db/schema.sql applied (tests, local use)."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.read_text())
    return conn


# --- live trace on the HF backend ---------------------------------------------------

def trace(backend, messages: list[dict], defense: str = "none", fitted: dict | None = None) -> dict:
    """Generate one answer with `defense` and return its schema rows (see module docstring).

    Uses the backend's own pieces (encode, generate_ids, the gate's `watching`, `steering`)
    and its configured probe / steering setting (HFBackend.with_saved_defenses()), then one
    extra forward pass over prompt + answer with readout hooks. Batch 1."""
    import torch
    from transformers import StoppingCriteriaList
    from leakproof.gate import watching
    from leakproof.steer import steering
    if defense not in DEFENSES:
        raise ValueError(f"unknown defense {defense!r}; expected one of {DEFENSES}")
    if getattr(backend, "risk_probe", None) is None:
        raise RuntimeError("call backend.configure_defenses() (or HFBackend.with_saved_defenses()) first")
    fitted = fitted or load_fitted()
    probe = Probe.of(backend.risk_probe)
    units: Units = fitted["units"]
    L = probe.layer
    w, b = probe.readout()
    steer_dirs = backend.steer_dirs if defense == "steer" else {}
    steered = sorted(steer_dirs)
    params = ({"alpha": backend.steer_alpha, "steer_layers": steered} if defense == "steer"
              else {"k": probe.k, "threshold_prob": probe.threshold} if defense == "gate" else None)
    model = backend.model
    with backend._lock:
        ctx = steering(model, steer_dirs, backend.steer_alpha) if steer_dirs else nullcontext()
        with ctx:
            enc = backend.encode([messages])
            P = enc["input_ids"].shape[1]
            fired, live_risks = None, None
            t0 = time.perf_counter()
            if defense == "gate":
                with watching(model, backend.risk_probe, 1, backend.stop_ids, P) as live:
                    seqs, mask = backend.generate_ids(enc, StoppingCriteriaList([live.criteria]))
                fired, live_risks = live.fired[0], live.risks[0]
            else:
                seqs, mask = backend.generate_ids(enc)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            n = int(mask[0].sum())
            # The same forward pass as backend.hidden_states(): full sequence, same mask and positions.
            attn = torch.cat([enc["attention_mask"], mask.long()], dim=1)
            pos = (attn.cumsum(dim=1) - 1).clamp(min=0)
            n_cap = max([L, *steered])
            with torch.no_grad(), capturing(model, w, n_cap, positions=slice(P, P + n), steered=steered,
                                            keep_raw_at=(L - 1, P - 1)) as rec:
                model(input_ids=seqs, attention_mask=attn, position_ids=pos)
    ids = seqs[0, P:P + n].tolist()
    tok = backend.tok
    response = tok.decode(ids, skip_special_tokens=True)
    spans = char_spans([len(tok.decode(ids[:t + 1], skip_special_tokens=True)) for t in range(n)])
    d = decompose(stack(rec), b, units, L) if n else None
    prompt_risk = None
    if fitted.get("prompt"):
        pp, pu = fitted["prompt"]
        assert pp.layer == L, "the last-prompt probe must read the same layer"
        prompt_risk = float(pu(pp.logit(rec["raw"])))
    tr = build_rows(model=backend.name, defense=defense, defense_params=params,
                    probe_version=f"{probe.version()}+u{units.version()}", probe_layer=L,
                    threshold=units.from_prob(probe.threshold), messages=messages, response=response,
                    ids=ids, spans=spans, d=d, prompt_risk=prompt_risk, gate_fired_pos=fired,
                    n_prompt_tokens=int(enc["attention_mask"][0].sum()), latency_ms=latency_ms)
    if live_risks is not None:
        tr["live_gate_risks"] = [float(r) for r in live_risks]  # for the live-vs-offline proof
    return tr


# --- mock (laptop, no model) ----------------------------------------------------------

class MockTracer:
    """Same rows as trace(), from MockBackend's scripted answers. The breakdown is made up
    (risk climbs through the middle layers on secret-looking tokens) but obeys the same
    identity, gate and steer rules, so the board can be built against it."""

    name = "mock"

    def __init__(self, probe_layer: int = 18, steer_layers=(16, 17, 18, 19, 20), alpha: float = 2.0):
        from leakproof.mock import MockBackend
        self.backend = MockBackend()
        self.L = probe_layer
        self.steer_layers = list(steer_layers)
        self.alpha = alpha
        self.units = Units(a=0.125, c=0.5, z_safe=-4.0, z_leak=4.0)  # logit -4 -> 0, +4 -> 1
        self.threshold = self.units.from_prob(self.backend.threshold)
        x = np.arange(self.L)
        shape = np.exp(-0.5 * ((x - 0.7 * self.L) / (0.2 * self.L)) ** 2)  # middle-late layers build it
        self.layer_share = shape / shape.sum()

    def trace(self, messages: list[dict], defense: str = "none") -> dict:
        if defense not in DEFENSES:
            raise ValueError(f"unknown defense {defense!r}; expected one of {DEFENSES}")
        out = self.backend.generate(messages, defense=defense)
        toks = out["tokens"]
        response = "".join(toks)
        p = np.clip(np.array(out["per_token_risk"], dtype=np.float64), 1e-6, 1 - 1e-6)
        risk = self.units(np.log(p) - np.log1p(-p))
        base = np.full(len(toks), float(self.units(-4.5)))  # the probe's offset: a bit below "safe"
        built = risk - base
        steered = self.steer_layers if defense == "steer" else []
        steer = {L: np.full(len(toks), 0.08 * self.alpha) for L in steered}
        removed_before = sum((v for L, v in steer.items() if L <= self.L), np.zeros(len(toks)))
        total = built + removed_before  # what the layers built before steering took its part out
        attn = 0.4 * self.layer_share[:, None] * total
        mlp = 0.6 * self.layer_share[:, None] * total
        d = {"risk": risk, "base": base, "attn": attn, "mlp": mlp, "steer": steer}
        lens = np.cumsum([len(t) for t in toks]).tolist()
        params = ({"alpha": self.alpha, "steer_layers": steered} if defense == "steer"
                  else {"k": self.backend.k, "threshold_prob": self.backend.threshold} if defense == "gate" else None)
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        from leakproof.mock import SENSITIVE_Q
        prompt_risk = 0.85 if SENSITIVE_Q.search(question) else 0.05
        return build_rows(model=self.name, defense=defense, defense_params=params,
                          probe_version=f"mock-L{self.L}", probe_layer=self.L, threshold=self.threshold,
                          messages=messages, response=response, ids=list(range(1000, 1000 + len(toks))),
                          spans=char_spans(lens), d=d, prompt_risk=prompt_risk,
                          gate_fired_pos=out["fired_at"], n_prompt_tokens=len(question.split()) + 40,
                          latency_ms=0)


# --- fitting (CPU, cached activations) -----------------------------------------------

def held_out_templates(src_runs: Path) -> set[str]:
    return set(json.loads((src_runs / "steer" / "split.json").read_text())["test_templates"])


def fit(src_runs: Path, gate_npz: Path, C: float = 0.01, out: Path = VIZ_DIR) -> dict:
    """Risk units for the gate probe + a last-prompt probe at the same layer, train templates only."""
    from sklearn.metrics import roc_auc_score
    from leakproof.probe import load_runs, make_probe, shuffled, template_of
    split = json.loads((src_runs / "steer" / "split.json").read_text())
    run_dirs = [src_runs.parent / r if not Path(r).is_absolute() else Path(r) for r in split["runs"]]
    held = set(split["test_templates"])
    gp = Probe.load(gate_npz)
    acts, index = load_runs(run_dirs)
    kind = np.array([r["kind"] for r in index])
    label = np.array([r["label"] for r in index])
    tmpl = np.array([template_of(r["id"]) for r in index])
    attack = (kind == "attack") & np.isin(label, ["full", "none"])
    train = attack & ~np.isin(tmpl, sorted(held))
    test = attack & np.isin(tmpl, sorted(held))
    assert not set(tmpl[train]) & held
    y = (label == "full").astype(int)

    z = gp.logit(acts["answer_mean"][:, gp.layer].astype(np.float32))
    units = Units.fit(z[train & (y == 0)], z[train & (y == 1)])
    r = units(z)
    p = 1 / (1 + np.exp(-z))
    thr_u = units.from_prob(gp.threshold)
    report = {
        "gate_npz": str(gate_npz), "gate_layer": gp.layer, "probe_version": gp.version(),
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "train_median_none": float(np.median(r[train & (y == 0)])),
        "train_median_full": float(np.median(r[train & (y == 1)])),
        "heldout_median_none": float(np.median(r[test & (y == 0)])),
        "heldout_median_full": float(np.median(r[test & (y == 1)])),
        "benign_median": float(np.median(r[kind == "benign"])),
        "threshold_prob": gp.threshold, "threshold_units": thr_u,
        "same_decisions_all_cached": bool(np.array_equal(r > thr_u, p > gp.threshold)),
        "n_cached": int(len(r)),
    }

    # Last-prompt probe at the gate layer (prompt_risk).
    Xp = acts["last_prompt"][:, gp.layer].astype(np.float32)
    pp = make_probe(C).fit(Xp[train], y[train])
    auc = float(roc_auc_score(y[test], pp.decision_function(Xp[test])))
    auc_shuf = [float(roc_auc_score(y[test], make_probe(C).fit(Xp[train], shuffled(y[train], seed))
                                    .decision_function(Xp[test]))) for seed in range(5)]
    prompt = Probe(gp.layer, pp[1].coef_[0].astype(np.float32), float(pp[1].intercept_[0]),
                   pp[0].mean_.astype(np.float32), pp[0].scale_.astype(np.float32))
    zp = prompt.logit(Xp)
    punits = Units.fit(zp[train & (y == 0)], zp[train & (y == 1)])
    report.update({"prompt_C": C, "prompt_heldout_auroc": auc, "prompt_shuffled_heldout_auroc": float(np.mean(auc_shuf)),
                   "prompt_shuffled_heldout_auroc_seeds": auc_shuf,
                   "prompt_heldout_median_none": float(np.median(punits(zp[test & (y == 0)]))),
                   "prompt_heldout_median_full": float(np.median(punits(zp[test & (y == 1)])))})

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "prompt_probe.npz", layer=prompt.layer, w=prompt.w, b=prompt.b, mean=prompt.mean,
             scale=prompt.scale, train_templates=np.array(sorted(set(tmpl[train]))))
    fitted = {"units": units.to_dict(), "probe_version": gp.version(), "prompt_probe": "prompt_probe.npz",
              "prompt_units": punits.to_dict(), "train_templates": sorted(set(tmpl[train]))}
    (out / "units.json").write_text(json.dumps(fitted, indent=2, sort_keys=True) + "\n")
    (out / "held_out_templates.json").write_text(json.dumps(sorted(held)) + "\n")
    (out / "fit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--src", type=Path, default=Path(os.environ.get("LEAKPROOF_SRC_RUNS", ROOT / "runs")))
    f.add_argument("--gate", type=Path, required=True)
    f.add_argument("--C", type=float, default=0.01)
    args = ap.parse_args()
    if args.cmd == "fit":
        print(json.dumps(fit(args.src.expanduser(), args.gate.expanduser(), args.C), indent=2))


if __name__ == "__main__":
    main()
