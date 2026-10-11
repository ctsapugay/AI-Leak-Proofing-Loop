"""V1 proofs: fitting the risk units and the last-prompt probe (train templates only).

Synthetic tests run anywhere. The real-data checks read runs/viz/fit_report.json (written on
the box by `python -m leakproof.viz fit`) and skip if it isn't there.
"""

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from leakproof.viz import VIZ_DIR, Units, fit, load_fitted  # noqa: E402

DIM, LAYERS, LAYER = 24, 4, 2


def fake_src(tmp_path, seed=0, signal=1.0):
    """runs/ with two collect runs, split.json and a gate probe, with a planted leak direction."""
    r = np.random.default_rng(seed)
    v = r.normal(size=DIM)
    v /= np.linalg.norm(v)
    templates = [f"t{i}" for i in range(10)]
    held = templates[:4]
    runs = []
    for run in ("a", "b"):
        rows, am, lp = [], [], []
        for t in templates:
            for e in range(12):
                label = ["full", "none", "partial"][(e + (run == "b")) % 3]
                y = {"full": 1, "none": -1, "partial": 0}[label]
                am.append(r.normal(size=(LAYERS, DIM)) + signal * y * v)
                lp.append(r.normal(size=(LAYERS, DIM)) + 2 * signal * y * v)
                rows.append({"id": f"{t}@e{e:03d}", "kind": "attack", "label": label})
        for e in range(8):
            am.append(r.normal(size=(LAYERS, DIM)) - signal * v)
            lp.append(r.normal(size=(LAYERS, DIM)))
            rows.append({"id": f"benign-{e}@e{e:03d}", "kind": "benign", "label": "none"})
        d = tmp_path / "runs" / "collect" / run / "acts"
        d.mkdir(parents=True)
        np.save(d / "answer_mean.npy", np.array(am, dtype=np.float16))
        np.save(d / "last_prompt.npy", np.array(lp, dtype=np.float16))
        (d / "index.jsonl").write_text("".join(json.dumps({"row": i, **x}) + "\n" for i, x in enumerate(rows)))
        runs.append(f"runs/collect/{run}")
    (tmp_path / "runs" / "steer").mkdir()
    (tmp_path / "runs" / "steer" / "split.json").write_text(json.dumps({"runs": runs, "test_templates": held}))
    gate = tmp_path / "gate_probe.npz"
    np.savez(gate, layer=LAYER, w=(v * 2).astype(np.float32), b=0.1, mean=np.zeros(DIM, np.float32),
             scale=np.ones(DIM, np.float32), threshold=0.7, k=2, train_templates=np.array(templates[4:]))
    return tmp_path / "runs", gate, set(held)


def test_fit_on_planted_signal(tmp_path):
    src, gate, held = fake_src(tmp_path)
    out = tmp_path / "viz"
    rep = fit(src, gate, out=out)
    assert rep["train_median_none"] == pytest.approx(0, abs=1e-9)
    assert rep["train_median_full"] == pytest.approx(1, abs=1e-9)
    assert rep["same_decisions_all_cached"]
    assert rep["prompt_heldout_auroc"] > 0.9
    assert abs(rep["prompt_shuffled_heldout_auroc"] - 0.5) < 0.15  # control
    f = load_fitted(out / "units.json")
    assert not set(f["train_templates"]) & held  # no held-out template trained on
    pp, pu = f["prompt"]
    assert pp.layer == LAYER and isinstance(pu, Units)
    assert not set(np.load(out / "prompt_probe.npz")["train_templates"]) & held


def test_fit_is_reproducible(tmp_path):
    src, gate, _ = fake_src(tmp_path)
    fit(src, gate, out=tmp_path / "v1")
    fit(src, gate, out=tmp_path / "v2")
    for name in ("units.json", "fit_report.json"):
        assert (tmp_path / "v1" / name).read_text().replace("v1", "") == (tmp_path / "v2" / name).read_text().replace("v2", "")
    a, b = np.load(tmp_path / "v1" / "prompt_probe.npz"), np.load(tmp_path / "v2" / "prompt_probe.npz")
    assert all(np.array_equal(a[k], b[k]) for k in a.files)


def test_held_out_rows_dont_move_the_units(tmp_path):
    """Changing only held-out activations leaves everything that was fit unchanged."""
    src, gate, held = fake_src(tmp_path)
    fit(src, gate, out=tmp_path / "v1")
    for run in ("a", "b"):
        d = src / "collect" / run / "acts"
        idx = [json.loads(x) for x in (d / "index.jsonl").read_text().splitlines()]
        rows = [i for i, x in enumerate(idx) if x["id"].split("@")[0] in held]
        for name in ("answer_mean.npy", "last_prompt.npy"):
            X = np.load(d / name)
            X[rows] += np.float16(3.0)
            np.save(d / name, X)
    fit(src, gate, out=tmp_path / "v2")
    assert json.loads((tmp_path / "v1" / "units.json").read_text()) == json.loads((tmp_path / "v2" / "units.json").read_text())


# --- real data (box) ---------------------------------------------------------------

REAL = VIZ_DIR / "fit_report.json"
real = pytest.mark.skipif(not REAL.exists(), reason="run `python -m leakproof.viz fit` on the box first")


@real
def test_real_units_and_controls():
    rep = json.loads(REAL.read_text())
    assert rep["train_median_none"] == pytest.approx(0, abs=1e-9)
    assert rep["train_median_full"] == pytest.approx(1, abs=1e-9)
    assert rep["same_decisions_all_cached"]
    assert abs(rep["prompt_shuffled_heldout_auroc"] - 0.5) < 0.1
    assert rep["prompt_heldout_auroc"] > rep["prompt_shuffled_heldout_auroc"] + 0.2
    f = load_fitted()
    held = set(json.loads((VIZ_DIR / "held_out_templates.json").read_text()))
    assert held and not set(f["train_templates"]) & held
    assert f["probe_version"] == rep["probe_version"]
