"""Hugging Face backend for the target: batched generation plus activation capture.

Same `chat(messages) -> str` as LMStudioBackend, so the runner and tests work
unchanged. `generate_batch` is the fast path and also returns, per answer, the
residual stream at every layer:

  last_prompt  [n_layers + 1, hidden]  the last prompt token, before any answer
                                       token exists ("does it know it's about to leak?")
  answer_mean  [n_layers + 1, hidden]  mean over the answer tokens (probe / v̂ input)

Index 0 is the embedding output, index L is the output of decoder layer L - 1
(as in `output_hidden_states`). The last index has the final RMSNorm applied, so
it's not the raw residual stream; probe and steer at middle layers anyway. Activations come from one forward pass over prompt +
answer after generation, so they're exactly what the model computed for the
answer it gave.

Needs torch + transformers (GX10 only; see scripts/setup_gx10.sh).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFENSES = ("none", "gate", "steer", "lora")


@dataclass
class Generation:
    text: str
    acts: dict[str, np.ndarray]  # {"last_prompt": [L+1, H], "answer_mean": [L+1, H]} fp16
    gate: dict | None = None  # with a gate: {"risks": [...], "fired_at": int | None, "emitted": str}


def answer_mask(gen: torch.Tensor, stop_ids: set[int]) -> torch.Tensor:
    """[batch, gen_len] bool: True for generated tokens before the first stop token."""
    is_stop = torch.isin(gen, torch.tensor(sorted(stop_ids), device=gen.device))
    return is_stop.cumsum(dim=1) == 0


def masked_mean(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of hs [batch, layers, seq, hidden] over positions where mask [batch, seq] is True.
    Rows with no True positions give zeros."""
    m = mask.to(hs.dtype)[:, None, :, None]
    return (hs * m).sum(dim=2) / m.sum(dim=2).clamp(min=1)


class HFBackend:
    """Greedy decoding, left-padded batches, bf16 on the GPU."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", max_new_tokens: int = 200,
                 batch_size: int = 32):
        self.name = model_id.split("/")[-1].lower()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tok.pad_token_id}
        self._lock = threading.Lock()  # the runner may call chat() from several threads
        self.gate = None  # a leakproof.gate.GateProbe turns the gate on
        self.risk_probe = None  # set by configure_defenses() for generate()
        # The activation cache needs an extra forward pass that holds every layer for the
        # whole batch (~50 GB at batch 48). Turn off when only text (and gate info) is needed.
        self.capture_acts = True

    @classmethod
    def from_env(cls) -> "HFBackend":
        return cls(
            model_id=os.environ.get("LEAKPROOF_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            batch_size=int(os.environ.get("LEAKPROOF_BATCH", "32")),
        )

    def check(self) -> str | None:
        return None

    @classmethod
    def with_saved_defenses(cls, gate_npz=None, best_json=None, **kwargs) -> "HFBackend":
        """Load the model and the latest saved gate probe (runs/gate/*/gate_probe.npz) and
        steering setting (runs/steer/sweep/*/best.json)."""
        import json
        from leakproof.data import ROOT
        from leakproof.gate import GateProbe
        gate_npz = gate_npz or sorted((ROOT / "runs" / "gate").glob("*/gate_probe.npz"))[-1]
        best_json = best_json or sorted((ROOT / "runs" / "steer" / "sweep").glob("*/best.json"))[-1]
        best = json.loads(open(best_json).read())
        lo, _, hi = best["window"].removeprefix("L").partition("-")
        layers = list(range(int(lo), int(hi or lo) + 1))
        backend = cls(**kwargs)
        backend.configure_defenses(GateProbe.load(gate_npz), layers, best["alpha"])
        return backend

    def chat(self, messages: list[dict]) -> str:
        with self._lock:
            return self.generate_batch([messages])[0].text

    # --- UI interface (M5) -----------------------------------------------------------

    def configure_defenses(self, probe, steer_layers: list[int], steer_alpha: float) -> None:
        """probe: a leakproof.gate.GateProbe (scores every answer and drives the gate);
        steer_layers / steer_alpha: the steering setting chosen in M3."""
        from leakproof.steer import load_vhat
        self.risk_probe = probe
        self.steer_dirs = load_vhat(steer_layers)
        self.steer_alpha = steer_alpha

    def generate(self, messages: list[dict], defense: str = "none") -> dict:
        """One answer for the UI: {text, tokens, per_token_risk, defense, fired_at}.

        tokens are the answer tokens the model produced (for the gate, the ones emitted
        before it cut in); per_token_risk[i] = P(leak) of the answer so far (running mean
        at the probe layer, same probe as the gate). With defense="gate" and the gate
        firing, text is the refusal. Needs configure_defenses() first."""
        from contextlib import nullcontext
        from leakproof.gate import REFUSAL, running_mean, watching
        from leakproof.steer import steering
        if defense not in DEFENSES:
            raise ValueError(f"unknown defense {defense!r}; expected one of {DEFENSES}")
        if defense == "lora":
            raise NotImplementedError("defense='lora': no LoRA adapter is loaded yet (Anthony's fine-tune)")
        if getattr(self, "risk_probe", None) is None:
            raise RuntimeError("call configure_defenses() before generate()")
        probe = self.risk_probe
        ctx = steering(self.model, self.steer_dirs, self.steer_alpha) if defense == "steer" else nullcontext()
        with self._lock, ctx:
            enc = self.encode([messages])
            P = enc["input_ids"].shape[1]
            fired = None
            if defense == "gate":
                from transformers import StoppingCriteriaList
                with watching(self.model, probe, 1, self.stop_ids, P) as live:
                    seqs, mask = self.generate_ids(enc, StoppingCriteriaList([live.criteria]))
                fired = live.fired[0]
            else:
                seqs, mask = self.generate_ids(enc)
            hs = self.hidden_states(seqs, enc["attention_mask"], mask)[0, probe.layer, P:]
        ids = seqs[0, P:][mask[0]]
        H = hs[mask[0]].float().cpu().numpy()
        risk = probe.risk(running_mean(H).astype(np.float32)).tolist() if len(H) else []
        return {
            "text": REFUSAL if fired is not None else self.tok.decode(ids, skip_special_tokens=True),
            "tokens": [self.tok.decode([t]) for t in ids.tolist()],
            "per_token_risk": [float(r) for r in risk],
            "defense": defense,
            "fired_at": fired,
        }

    def generate_batch(self, conversations: list[list[dict]]) -> list[Generation]:
        out: list[Generation] = []
        for i in range(0, len(conversations), self.batch_size):
            out += self._generate(conversations[i:i + self.batch_size])
        return out

    def encode(self, conversations: list[list[dict]]):
        texts = [self.tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                 for c in conversations]
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

    @torch.no_grad()
    def generate_ids(self, enc, stopping_criteria=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy generation. Returns (seqs [batch, prompt + gen], answer mask [batch, gen])."""
        seqs = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   pad_token_id=self.tok.pad_token_id,
                                   stopping_criteria=stopping_criteria)
        prompt_len = enc["input_ids"].shape[1]  # left padding: every prompt ends here
        return seqs, answer_mask(seqs[:, prompt_len:], self.stop_ids)

    @torch.no_grad()
    def hidden_states(self, seqs: torch.Tensor, prompt_attn: torch.Tensor,
                      ans_mask: torch.Tensor) -> torch.Tensor:
        """Re-run prompt + answer once; returns [batch, L+1, seq, hidden]. Positions skip
        the left padding, the same way generate() computes them."""
        attn = torch.cat([prompt_attn, ans_mask.long()], dim=1)
        pos = (attn.cumsum(dim=1) - 1).clamp(min=0)
        hs = self.model(input_ids=seqs, attention_mask=attn, position_ids=pos,
                        output_hidden_states=True).hidden_states
        return torch.stack(hs, dim=1)

    def _generate(self, conversations: list[list[dict]]) -> list[Generation]:
        enc = self.encode(conversations)
        prompt_len = enc["input_ids"].shape[1]
        live = None
        if self.gate is None:
            seqs, mask = self.generate_ids(enc)
        else:
            from transformers import StoppingCriteriaList
            from leakproof.gate import watching
            with watching(self.model, self.gate, len(conversations), self.stop_ids, prompt_len) as live:
                seqs, mask = self.generate_ids(enc, StoppingCriteriaList([live.criteria]))
        if self.capture_acts:
            hs = self.hidden_states(seqs, enc["attention_mask"], mask)
            last_prompt = hs[:, :, prompt_len - 1, :].float().cpu().numpy().astype(np.float16)
            answer_mean = masked_mean(hs[:, :, prompt_len:, :], mask).float().cpu().numpy().astype(np.float16)
            del hs
        gen = seqs[:, prompt_len:]

        results = []
        for b in range(len(conversations)):
            text = self.tok.decode(gen[b][mask[b]], skip_special_tokens=True)
            gate = None
            if live is not None:
                gate = {"risks": live.risks[b], "fired_at": live.fired[b], "emitted": text}
                if live.fired[b] is not None:
                    from leakproof.gate import REFUSAL
                    text = REFUSAL
            acts = ({"last_prompt": last_prompt[b], "answer_mean": answer_mean[b]}
                    if self.capture_acts else {})
            results.append(Generation(text, acts, gate))
        return results
