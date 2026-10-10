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


@dataclass
class Generation:
    text: str
    acts: dict[str, np.ndarray]  # {"last_prompt": [L+1, H], "answer_mean": [L+1, H]} fp16


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

    @classmethod
    def from_env(cls) -> "HFBackend":
        return cls(
            model_id=os.environ.get("LEAKPROOF_HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            batch_size=int(os.environ.get("LEAKPROOF_BATCH", "32")),
        )

    def check(self) -> str | None:
        return None

    def chat(self, messages: list[dict]) -> str:
        with self._lock:
            return self.generate_batch([messages])[0].text

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
    def generate_ids(self, enc) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy generation. Returns (seqs [batch, prompt + gen], answer mask [batch, gen])."""
        seqs = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   pad_token_id=self.tok.pad_token_id)
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
        seqs, mask = self.generate_ids(enc)
        hs = self.hidden_states(seqs, enc["attention_mask"], mask)
        last_prompt = hs[:, :, prompt_len - 1, :]
        answer_mean = masked_mean(hs[:, :, prompt_len:, :], mask)
        gen = seqs[:, prompt_len:]

        results = []
        for b in range(len(conversations)):
            text = self.tok.decode(gen[b][mask[b]], skip_special_tokens=True)
            results.append(Generation(text, {
                "last_prompt": last_prompt[b].float().cpu().numpy().astype(np.float16),
                "answer_mean": answer_mean[b].float().cpu().numpy().astype(np.float16),
            }))
        return results
