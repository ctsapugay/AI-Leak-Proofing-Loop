"""Hugging Face backend for the target: batched generation plus activation capture.

Same `chat(messages) -> str` as LMStudioBackend, so the runner and tests work
unchanged. `generate_batch` is the fast path and also returns, per answer, the
residual stream at every layer:

  last_prompt  [n_layers + 1, hidden]  the last prompt token, before any answer
                                       token exists ("does it know it's about to leak?")
  answer_mean  [n_layers + 1, hidden]  mean over the answer tokens (probe / v̂ input)

Index 0 is the embedding output, index L is the output of layer L (as in
`output_hidden_states`). Activations come from one forward pass over prompt +
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

    @torch.no_grad()
    def _generate(self, conversations: list[list[dict]]) -> list[Generation]:
        texts = [self.tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                 for c in conversations]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        prompt_len = enc["input_ids"].shape[1]  # left padding: every prompt ends here

        seqs = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   pad_token_id=self.tok.pad_token_id)

        # Answer mask: generated tokens before the first stop token.
        gen = seqs[:, prompt_len:]
        is_stop = torch.isin(gen, torch.tensor(sorted(self.stop_ids), device=gen.device))
        answer_mask = (is_stop.cumsum(dim=1) == 0)

        # Re-run prompt + answer once to read every layer. Positions must skip the left
        # padding, the same way generate() computes them.
        attn = torch.cat([enc["attention_mask"], answer_mask.long()], dim=1)
        pos = (attn.cumsum(dim=1) - 1).clamp(min=0)
        hs = self.model(input_ids=seqs, attention_mask=attn, position_ids=pos,
                        output_hidden_states=True).hidden_states
        hs = torch.stack(hs, dim=1)  # [batch, L+1, seq, hidden]

        last_prompt = hs[:, :, prompt_len - 1, :]
        m = answer_mask.to(hs.dtype)[:, None, :, None]  # [batch, 1, gen, 1]
        answer_mean = (hs[:, :, prompt_len:, :] * m).sum(dim=2) / m.sum(dim=2).clamp(min=1)

        results = []
        for b in range(len(conversations)):
            text = self.tok.decode(gen[b][answer_mask[b]], skip_special_tokens=True)
            results.append(Generation(text, {
                "last_prompt": last_prompt[b].float().cpu().numpy().astype(np.float16),
                "answer_mean": answer_mean[b].float().cpu().numpy().astype(np.float16),
            }))
        return results
