"""How players talk to their model and how benchmarks score it.

* :class:`TorchLM` runs a :class:`~airace_ml.model.transformer.Transformer` for two jobs:
  batched, KV-cached, seeded **generation** and right-padded **continuation scoring**.
* :func:`build_chat_prompt` turns a chat history into a prompt that always fits the model's
  attention span: it starts with ``<|bos|>``, ends with ``<|ai|>``, and the oldest turns are
  dropped first.
* :class:`LanguageModel` is the narrow interface benchmarks depend on, so they can be tested
  against scripted fakes without a real network.

Everything runs under ``torch.inference_mode()`` (a KV cache built with grad enabled would keep the
autograd graph alive), with bf16 autocast on CUDA. All randomness comes from a per-call
``torch.Generator`` seeded by the caller.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from airace_ml.device import pick_device
from airace_ml.model.transformer import Transformer
from airace_ml.tokenizer import Role, Tok, encode_chat, encode_doc

DEFAULT_BATCH_SIZE = 32
# Smallest chat prompt that can hold one truncated turn: <|bos|> <|role|> <|end|> <|ai|>.
_MIN_CHAT_BUDGET = 4
_FLOAT_TINY = torch.finfo(torch.float32).tiny


@dataclass
class ContinuationScore:
    """Log-probability of a continuation given its context, and how many tokens were scored."""

    sum_logprob: float
    n_tokens: int


@dataclass
class Generation:
    """One generated reply.

    ``tokens`` excludes the stop token. ``token_probs[i]`` is the probability of ``tokens[i]``
    and ``top1_probs[i]`` the probability of the most likely token at that step, both from the
    raw softmax at temperature 1 (before temperature and top-p), so they measure the model's own
    confidence. ``stopped`` is True when generation ended on a stop token rather than on a length
    limit.
    """

    tokens: list[int]
    token_probs: list[float]
    top1_probs: list[float]
    stopped: bool


class LanguageModel(Protocol):
    """What benchmarks need from a model."""

    ctx_len: int

    def score_continuations(
        self, contexts: list[list[int]], continuations: list[list[int]]
    ) -> list[ContinuationScore]: ...

    def generate(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        stop_ids: Sequence[int] | None = None,
    ) -> list[Generation]: ...


def _tail(seq: Sequence[int], n: int) -> list[int]:
    """The last ``n`` items (none for ``n <= 0``; ``seq[-0:]`` would be everything)."""
    return list(seq[max(len(seq) - n, 0) :])


def build_chat_prompt(
    tok: Tok, history: Sequence[tuple[Role, str]], ctx_len: int, reserve: int
) -> list[int]:
    """Chat history as prompt ids, at most ``ctx_len - reserve`` long, ending with ``<|ai|>``.

    The format is exactly ``encode_chat(..., add_generation_prompt=True)``; the oldest turns are
    dropped (whole) until it fits. If even the newest turn alone is too long, its role marker and
    ``<|end|>`` are kept around the tail of its text tokens.
    """
    budget = ctx_len - reserve
    if reserve < 0 or budget < _MIN_CHAT_BUDGET:
        raise ValueError(
            f"reserve={reserve} leaves {budget} prompt tokens of ctx_len={ctx_len}; "
            f"need at least {_MIN_CHAT_BUDGET}"
        )
    kept: list[list[int]] = []  # newest first
    used = 2  # <|bos|> and the trailing <|ai|>
    for turn in reversed(history):
        ids = encode_chat(tok, [turn])[1:]  # [role, *text, <|end|>]
        if used + len(ids) > budget:
            if not kept:
                kept.append([ids[0], *_tail(ids[1:-1], budget - _MIN_CHAT_BUDGET), ids[-1]])
            break
        kept.append(ids)
        used += len(ids)
    body = [token for ids in reversed(kept) for token in ids]
    return [tok.bos_id, *body, tok.ai_id]


def _sample(logits: Tensor, temperature: float, top_p: float, rng: torch.Generator) -> Tensor:
    """One token per row of ``logits`` ``[B, V]``: greedy at temperature 0, else nucleus sampling."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p >= 1.0:
        return torch.multinomial(probs, 1, generator=rng)[:, 0]
    sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
    # A token goes once the mass ahead of it already reaches top_p, so the top token always stays.
    sorted_probs = sorted_probs.masked_fill(sorted_probs.cumsum(-1) - sorted_probs >= top_p, 0.0)
    pick = torch.multinomial(sorted_probs, 1, generator=rng)  # multinomial renormalizes
    return sorted_ids.gather(1, pick)[:, 0]


class TorchLM:
    """A :class:`~airace_ml.model.transformer.Transformer` and its tokenizer behind
    :class:`LanguageModel`.

    ``device`` defaults to :func:`~airace_ml.device.pick_device`; the model is moved there and put
    in eval mode. At most ``batch_size`` sequences go through the model at once; longer lists are
    processed in chunks. Seeded sampling draws from one generator per call, so a given call is
    reproducible, but a prompt's samples depend on the other prompts in the same call.
    """

    def __init__(
        self,
        model: Transformer,
        tok: Tok,
        device: torch.device | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.device = device if device is not None else pick_device()
        self.model = model.to(self.device).eval()
        self.tok = tok
        self.batch_size = batch_size

    @property
    def ctx_len(self) -> int:
        return self.model.shape.ctx_len

    # -- chat and plain completion ------------------------------------------------------------

    def chat_reply(
        self,
        history: Sequence[tuple[Role, str]],
        *,
        max_new_tokens: int = 96,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 0,
    ) -> str:
        """The model's next reply to ``history``, as text (special tokens removed).

        Room for the reply is reserved in the prompt, but never more than half the context, so a
        small model still sees a useful amount of history.
        """
        reserve = min(max_new_tokens, self.ctx_len // 2)
        prompt = build_chat_prompt(self.tok, history, self.ctx_len, reserve)
        gen = self.generate(
            [prompt], max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed
        )[0]
        return self.tok.decode(gen.tokens)

    def complete(
        self,
        text: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 0,
    ) -> str:
        """Continue ``text`` as a plain document; returns only the newly generated text."""
        gen = self.generate(
            [encode_doc(self.tok, text)],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )[0]
        return self.tok.decode(gen.tokens)

    # -- generation ---------------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[list[int]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        stop_ids: Sequence[int] | None = None,
    ) -> list[Generation]:
        """Generate up to ``max_new_tokens`` tokens after each prompt.

        ``temperature == 0`` is greedy; otherwise tokens are sampled from the top-``top_p``
        nucleus of ``softmax(logits / temperature)``. ``stop_ids=None`` means ``(end, bos)``; an
        empty sequence means never stop early. A prompt longer than ``ctx_len - 1`` keeps its last
        tokens (and its leading ``<|bos|>``), and a row stops once prompt plus reply reach
        ``ctx_len``.
        """
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        stops = frozenset((self.tok.end_id, self.tok.bos_id) if stop_ids is None else stop_ids)
        fitted = [self._fit_prompt(p) for p in prompts]
        if max_new_tokens == 0:
            return [Generation([], [], [], False) for _ in fitted]
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        out: list[Generation] = []
        for lo in range(0, len(fitted), self._chunk_size()):
            chunk = fitted[lo : lo + self._chunk_size()]
            out.extend(self._generate_chunk(chunk, max_new_tokens, temperature, top_p, stops, rng))
        return out

    def _generate_chunk(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stops: frozenset[int],
        rng: torch.Generator,
    ) -> list[Generation]:
        """Left-pad, prefill once, then decode one token per step from the KV cache."""
        n = len(prompts)
        ids, mask = self._left_pad(prompts)
        lens = mask.sum(dim=1)
        # Positions count real tokens only, so padding never shifts a row's RoPE phase.
        positions = (mask.long().cumsum(dim=1) - 1).clamp(min=0)
        caps = [min(max_new_tokens, self.ctx_len - len(p)) for p in prompts]
        cache = self.model.new_kv_cache()  # one per batch; it fills lazily and is never reused
        with self._autocast():
            logits = self.model(ids, positions=positions, key_padding_mask=mask, kv_cache=cache)
        logits = logits[:, -1].float()
        gens = [Generation([], [], [], False) for _ in prompts]
        done = [False] * n
        for step in range(max(caps)):
            raw = torch.softmax(logits, dim=-1)  # temperature 1, before top-p
            nxt = _sample(logits, temperature, top_p, rng)
            # Clamped so a probability stays in (0, 1] even if float32 underflows to zero.
            token_p = raw.gather(1, nxt[:, None])[:, 0].clamp_min(_FLOAT_TINY)
            top1 = raw.max(dim=-1).values
            nxt_l, token_p_l, top1_l = nxt.tolist(), token_p.tolist(), top1.tolist()
            for i, gen in enumerate(gens):
                if done[i]:
                    continue
                if nxt_l[i] in stops:
                    gen.stopped = done[i] = True
                    continue
                gen.tokens.append(nxt_l[i])
                gen.token_probs.append(token_p_l[i])
                gen.top1_probs.append(top1_l[i])
                done[i] = len(gen.tokens) >= caps[i]
            if all(done):
                break
            # Finished rows keep decoding harmlessly (their output is ignored); their positions
            # are clamped so nothing is ever fed past the attention span.
            mask = torch.cat([mask, mask.new_ones(n, 1)], dim=1)
            pos = (lens + step).clamp(max=self.ctx_len - 1)[:, None]
            with self._autocast():
                logits = self.model(
                    nxt[:, None], positions=pos, key_padding_mask=mask, kv_cache=cache
                )
            logits = logits[:, -1].float()
        return gens

    def _fit_prompt(self, prompt: Sequence[int]) -> list[int]:
        """Left-truncate to ``ctx_len - 1`` tokens, keeping a leading ``<|bos|>``."""
        if not prompt:
            raise ValueError("empty prompt")
        limit = self.ctx_len - 1
        if len(prompt) <= limit:
            return list(prompt)
        if prompt[0] == self.tok.bos_id:
            return [prompt[0], *_tail(prompt, limit - 1)]
        return _tail(prompt, limit)

    def _left_pad(self, seqs: list[list[int]]) -> tuple[Tensor, Tensor]:
        """Token ids and a True-for-real mask, both ``[B, longest]`` on the device."""
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.tok.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.bool)
        for i, seq in enumerate(seqs):
            ids[i, width - len(seq) :] = torch.tensor(seq, dtype=torch.long)
            mask[i, width - len(seq) :] = True
        return ids.to(self.device), mask.to(self.device)

    # -- scoring ------------------------------------------------------------------------------

    @torch.inference_mode()
    def score_continuations(
        self, contexts: list[list[int]], continuations: list[list[int]]
    ) -> list[ContinuationScore]:
        """Total log-probability of each continuation given its context.

        An empty continuation scores ``(0.0, 0)``. If context plus continuation exceed
        ``ctx_len``, the context is left-truncated (keeping a leading ``<|bos|>``); a continuation
        too long to leave even one context token is cut to what fits, and ``n_tokens`` reports how
        many tokens were actually scored.
        """
        if len(contexts) != len(continuations):
            raise ValueError(f"{len(contexts)} contexts but {len(continuations)} continuations")
        results = [ContinuationScore(0.0, 0) for _ in contexts]
        work: list[tuple[int, list[int], list[int]]] = []
        for i, (ctx, cont) in enumerate(zip(contexts, continuations, strict=True)):
            if not ctx:
                raise ValueError("empty context")
            if cont:
                work.append((i, *self._fit_pair(ctx, cont)))
        for lo in range(0, len(work), self._chunk_size()):
            chunk = work[lo : lo + self._chunk_size()]
            for (i, _, cont), total in zip(chunk, self._score_chunk(chunk), strict=True):
                results[i] = ContinuationScore(total, len(cont))
        return results

    def _fit_pair(self, ctx: Sequence[int], cont: Sequence[int]) -> tuple[list[int], list[int]]:
        cont = list(cont[: self.ctx_len - 1])  # at least one context token must fit
        room = self.ctx_len - len(cont)
        if len(ctx) <= room:
            return list(ctx), cont
        if ctx[0] == self.tok.bos_id:
            return [ctx[0], *_tail(ctx, room - 1)], cont
        return _tail(ctx, room), cont

    def _score_chunk(self, chunk: list[tuple[int, list[int], list[int]]]) -> list[float]:
        """Right-padded batch: causal attention keeps real tokens blind to the padding."""
        width = max(len(ctx) + len(cont) for _, ctx, cont in chunk)
        ids = torch.full((len(chunk), width), self.tok.pad_id, dtype=torch.long)
        for row, (_, ctx, cont) in enumerate(chunk):
            ids[row, : len(ctx) + len(cont)] = torch.tensor(ctx + cont, dtype=torch.long)
        with self._autocast():
            logits = self.model(ids.to(self.device))
        totals = []
        for row, (_, ctx, cont) in enumerate(chunk):
            # The logits at position p predict token p + 1, so the continuation starts at the
            # last context position.
            logp = logits[row, len(ctx) - 1 : len(ctx) - 1 + len(cont)].float().log_softmax(-1)
            target = torch.tensor(cont, dtype=torch.long, device=self.device)[:, None]
            totals.append(logp.gather(1, target).sum())
        return torch.stack(totals).tolist()

    # -- helpers ------------------------------------------------------------------------------

    def _chunk_size(self) -> int:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")
        return self.batch_size

    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()
