"""Scripted stand-ins for a language model, so benchmark scoring can be tested exactly."""

from collections.abc import Callable, Sequence

from airace_ml.evals.suite import Suite
from airace_ml.infer.lm import ContinuationScore, Generation
from airace_ml.skills.types import CheckItem, ExactItem, MCItem, PairItem
from airace_ml.tokenizer import Tok, encode_chat, encode_doc

WRONG = -10.0


class ScriptedLM:
    """A :class:`~airace_ml.infer.lm.LanguageModel` whose replies and scores are plain functions.

    ``reply(prompt_ids)`` gives the text of the reply to a prompt (default: nothing) and
    ``score(context, continuation)`` the summed log-probability of a continuation (default: 0.0).
    Every reply token gets probability 0.5, and the reply is never cut to ``max_new_tokens``.
    """

    def __init__(
        self,
        tok: Tok,
        ctx_len: int = 256,
        reply: Callable[[list[int]], str] | None = None,
        score: Callable[[list[int], list[int]], float] | None = None,
    ) -> None:
        self.tok = tok
        self.ctx_len = ctx_len
        self._reply = reply if reply is not None else lambda prompt: ""
        self._score = score if score is not None else lambda context, continuation: 0.0

    def score_continuations(
        self, contexts: list[list[int]], continuations: list[list[int]]
    ) -> list[ContinuationScore]:
        if len(contexts) != len(continuations):
            raise ValueError(f"{len(contexts)} contexts but {len(continuations)} continuations")
        return [
            ContinuationScore(float(self._score(list(ctx), list(cont))), len(cont))
            for ctx, cont in zip(contexts, continuations, strict=True)
        ]

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
        generations = []
        for prompt in prompts:
            ids = self.tok.encode(self._reply(list(prompt)))
            generations.append(Generation(ids, [0.5] * len(ids), [0.5] * len(ids), True))
        return generations


def _prompt(tok: Tok, prompt: str, chat: bool) -> tuple[int, ...]:
    if chat:
        return tuple(encode_chat(tok, [("user", prompt)], add_generation_prompt=True))
    return tuple(encode_doc(tok, prompt))


def answer_key_lm(tok: Tok, suite: Suite) -> ScriptedLM:
    """A model that knows every answer of ``suite``.

    It scores the correct option and the good sentence of a pair at 0.0 and everything else at
    -10, and replies with an exact item's first answer or a checked item's reference. Lookups
    are by exact token ids, encoded the way the benchmark specifies, so a prompt or option
    encoded any other way gets -10 and no reply. Raises ``ValueError`` if two items disagree on
    the same key.
    """
    scores: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    replies: dict[tuple[int, ...], str] = {}

    def put(table: dict, key: tuple, value: object) -> None:
        if table.setdefault(key, value) != value:
            raise ValueError(f"answer key conflict at {key}")

    for items in suite.items.values():
        for item in items:
            if isinstance(item, PairItem):
                bos = (tok.bos_id,)
                put(scores, (bos, tuple(tok.encode(item.good))), 0.0)
                put(scores, (bos, tuple(tok.encode(item.bad))), WRONG)
            elif isinstance(item, MCItem):
                context = _prompt(tok, item.prompt, item.chat)
                lead = "" if item.chat else " "
                for k, option in enumerate(item.options):
                    value = 0.0 if k == item.answer_index else WRONG
                    put(scores, (context, tuple(tok.encode(lead + option))), value)
            elif isinstance(item, ExactItem):
                put(replies, _prompt(tok, item.prompt, item.chat), item.answers[0])
            elif isinstance(item, CheckItem):
                put(replies, _prompt(tok, item.prompt, item.chat), item.reference)

    def reply(prompt: list[int]) -> str:
        return replies.get(tuple(prompt), "")

    def score(context: list[int], continuation: list[int]) -> float:
        return scores.get((tuple(context), tuple(continuation)), WRONG)

    return ScriptedLM(tok, reply=reply, score=score)
