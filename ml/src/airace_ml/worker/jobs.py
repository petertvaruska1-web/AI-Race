"""What a worker does with each job type (plan Task 18).

* ``train``: fetch the parent checkpoint, train the run with M1's trainer while streaming its
  events, then upload the new checkpoint (without its resume state).
* ``eval``: score a version on bench-v1, measure its personality and describe it against the
  other AIs the server names.
* ``generate``: one chat reply, streamed piece by piece.
* ``feedback``: the product's AI answers customer questions from each group, and each reply is
  scored by its question's check.

The model's words are passed on exactly as generated (R1).
"""

import json
import logging
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from airace_ml.battles import battle_challenges, score_challenge
from airace_ml.data.identity import data_identity
from airace_ml.evals.judge import Judge, check_judge_provenance
from airace_ml.evals.novelty import NoveltyIndex, check_index_provenance
from airace_ml.evals.suite import run_benchmarks, score_reply
from airace_ml.feedback import pick_questions
from airace_ml.paths import judge_dir, novelty_path, tokenizer_path
from airace_ml.personality.fingerprint import Fingerprint, describe, measure_fingerprint
from airace_ml.tokenizer import Tok
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import event_to_dict
from airace_ml.train.trainer import RESUME_DIR, has_resume_state, train_run
from airace_ml.worker.agent import JobContext
from airace_ml.worker.blobs import download_tar, upload_dir_tar
from airace_ml.worker.models import ModelCache

log = logging.getLogger("airace.worker")


def _finite(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def _spec_str(spec: dict[str, Any], key: str) -> str:
    v = spec.get(key)
    if not isinstance(v, str) or not v:
        raise ValueError(f"the job spec needs {key}")
    return v


class JobRunner:
    """The handlers for one worker, sharing its data, device and model cache."""

    def __init__(
        self,
        data_root: Path,
        device: torch.device | str,
        work_root: Path | None = None,
        models_dir: Path | None = None,
        bench_max_items: int | None = None,
        model_cache_size: int = 4,
    ) -> None:
        self.data_root = Path(data_root)
        self.device = torch.device(device)
        self.work_root = Path(work_root) if work_root else self.data_root / "worker" / "work"
        self.tok = Tok.load(tokenizer_path(self.data_root))
        self.models = ModelCache(
            Path(models_dir) if models_dir else self.data_root / "worker" / "models",
            self.tok,
            self.device,
            capacity=model_cache_size,
        )
        self.bench_max_items = bench_max_items
        self._creativity: tuple[Judge | None, NoveltyIndex | None] | None = None

    def handlers(self) -> dict[str, Callable[[dict[str, Any], JobContext], Any]]:
        return {
            "train": self.handle_train,
            "eval": self.handle_eval,
            "generate": self.handle_generate,
            "feedback": self.handle_feedback,
            "battle": self.handle_battle,
        }

    # -- train -------------------------------------------------------------------------------------

    def handle_train(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        run_id = _spec_str(spec, "runId")
        out_url = _spec_str(spec, "outTarUrl")
        if not isinstance(spec.get("config"), dict):
            # ValueError, not TypeError: the agent reports ValueErrors as not worth retrying.
            raise ValueError("the job spec needs a training config")  # noqa: TRY004
        # Kept per run, so a job that comes back to this worker resumes where it stopped.
        work = self.work_root / "runs" / run_id
        config = dict(spec["config"])
        parent_url = spec.get("parentTarUrl")
        if parent_url:
            parent = work / "parent"
            if not (parent / "meta.json").is_file():
                shutil.rmtree(parent, ignore_errors=True)
                download_tar(parent_url, parent)
            config["parent_dir"] = str(parent)
        else:
            config.pop("parent_dir", None)
        cfg = TrainRunConfig.from_json(json.dumps(config))
        cfg.validate()
        out = work / "out"
        resume = has_resume_state(out)
        if not resume:
            shutil.rmtree(out, ignore_errors=True)
        result = train_run(
            cfg,
            out_dir=out,
            data_root=self.data_root,
            device=self.device,
            on_event=lambda e: ctx.progress(event_to_dict(e)),
            resume=resume,
        )
        upload_dir_tar(out, out_url, exclude=(RESUME_DIR,))
        shutil.rmtree(work, ignore_errors=True)
        return {
            "steps": result.steps,
            "final_loss": _finite(result.final_loss),
            "heldout_losses": {k: _finite(v) for k, v in result.heldout_losses.items()},
            "status": result.status,
        }

    # -- eval --------------------------------------------------------------------------------------

    def creativity_tools(self) -> tuple[Judge | None, NoveltyIndex | None]:
        """The installed judge and novelty index, or ``None`` for one that is missing or was built
        from other data; creativity is then reported as not measured."""
        if self._creativity is None:
            identity = data_identity(self.data_root)
            judge = index = None
            try:
                judge = Judge.load(judge_dir(self.data_root), self.device)
                check_judge_provenance(judge_dir(self.data_root), identity)
            except (OSError, ValueError, TypeError) as e:
                log.warning("creativity will not be measured: no usable judge (%s)", e)
                judge = None
            try:
                index = NoveltyIndex.load(novelty_path(self.data_root))
                check_index_provenance(index, identity)
            except (OSError, ValueError, TypeError) as e:
                log.warning("creativity will not be measured: no usable novelty index (%s)", e)
                index = None
            self._creativity = (judge, index)
        return self._creativity

    def handle_eval(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        lm, lock = self.models.get(_spec_str(spec, "versionId"), _spec_str(spec, "modelTarUrl"))
        judge, index = self.creativity_tools()
        if judge is None or index is None:
            judge = index = None  # creativity needs both
        with lock:
            report = run_benchmarks(
                lm, lm.tok, judge=judge, novelty=index, max_items_per_category=self.bench_max_items
            )
            fp = measure_fingerprint(lm, lm.tok)
        population = []
        for traits in spec.get("population") or []:
            try:
                population.append(Fingerprint.from_dict({"traits": traits, "samples": []}))
            except (TypeError, ValueError):
                continue  # traits from another version of the measure: not comparable
        return {
            "report": report.to_dict(),
            "traits": dict(fp.traits),
            "personalityWords": describe(fp, population),
        }

    # -- generate ----------------------------------------------------------------------------------

    def handle_generate(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        history = spec.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError("the job spec needs a conversation")
        turns = [(str(role), str(text)) for role, text in history]
        lm, lock = self.models.get(_spec_str(spec, "versionId"), _spec_str(spec, "modelTarUrl"))
        with lock:
            text = lm.stream_chat_reply(
                turns,
                max_new_tokens=int(spec.get("maxNewTokens", 96)),
                temperature=float(spec.get("temperature", 0.8)),
                top_p=float(spec.get("topP", 0.95)),
                seed=int(spec.get("seed", 0)),
                on_text=ctx.chunk,
            )
        return {"text": text}

    # -- feedback ----------------------------------------------------------------------------------

    def handle_feedback(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        groups = spec.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("the job spec needs customer groups")
        per_group = int(spec.get("perGroup", 1))
        seed = int(spec.get("seed", 0))
        lm, lock = self.models.get(_spec_str(spec, "versionId"), _spec_str(spec, "modelTarUrl"))
        results = []
        with lock:
            for group in groups:
                for item in pick_questions(str(group), per_group, seed):
                    reply = lm.chat_reply(
                        [("user", item.prompt)], max_new_tokens=96, temperature=0.8, seed=seed
                    )
                    results.append(
                        {
                            "group": group,
                            "questionId": item.id,
                            "question": item.prompt,
                            "reply": reply,
                            "score": score_reply(item, reply),
                        }
                    )
        return {"results": results}

    # -- battle ------------------------------------------------------------------------------------

    def handle_battle(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        """Both AIs answer the same fresh challenges; each reply streams as ``battleText``
        progress, then is scored by the challenge's check (stories: the judge)."""
        skill = _spec_str(spec, "skill")
        rounds = int(spec.get("rounds", 3))
        seed = int(spec.get("seed", 0))
        sides = {}
        for side in ("a", "b"):
            s = spec.get(side)
            if not isinstance(s, dict):
                raise ValueError(f"the battle spec needs side {side!r}")  # noqa: TRY004 (not retried)
            sides[side] = (_spec_str(s, "versionId"), _spec_str(s, "modelTarUrl"))
        challenges = battle_challenges(skill, rounds, seed)
        judge = index = None
        if skill == "creativity":
            judge, index = self.creativity_tools()
            if judge is None or index is None:
                raise ValueError("creativity battles need the judge and the novelty index")
        story = skill == "creativity"
        out = []
        for k, item in enumerate(challenges):
            answers = {}
            for side in ("a", "b"):
                version_id, url = sides[side]
                lm, lock = self.models.get(version_id, url, transient=side == "b")
                with lock:
                    reply = lm.stream_chat_reply(
                        [("user", item.prompt)],
                        max_new_tokens=96,
                        temperature=0.9 if story else 0.8,
                        top_p=0.95,
                        seed=seed + k,
                        on_text=lambda t, k=k, side=side: ctx.progress(
                            {"kind": "battleText", "round": k, "side": side, "text": t}
                        ),
                    )
                score = score_challenge(item, reply, tok=lm.tok, judge=judge, novelty=index)
                answers[side] = {"reply": reply, "score": float(score)}
            out.append({"prompt": item.prompt, **answers})
        return {"rounds": out}
