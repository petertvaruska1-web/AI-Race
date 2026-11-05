"""What a worker does with each job type (plan Task 18).

* ``train``: fetch the parent checkpoint, train the run with M1's trainer while streaming its
  events, then upload the new checkpoint (without its resume state).
* ``eval``: score a version on bench-v1, measure its personality and describe it against the
  other AIs the server names.
* ``generate``: one chat reply, streamed piece by piece.
* ``feedback``: the product's AI answers customer questions from each group, and each reply is
  scored by its question's check.
* ``battle``: two AIs answer the same fresh challenges, side by side.
* ``minigame``: one or two AIs play a Playground game, step by step (``minigames/run.py``).
* ``keynote``: the product's AI answers its launch keynote's demo question, live.
* ``appReview``: a partner app's users try the AI on fresh questions in the app's skill.
* ``interview``: score the answers a player published from an interview (no player model).

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

from airace_ml.battles import battle_challenges, battle_prompt, expected_answer, score_challenge
from airace_ml.data.identity import data_identity
from airace_ml.discoveries import eval_signals, find_discoveries
from airace_ml.evals.creativity import score_story
from airace_ml.evals.judge import Judge, check_judge_provenance
from airace_ml.evals.novelty import NoveltyIndex, check_index_provenance
from airace_ml.evals.suite import run_benchmarks, score_reply
from airace_ml.feedback import GROUPS, customer_pool, pick_questions
from airace_ml.interviews import score_answer
from airace_ml.minigames.content import game_items
from airace_ml.minigames.run import play
from airace_ml.paths import judge_dir, novelty_path, tokenizer_path
from airace_ml.personality.fingerprint import Fingerprint, describe, measure_fingerprint
from airace_ml.tokenizer import Tok
from airace_ml.train.config import TrainRunConfig
from airace_ml.train.events import event_to_dict
from airace_ml.train.trainer import CHECKPOINT_EVERY, RESUME_DIR, has_resume_state, train_run
from airace_ml.worker.agent import JobContext
from airace_ml.worker.blobs import download_tar, upload_dir_tar
from airace_ml.worker.models import ModelCache
from airace_ml.worker.moving import RESUME_UPLOAD_EVERY_S, ResumeUploader, restore_resume

log = logging.getLogger("airace.worker")
MAX_INTERVIEW_ANSWERS = 5


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
        # Steps between the trainer's resume saves, and seconds between uploads of them (Fair mode).
        self.checkpoint_every = CHECKPOINT_EVERY
        self.resume_upload_every_s = RESUME_UPLOAD_EVERY_S
        self._creativity: tuple[Judge | None, NoveltyIndex | None] | None = None

    def handlers(self) -> dict[str, Callable[[dict[str, Any], JobContext], Any]]:
        return {
            "train": self.handle_train,
            "eval": self.handle_eval,
            "generate": self.handle_generate,
            "feedback": self.handle_feedback,
            "battle": self.handle_battle,
            "minigame": self.handle_minigame,
            "keynote": self.handle_keynote,
            "appReview": self.handle_app_review,
            "interview": self.handle_interview,
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
            # Fair mode: another card may have saved this run's progress to the host.
            from_url = spec.get("resumeFromUrl")
            if isinstance(from_url, str) and from_url:
                resume = restore_resume(from_url, out) and has_resume_state(out)
                if not resume:
                    shutil.rmtree(out, ignore_errors=True)
        resume_url = spec.get("resumeUrl")
        uploader = (
            ResumeUploader(resume_url, self.resume_upload_every_s)
            if isinstance(resume_url, str) and resume_url
            else None
        )
        try:
            result = train_run(
                cfg,
                out_dir=out,
                data_root=self.data_root,
                device=self.device,
                on_event=lambda e: ctx.progress(event_to_dict(e)),
                resume=resume,
                checkpoint_every=self.checkpoint_every,
                on_resume_saved=uploader.saved if uploader else None,
            )
        finally:
            if uploader:
                uploader.close()
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
        legibility_judge = judge  # well-formed replies need only the judge
        if judge is None or index is None:
            judge = index = None  # creativity needs both
        with lock:
            report = run_benchmarks(
                lm, lm.tok, judge=judge, novelty=index, max_items_per_category=self.bench_max_items
            )
            fp = measure_fingerprint(lm, lm.tok)
            signals = eval_signals(report, fp, legibility_judge)
        population = []
        for traits in spec.get("population") or []:
            try:
                population.append(Fingerprint.from_dict({"traits": traits, "samples": []}))
            except (TypeError, ValueError):
                continue  # traits from another version of the measure: not comparable
        previous = spec.get("previous")
        peers = [p for p in spec.get("peers") or [] if isinstance(p, dict)]
        thresholds = spec.get("thresholds") or {}
        return {
            "report": report.to_dict(),
            "traits": dict(fp.traits),
            "personalityWords": describe(fp, population),
            "signals": signals,
            "breakdown": {key: [value, n] for key, (value, n) in report.tag_breakdown().items()},
            "discoveries": find_discoveries(
                report, fp, previous if isinstance(previous, dict) else None, peers, thresholds
            ),
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
        out = []
        for k, item in enumerate(challenges):
            # Asked as the benchmark asks its kind, so each AI shows what it can really do.
            prompt, chat, max_new_tokens, temperature = battle_prompt(item)
            answers = {}
            for side in ("a", "b"):
                version_id, url = sides[side]
                lm, lock = self.models.get(version_id, url, transient=side == "b")
                with lock:
                    reply = lm.stream_reply(
                        prompt,
                        chat=chat,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        seed=seed + k,
                        on_text=lambda t, k=k, side=side: ctx.progress(
                            {"kind": "battleText", "round": k, "side": side, "text": t}
                        ),
                    )
                score = score_challenge(item, reply, tok=lm.tok, judge=judge, novelty=index)
                answers[side] = {"reply": reply, "score": float(score)}
            out.append({"prompt": item.prompt, "expected": expected_answer(item), **answers})
        return {"rounds": out}

    # -- minigame ----------------------------------------------------------------------------------

    def handle_minigame(self, spec: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
        """Plays a Playground game with one AI (or two in a duel; ``b`` is the rival), streaming
        each step as ``minigameStep`` and each typed reply's pieces as ``minigameText``."""
        game = _spec_str(spec, "game")
        ladder = game_items(
            game,
            _spec_str(spec, "difficulty"),
            int(spec.get("seed", 0)),
            spec.get("settings") or {},
        )
        raw = spec.get("sides")
        if not isinstance(raw, list) or not raw:
            raise ValueError("the minigame spec needs a side to play")
        sides = {}
        for s in raw:
            if not isinstance(s, dict) or s.get("side") not in ("a", "b"):
                raise ValueError("each minigame side is a or b")
            sides[s["side"]] = self.models.get(
                _spec_str(s, "versionId"), _spec_str(s, "modelTarUrl"), transient=s["side"] == "b"
            )
        story_score: Callable[[str, str], float] | None = None
        if game == "story":
            judge, index = self.creativity_tools()
            if judge is not None and index is not None:

                def story_score(story: str, prompt: str) -> float:
                    return score_story(story, prompt, self.tok, judge, index)

        result = play(
            game,
            ladder,
            sides,
            story_score=story_score,
            seed=int(spec.get("seed", 0)),
            on_step=lambda side, step: ctx.progress({"kind": "minigameStep", "side": side, **step}),
            on_text=lambda side, attempt, text: ctx.progress(
                {"kind": "minigameText", "side": side, "attempt": attempt, "text": text}
            ),
        )
        return {"sides": result}

    # -- interview ---------------------------------------------------------------------------------

    def handle_interview(self, spec: dict[str, Any], ctx: JobContext | None) -> dict[str, Any]:
        """Scores each published answer: by its card's check when it has a right answer, else for
        how fluently it reads (with the judge when one is installed)."""
        answers = spec.get("answers")
        if not isinstance(answers, list) or not 1 <= len(answers) <= MAX_INTERVIEW_ANSWERS:
            raise ValueError("the interview spec needs 1 to 5 answers")
        judge, _ = self.creativity_tools()
        scored = []
        for answer in answers:
            card = answer.get("cardId")
            score, checked = score_answer(
                card if isinstance(card, str) else None, str(answer.get("reply", "")), judge
            )
            scored.append({"score": min(max(score, 0.0), 1.0), "checked": checked})
        return {"answers": scored}

    # -- keynote -----------------------------------------------------------------------------------

    def handle_keynote(self, spec: dict[str, Any], ctx: JobContext | None) -> dict[str, Any]:
        """The launch keynote's live demo: the AI answers one customer question, as customers ask."""
        question_id = _spec_str(spec, "questionId")
        item = next((q for g in GROUPS for q in customer_pool(g) if q.id == question_id), None)
        if item is None:
            raise ValueError(f"no customer question {question_id!r}")
        lm, lock = self.models.get(_spec_str(spec, "versionId"), _spec_str(spec, "modelTarUrl"))
        with lock:
            reply = lm.chat_reply(
                [("user", item.prompt)],
                max_new_tokens=96,
                temperature=0.8,
                seed=int(spec.get("seed", 0)),
            )
        return {
            "questionId": item.id,
            "prompt": item.prompt,
            "reply": reply,
            "score": score_reply(item, reply),
        }

    # -- app review --------------------------------------------------------------------------------

    def handle_app_review(self, spec: dict[str, Any], ctx: JobContext | None) -> dict[str, Any]:
        """A partner app's users try the AI on fresh challenges in the app's skill, scored like a
        battle round (stories by the judge)."""
        skill = _spec_str(spec, "skill")
        challenges = battle_challenges(skill, int(spec.get("n", 2)), int(spec.get("seed", 0)))
        judge = index = None
        if skill == "creativity":
            judge, index = self.creativity_tools()
            if judge is None or index is None:
                raise ValueError("story app reviews need the judge and the novelty index")
        lm, lock = self.models.get(_spec_str(spec, "versionId"), _spec_str(spec, "modelTarUrl"))
        reviews = []
        with lock:
            for k, item in enumerate(challenges):
                prompt, chat, max_new_tokens, temperature = battle_prompt(item)
                reply = lm.stream_reply(
                    prompt,
                    chat=chat,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    seed=int(spec.get("seed", 0)) + k,
                    on_text=lambda _t: None,
                )
                score = score_challenge(item, reply, tok=lm.tok, judge=judge, novelty=index)
                reviews.append({"prompt": item.prompt, "reply": reply, "score": float(score)})
        return {"reviews": reviews}
