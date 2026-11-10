# Training-system upgrade: findings and plan (2026-11-10, session cut short by the usage limit)

The owner asked (2026-11-10) for an analysis and then an upgrade of the AI training system
(faster, more efficient, better/smarter/larger AIs), with **a plan of my own, not the earlier
2026-11-07 plan**, plus the removal of every bug found ("improve the whole thing").
**No product code was changed in this session.** Everything below is measured on the owner's
RTX 3050 with the game not running. Scripts: `docs/handoff/upgrade-2026-11-10/`.

## 1. Measured today (baseline)
- First AI (4x128, 12.6M tokens): 70.4 s. Model step 65%, **the 20 sample-writing pauses 20%**,
  data sampler 7%, snapshots and saves 2%. Medium 8x256, 6.3M tokens: 102 s, samples 24%.
- Raw model step, tokens a second at 8,192-token batches: tiny 271k, small 144k, medium 81k,
  large 10x384 43k, xl 12x512 25k (3.9 GB). Bigger batches add under 15%: the card is already full.
- Only 20-27% of a step's GPU time is matrix maths; the rest is memory-bound small operations.
- `torch.compile` is not available (no Triton on Windows). Ruled out: it would need a
  community compiler package on every friend's PC.

## 2. Findings, biggest first
1. **Casual typing breaks every AI (the main cause of "babbling").** 99.5% of user turns in the
   training conversations start with a capital and all end with . ? or !. "hello" is three rare
   tokens (`he|ll|o`), "Hello" is one. The same first AI answers "Hello" with "Hello! How can I
   help you today?" and "hello" with "dal, I don't know." Lowercase names ("france") and a
   missing "?" break even the reference judge. The owner's test prompts were all lowercase.
   Fix: (a) at training time, type a share of chat documents' user turns casually (lowercase,
   no final punctuation; replies untouched), in the sampler, no corpus rebuild;
   (b) at serving time, capitalise the first letter of a user turn (helps existing AIs at once).
   Experiment code: `casual.py`, `probe_casual.py`, `exp.py --casual 0.3` (not yet run).
2. **Answers are drawn too hot.** Same AI, well-formed replies (the gate's measure):
   first AI 50% at temperature 0.8, 77% at 0.6, 83-85% at 0.4-0.5; small brain 60% / 87% / 90%.
   Garble-like words 1.2% -> 0.7%. Plan: serve chat and customer answers at about 0.5-0.6;
   leave bench-v1 and the fingerprint as they are (old scores stay comparable).
3. **"Careful" is a handicap** (6e-4, about 4.5x below the best rate); the owner used it on every
   run. AdamW on the first-AI recipe, held-out loss: 1.5e-3 2.617, 2.68e-3 (default) 2.568,
   4e-3 2.546, 6e-3 2.591. Seed noise 0.005. Retune the slider so Careful is the safe optimum.
4. **A better optimizer learns about 1.3x more per token.** Muon (orthogonalised momentum for
   the weight matrices, Adam for the embedding and gains), same recipe: 2.468-2.492 against
   AdamW's best 2.546; AdamW needs 16.4M tokens (1.3x) for 2.484. Best so far: Muon rate
   0.01-0.02, embedding rate 1.2e-2; 0.04 and above is worse (no divergence, just poor).
   It costs about 6% per step when batched by matrix shape. Still to check: medium and long
   runs, continuation after growth, and that the slider keeps a real trade-off.
   Weight decay 0 = no change. Linear-to-zero schedule is worse; batch 4,096 no better and slower.
5. **Engine speed, same maths (exact on the CPU):** `F.rms_norm` instead of the hand-written
   norm = **+20-22%** (and slightly more accurate under bf16); loss inside autocast +2-5%;
   merging the q/k/v or gate/up matrices is slower (dropped); RoPE rewrite no gain so far.
   Trainer: write the probe samples in one batch and decode without masks when nothing is
   padded; plain ndarray view of the token memmap (sampler 0.78 -> 0.54 ms); snapshots on the
   GPU; time-based resume saves. Expected: first AI 70 s -> about 42 s; medium short 102 -> 63 s.
6. **Bugs found (not yet fixed):**
   - `worker/jobs.py` + `trainer._same_config`: the resume check compares `parent_dir`, a path
     on one PC, so a run with a parent that moves to another card fails with "belongs to a
     different run config", and that error is not retried. This is the owner's stuck v4 run.
     Fix: compare everything but the path (plus the parent's version id); a state that can't be
     resumed is discarded and the run starts over instead of failing.
   - `training/service.ts` `trainFailed`/`trainDone` delete the resume blob first; if that throws
     (a file still open on Windows) the world is never told and the run hangs "finishing".
     Fix: tell the world first, clean up best-effort.
   - A finished run whose upload fails is retrained from scratch on the retry.
   - Helpers carry no code version: an out-of-date friend's helper silently runs the old trainer.
   - Resume states grow with the brain (model + optimizer + snapshot): about 460 MB at 19M
     parameters and 960 MB at 40M, over the 512 MB blob limit; needed before larger brains.
   - `max_params = vram_mb * 10,000` is too loose for big brains (xl needs 3.9 GB).
7. Typo pages are 10.7% of the web pool and 2.5% of conversations at standard cleaning; measure
   garble again after items 1-4 before touching the cleaning thresholds.

## 3. The plan (mine), in build order
1. **Reliability:** the bugs in 2.6, each with a failing test first.
2. **Engine speed:** 2.5, with exactness tests and a before/after timing table.
3. **Talking:** casual-typing training and first-letter capitalisation (2.1), serving
   temperature (2.2), each kept only if `probe_casual.py` / `probe_quality.py` show the gain.
4. **Learning:** the optimizer and schedule that win the remaining experiments (2.4), the
   slider retuned (2.3), an engine version in the resume state; then rerun the feasibility gate.
5. **Bigger brains and longer runs** for new worlds (a new balance version; old worlds keep
   balance-v2): large 10x384 and xl 12x512, lengths up to about 30 minutes on an RTX 3050,
   smaller resume states, a true memory limit per card; then rebuild the rivals for new worlds.
6. Report to the owner in plain language with before/after answers.

## 4. Resume here
- Experiment results: `docs/handoff/upgrade-2026-11-10/results.jsonl` (one line per run).
  Sweep 2 (`sweep2.json`: slider ends, schedules, Muon details) was still running when the
  session stopped; its remaining lines are in the session scratchpad's `results.jsonl`.
- Next experiments: `exp.py --casual 0.3` against a twin without it, judged by
  `probe_casual.py`; Muon against AdamW on small and medium at 12.6M and 25M tokens and after
  growth; `--init zero`.
- Then write the spec (`docs/superpowers/specs/`) and an ADR with the chosen values, and build
  in the order above.
