# Final whole-branch review of Milestone 1 (code at 93f2737)

Written by the controller on 2026-10-20.

**Inputs:**
- 18 finders (5 subsystems × 3 lenses, plus 3 specialists), a completeness critic and 6 extra finders. The extras checked real Hugging Face rows through Firecrawl.
- De-duplication into 56 findings.
- Adversarial verification by 13 Sonnet skeptics, with all three lenses applied to every finding.
- The controller's rulings.

Machine-readable inputs: `agent-results.json` (finders, critic, dedup) and `verify/verify-results.json` (verdicts). Implementer briefs: `docs/handoff/fix-wave/group-*.md`.

## 1. Verdict
**Not yet ready for the owner's real runs.** One defect makes a third of the coding benchmark unanswerable for every model (F1).

Several others would fail or silently degrade the next real steps:
- the judge build may not fit in 8 GB (F2)
- a tiny test build can freeze the wrong tokenizer (F3)
- the hour-long data build has no retry or resume (F11)
- real Gutenberg and TinyStories text arrives damaged (F12, F14)

Several measurements also give credit to models that know nothing, or decide on noise:
- knowledge base rates (F5)
- constant-output creativity (F6)
- the G4 noise floor (F9)
- single-seed verdicts (F10)
- G7(a)'s garble measure (F13)

All of these are fixable in the cloud with small or medium changes, and none needs a redesign. After the fix wave in section 2 and its re-review, the branch is ready for Task 15, the evaluator builds and the gate.

## 2. Findings to fix (43), most severe first
| ID | Severity | Group | Finding |
|---|---|---|---|
| F1 | Critical | A | Function-completion bench prompts end on a bare newline token that training never follows with indentation, so no trained model can pass the 50 functi |
| F2 | Important | D | Trainer runs a whole batch_tokens batch in one forward/backward with no micro-batching, so the reference judge recipe (8x384x256, 32768 tokens/step) l |
| F3 | Important | C | A full corpus build silently reuses (and freezes as tok-v1) a tokenizer and shared vocab files left by an earlier tiny build in the same data root; th |
| F5 | Important | A | Knowledge (and consistency) bench answers follow the KB's skewed base rates while chance is set to 1/options (MC) and 0 (exact), so a model that ignor |
| F6 | Important | E | Creativity rewards degenerate output: the same story for all 24 prompts scores 21-52, and a short stock sentence gets copy-level credit that can beat  |
| F9 | Important | F | G4's 'measurably different' clause (seed distance > 0.05) sits about 40x below the fingerprint's own sampling noise, so it can never fail and G4 passe |
| F10 | Important | F | Spec §4.12 says '3 seeds each', but G2, G5, G6 and G7 (and G1) are decided on seed 1 alone with zero-margin thresholds, so verdicts can flip by seed |
| F11 | Important | C | The hour-long full build has no retry, resume or source pre-flight: a Hub error when any of the 14 sources is opened, a 500/502/503 lasting over ~25 s |
| F12 | Important | B | Real sedthh/gutenberg_english TEXT puts a blank line after every wrapped line, so every books passage has blank lines mid-sentence and real paragraph  |
| F13 | Important | F | G7(a) cannot be expected to pass on real data: thorough cleaning lowers the unknown-word share of real web training text only to 79-82% of light's, at |
| F14 | Important | B | About 7.5% of the TinyStories rows the build reads contain real mojibake ('â€œ', 'â€™', 'â€'), which enters the creative corpus tagged noise 'none' an |
| F4 | Minor | E | The judge and the novelty index record nothing about the tokenizer and corpora they were built from: after a rebuild, build-judge reuses old weights w |
| F7 | Minor | D | Replay ('Review what it knows') of a parent whose mix included Notebook or Coaching fails with a MixtureError, or silently trains the child's current  |
| F8 | Minor | F | G5 continuation runs retrain on the starter's exact token stream, so the 'extra tokens' are a verbatim second epoch |
| F16 | Minor | B | Full-scale token quotas are extrapolated from the first 200 accepted docs at a fixed 3.6 bytes/token and never checked against real totals, so dataset |
| F17 | Minor | C | The full build prints nothing until it finishes, so a stall, a rate-limit loop or an indefinitely hung Hub metadata call (no timeout) looks exactly li |
| F18 | Minor | C | CLAUDE.md's setup and corpus-build commands omit the 'content' extra, so the owner's next step (full build) fails at once; the error's own fix uses ba |
| F19 | Minor | F | A refused Windows rename of the post-training record (judge train_config.json, gate gate_run.json) throws away a finished training: os.replace has no  |
| F20 | Minor | F | Gate reruns into the same --out with --quick, another device or another first seed silently overwrite the finished full-scale runs (same run names for |
| F21 | Minor | F | build-judge and train silently fall back to the CPU when CUDA is unavailable; on the CPU the judge needs about 10.7 GB of RAM per step and takes days |
| F22 | Minor | D | Tokens actually trained per step differ from batch_tokens; validate() accepts batch_tokens < ctx_len, which trains up to 2-2.56x the paid token budget |
| F23 | Minor | D | Lone UTF-16 surrogates in notebook, coaching or probe_prompts pass validate() but crash the tokenizer with TypeError; a bad probe prompt only fails mi |
| F24 | Minor | A | Knowledge exact items keep the default 8-token reply budget, but bench-v1 asks for 'Bandar Seri Begawan', which a 4096-entry BPE likely splits into 9- |
| F25 | Minor | A | Nothing pins the bench-v1 item set: it is regenerated from code, KB data and numpy streams at runtime, so it can drift silently under the same version |
| F26 | Minor | B | Code dataset copies carry typos that break programs, no cleaning level can remove them, and about 44% also survive dedup, contradicting the Task 14 ru |
| F27 | Minor | B | Boilerplate noise barely moves the quality score on real-length documents, so cleaning almost never removes it |
| F28 | Minor | C | airace-content does not switch stdout to UTF-8 (unlike airace-ml), so the final summary crashes when stdout is redirected on Windows and the path has  |
| F29 | Minor | C | Sources are streamed from an unpinned Hugging Face revision and the manifest does not record which revision was used |
| F30 | Minor | B | Running the same full build twice gives a different corpus, because quota counting switches from the byte estimate to the frozen tokenizer |
| F33 | Minor | E | A model with non-finite logits crashes creativity, the fingerprint and chat (torch.multinomial RuntimeError) instead of scoring finite |
| F34 | Minor | E | Definitions that should be shared are duplicated: the G2 well-formed check counts words differently from creativity and the gate, and the 96-token sto |
| F38 | Minor | F | CLI commands print Python tracebacks for a damaged corpus, a file-system error or a corrupt tokenizer.json (bare Exception from tokenizers), breaking  |
| F39 | Minor | F | `train --resume` on a folder holding a finished run tells the user to run without --resume, which overwrites the finished model |
| F41 | Minor | F | Every gate rerun regenerates report.md from scratch, discarding any 'Iterations' notes written into it |
| F42 | Minor | E | A slow test writes a judge into the shared session-scoped tiny_data_root, so running slow and default tests together fails other tests |
| F43 | Minor | E | The GPU test asserts exact greedy-token equality between CPU fp32 and CUDA bf16, but the fp32 top-2 margin (0.014 logits) is far below bf16 rounding e |
| F44 | Minor | A | Two brief tests check nothing: an empty tag breakdown, and dataset names that are single letters found in every error message |
| F45 | Minor | B | No test runs non-ASCII text through the corpus build (Review Focus 5): every source fixture is pure ASCII |
| F46 | Minor | E | No test checks that the gate's data identity changes when the corpus is rebuilt, and the default suite never runs the gate's real input loading |
| F47 | Minor | D | Three trainer resume branches, including the Windows 'refused rename' path, are never executed by any test |
| F52 | Minor | B | Topic keywords mis-tag real text on dates and false friends: month and weekday names alone make about 7% of web docs 'school' |
| F53 | Minor | A | Free-answer animal-class items reject the plural answers people usually give ('mammals', 'birds') |
| F54 | Minor | A | Planet-order free answers accept '5th' but not the bare digit '5' |

Each finding's ruling, exact fix and required test are in the group brief (`docs/handoff/fix-wave/group-<G>.md`) and the ledger.

## 3. Refuted findings (11)
10 refutations are accepted; F6 is overruled and moves to section 2.

- **F31** known_vocab.txt is built with an ASCII-only word rule, but the gate's garble measure reads words as Unicode letter runs,: deliberate Unicode rule in the gate (test_gate.py:331); accents add the same floor to both G7 arms; the finder's fix would hide the mojibake signal G7 measures.
- **F35** G2 legibility passes a constant-reply or prompt-parroting model, and the report shows only the 10 'open' exchanges: G2 is a legibility criterion by spec §4.12.2; responsiveness is measured elsewhere; a constant reply across 20 prompts at T=0.8 is not plausible; outcome.json keeps all 20 exchanges.
- **F36** G7(a) garble rate rewards empty or looping completions, so a degenerate thorough-cleaning model can pass: no mechanism makes only the thorough arm degenerate (same mix, budget and sampling); the real G7(a) risk is F13.
- **F37** A gate run that stopped unstable is retrained on every rerun, so on CUDA a gate criterion can be passed by rerolling: deliberate (Task 20 dispatch ruling 5; pinned by a test); retraining an unstable run gives a transient glitch another chance and costs minutes.
- **F40** The CLI prints raw model text, so control and escape characters from the model can clear or rewrite the console (the gat: the corpus build strips control characters, so trained models do not emit them; damaging sequences are vanishingly improbable on the owner's own console.
- **F48** The default suite takes 118-137 s here, with little margin on Windows; trainer tests repeat the same reference training : the suite measured 114 s against the 3-minute budget; the finder's figure was taken under parallel load.
- **F49** Benchmark items are not tagged topic x skill x format as spec §4.7 requires; most categories carry one axis and none car: tags follow the plan's reviewed scheme, are metadata only and can be added later by item id without re-measuring; their consumer is M4 (recorded as an M4 note).
- **F50** Spec §4.7's tiny mini-eval during training for live charts has no implementation and no recorded deferral: the plan's Task 8 telemetry contract was implemented exactly; the live mini-eval belongs to the M2 live-training view (recorded as an M2 note).
- **F55** Birds get 'reptile' as a distractor and as a planted false fact, though under the modern grouping birds are reptiles: the KB uses school-level animal classes, where 'a peacock is a kind of reptile' is clearly false.
- **F56** 'A newborn hen' is a self-contradictory bench subject (a hen is an adult female chicken): 'a baby hen is called a chick' follows the KB's 'baby <adult name>' convention; no model is marked wrong unfairly.
- **F6** (creativity rewards constant output): the refutation is **overruled**. The spec defines creativity as a product with diversity, and ADR 0001 requires degenerate models to score near 0. The fix is narrowed to a real diversity multiplier; see group E.

## 4. Deferred or done

- **F15** Simplicity thresholds (0.35 web/edu, 0.30 Simple Wikipedia) keep about 97-100% of real documents, so the spec's simplici: No code change in the cloud: a product tuning call that needs the real simple_vocab. Task 15 step (before the full build): measure per-source simplicity quantiles on the first ~2000 rows of fineweb, fineweb_edu, simplewiki and gutenberg with the real simple_vocab; set simplicity_min at a chosen keep-rate in assemble.py RECIPES (about 0.6 keeps the simpler ~90% of FineWeb, 0.7 about 80%), or record that the filter stays deliberately mild. Record the numbers.
- **F32** Player notebook/coaching data, already trained on in M1, is invisible to the novelty index and is not screened against b: No M1 code change (notebook and coaching ship with M4 per the spec's milestone table). Obligations recorded in the handoff: (1) creativity scoring must count the model's own notebook and coaching text as 'seen' (NoveltyIndex.novelty(seen=)); (2) screen notebook and coaching entries against bench-v1 prompts before training once discovery cards can show real prompts.
- **F51** Progress log and handoff resume guide are stale (last updated after Task 11): Fixed by the controller on 2026-10-14 (commits 52415fb, 4d6af96).

## 5. Notes for later milestones
- **M2:**
  - A continuation run must never reuse its parent's sampler stream (from F8).
  - Add the live mini-eval event when the live-training view is built (from F50).
- **M4:**
  - Notebook and coaching novelty, and screening them against bench-v1 prompts (F32).
  - The full topic × skill × format tag space for discoveries; tags join by item id without re-measuring (from F49).

## 6. What the cloud could not verify (the local session must check)
- **F2:** the real GPU memory of a judge step. Run `python -m uv run pytest -m gpu` on the RTX 3050 after the fix wave. Its new memory test must pass before the 1-hour judge build.
- **F11 and F29:** real Hub behaviour, and pinning each source's commit sha in Task 15.
- **F15:** the simplicity thresholds, measured on real data before the full build (`docs/handoff/README.md` §5).
- **F13:** the data-level G7(a) ratio printed by the gate. If it is above about 0.7, strengthen web noise or the cleaning thresholds before trusting G7(a).
- **F24:** the tok-v1 reply-budget test, skipped until a tokenizer exists. It must pass after Task 15.
- **F10:** the real-gate time, about 30–40 minutes longer than first estimated.
