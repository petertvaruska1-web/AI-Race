# SDD ledger — plan: docs/superpowers/plans/2026-10-06-m1-it-learns.md

Spec: docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md (reachable)
Branch: m1-it-learns (from main @ c28acd2), working in place (no worktree)

## Setup rulings
- Ruling: work on branch `m1-it-learns` in place, not in a git worktree — the GPU venv (~3 GB) and generated data/ (corpora, judge, runs) would be duplicated or deleted on worktree cleanup, and the owner wants to see work in the project folder — cost if wrong: none to main; worst case a later move to a worktree.
- Ruling: proceed from plan to execution without a plan-review pause — the owner explicitly instructed autonomous execution after creative questions, and named subagent-driven development — cost if wrong: rework if the plan misses intent (plan is committed and visible).

## Pre-flight scan

| Pair / task | Produces vs consumes | Finding |
|---|---|---|
| T1→all | conftest.py, tests/__init__.py created; pytest addopts `-m 'not slow and not gpu'` | OK (CLI `-m gpu` overrides addopts) |
| T2→T5,T6,T8,T14,T16-19 | Tok, encode_doc, encode_chat, decode(skip_special) | OK, names consistent |
| T3→T4 | block layout names attn.wq/wk/wv/wo, mlp.w_gate/w_up/w_down, RMSNorm.eps buffer | OK |
| T3→T5 | forward(idx, positions, key_padding_mask, kv_cache), new_kv_cache() | OK |
| T5 tests | use `tiny_lm.model` | Interface omits attribute names → Ruling R3 |
| T6→T8 | load_pools(root, weights, prep, purchases, custom); DocPool.from_docs drops empties | OK; T8 notebook-blank test relies on it |
| T6 conftest→T20 | tiny_data_root lacks known_vocab.txt / false_facts.json needed by quick gate | Gap → Ruling R6 |
| T7→T8,T17 | TrainRunConfig, steps/batch_size properties, learning_style | OK; judge cfg 8x384x256 ≈ 15.7M params valid |
| T8 self | loss_override NaN at fixed step + rollback to snapshot: if step counter rewinds, test loops forever | Conflict → Ruling R4 |
| T8 self | stop_after_steps with checkpoint_every=10, stop at 20 | Ambiguous when resume state is saved → Ruling R5 |
| T9→T12 | run_program, call_function | OK |
| T9 self | `2 ** 10` expects "Unsupported: Pow" (operator name, not BinOp) | Clarify → Ruling R7 |
| T10→T11,T12,T14,T16 | TextDoc, MCItem, ExactItem, CheckItem(reference), PairItem, skill_rng, reserved_for_bench | OK |
| T10→T16 | consistency_groups 40×3 = 120; knowledge 150 MC + 50 exact = 200 | OK matches T16 counts |
| T11/T12→T16 | reasoning 200, pattern 150, coding 100+50, instruction 120, grammar 200 | OK |
| T13 self | SimHash Hamming ≤3 must catch a one-word edit in a 40-word doc (~8 bits expected flip) | Test infeasible → Ruling R1 |
| T13→T14 | inject_noise, cluster_near_duplicates, quality_score, tag_topic | OK after R1 (T14 only calls cluster_near_duplicates) |
| T14 self | tiny scale via fixture rows; heldout min 8 docs | OK |
| T16→T17,T18 | tests/fakes.py ScriptedLM(reply, score) | OK |
| T17→T19,T20 | build_novelty_index, build_judge, Judge, is_well_formed | OK |
| T18 self | register test words must be in lexicons | Clarify → Ruling R8 |
| T19→T20 | gate subcommand placeholder exit 2 until T20 | OK |
| T20 self | evaluator tests arithmetic checked | OK |

## Pre-flight rulings
- R1 Ruling: T13 dedup uses MinHash, not SimHash — `minhash_signature(text: str, num_perm: int = 64) -> np.ndarray[uint64]` over word 3-shingles; `cluster_near_duplicates(texts, threshold: float = 0.7) -> (cluster int32, canonical bool)` with exact normalized-hash matches + LSH 16 bands × 4 rows, candidates verified by estimated Jaccard ≥ threshold, union-find — SimHash cannot meet the plan's own one-word-edit test — cost if wrong: none downstream (only T14 calls it, by the cluster function).
- R3 Ruling: T5 `TorchLM` exposes public attributes `model`, `tok`, `device`, `ctx_len` — tests use `tiny_lm.model` — cost if wrong: none.
- R4 Ruling: T8 rollback restores weights + optimizer state from the last in-memory snapshot, skips the bad batch, and the step counter and sampler continue forward (no rewind) — rewinding would re-hit a fixed-step NaN forever; "skip ahead" matches the plan's intent — cost if wrong: a rolled-back run trains slightly fewer effective steps.
- R5 Ruling: T8 `stop_after_steps` saves the resume state at the stop step regardless of `checkpoint_every`, then returns status "interrupted" — cost if wrong: none.
- R6 Ruling: T20 extends the `tiny_data_root` fixture to also write `known_vocab.txt` and `false_facts.json` into the tiny corpus dir (and lazily build tiny judge + novelty index) — the quick gate needs them — cost if wrong: none.
- R7 Ruling: T9 unsupported operator nodes report the operator class name (`Unsupported: Pow`), other unsupported nodes the node class name — cost if wrong: none.
- R8 Ruling: T18 lexicons must contain at least the words the tests use (formal: therefore, however, additionally; casual: yeah, lol, gonna, hey) — cost if wrong: none.

## Task log
Task 1: minor (deferred): REPO_ROOT not asserted directly (add `(REPO_ROOT/"ml"/"pyproject.toml").is_file()`); pick_device precedence (prefer beats env) undocumented, unknown values silently treated as cuda; test file not `ruff format`-clean; novelty/judge "v1" hardcoded.
Task 1: env note: uv not on PATH — all briefs' `uv ...` commands must run as `python -m uv ...`.
Task 1: complete (commits c28acd2..747d806, review clean)
Task 2: minor (deferred): conftest imports tokenizer at module level (collection-wide failure on import error); no vocab_size==VOCAB_SIZE assert for real tok-v1 (Task 14/15 should assert); cp1252 console printing → Task 19 UTF-8 reconfigure; two overlapping tests. Note: fixture_texts = 150 paragraph strings (not per-file) — accepted.
Task 2: complete (commits 747d806..195e43f, review clean)
Task 3: minor (deferred): RMSNorm growth recipe undocumented (Task 4 must use weight*=sqrt(d/d'), eps*=d/d' incl norm_f; tolerance ~1e-6 achievable); Attention.n_head cached at construction (Task 4 must build a new Transformer, not swap modules); key_padding_mask dtype not validated (int mask fails inside SDPA); KV cache retains autograd graph under grad mode (Task 5 must use inference_mode); save_checkpoint not atomic (temp+rename); from_json bare KeyError on missing format; no min/max legal-shape boundary tests; brief test style not ruff-format clean.
Task 3: fix round 1 dispatched (resume a2583e4) — Important: save_checkpoint lacks meta.shape==model.shape guard
Task 3: fix round 1/5 (1 addressed, 0 open; commits 69c03d4..c1ac884)
Task 3: complete (commits 195e43f..c1ac884, review clean after 1 fix round)
Task 4: minor (deferred): insertion_layout(1,6) places fresh blocks before old block 0 (plan formula; function-preserving); no ctx_len-decrease test; no "trains after growth" committed test; eps.double() fails on MPS (n/a for CPU/CUDA); redundant shrink guard in insertion_layout.
Task 4: complete (commits c1ac884..17207f2, review clean)
Task 5: minor (deferred): complete() reserves no room for reply on ctx-length text (Task 20 must use short prefixes); batch_size re-validated per chunk; LanguageModel.ctx_len attr vs TorchLM property (protocol mismatch, no checker); prefill/scoring logits [B,W,V] memory (~512MB at B32/ctx1024) — size batch_size in Task 16; bf16 scoring on CUDA not bit-comparable to CPU; seeded sampling depends on batch composition (not per-prompt reproducible across different batches).
Task 5: fix round 1 dispatched (resume a0d4d0c) — Important: TorchLM mutates caller model mode/device
Task 5: fix round 1/5 (1 addressed, 0 open; commits 6e5546c..d7b8d44)
Task 5: complete (commits 17207f2..d7b8d44, review clean after 1 fix round)
Task 6: Ruling: DocPool.from_docs drops docs with no content tokens (empty or only special ids < 16) — an encoded blank notebook entry is [bos] and would otherwise train on repeated <|bos|>; Review Focus 3 — cost if wrong: none (no legitimate doc is specials-only).
Task 6: Ruling: purchases missing a dataset = fully owned (1.0) — matches TrainRunConfig "missing means 1.0" in the plan; the game layer (M3) will always send explicit purchase fractions — cost if wrong: an unbought dataset trains fully if the server omits it; M3 must send all fractions.
Task 6: minor (deferred): rng_state excludes current weights (Task 8 must re-derive phase weights from step on resume); set_rng_state merges unknown stats keys; weight-check duplicated sampler.py:93/:160, non-numeric weight → TypeError; write_corpus not atomic, no topic/noise_kind range check, no offsets monotonic check; missing corpus dir → FileNotFoundError; eligible_docs excludes heldout before dedup → Task 14 must never hold out a cluster's canonical doc (hold out whole clusters); unused `topics` param in brief test helper.
Task 6: fix round 1 dispatched (resume a20e1f3) — Important: zero-weight keys vs load_pools; blank notebook [bos] escapes empty check
Task 6: Ruling (amends earlier): from_docs drops docs made only of framing ids 0-4 (pad,bos,end,user,ai), not all ids<16 — brief tests use synthetic ids <16 as content; reserved ids 5-15 never arise from text (encode_special_tokens) — whitespace-only text must be stripped before encoding (Task 8 does) — cost if wrong: none.
Task 6: fix round 1/5 (2 addressed, 0 open; commits 3d827c5..080967d)
Task 6: complete (commits d7b8d44..080967d, review clean after 1 fix round)
Task 7: Ruling: validate() also enforces seed int>=0, run_id ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ (path-safe; future worker uses it in paths), replay>0 requires parent_dir — same external-JSON contract as the Important finding; path traversal is security-relevant — cost if wrong: a legit run_id format rejected (server generates ids, so none).
Task 7: minor (deferred): SpikeDetector flags every step after a persistent loss shift (by design; Task 8 caps at 3 spikes → stop); "heldouteval" type spelling is the TS contract.
Task 7: fix round 1 dispatched (resume a0dd18e) — Important: JSON contract types/ValueError; event_to_dict NaN/inf
Task 7: note for M2: TrainRunConfig rejects integer fields serialized as floats (32768.0); JS JSON.stringify emits integers without .0 so OK.
Task 7: fix round 1/5 (2 addressed, 0 open; commits 256ae5b..bbc8f99)
Task 7: minor (deferred): giant int weight (10**400) → OverflowError in _normalized (config.py:92); weights summing to inf pass validate and normalize to zeros; run_id regex accepts Windows-reserved names (con, trailing dot) — future worker should not use run_id directly as a Windows dir name, or extend regex.
Task 7: complete (commits 080967d..bbc8f99, review clean after 1 fix round)
Task 8: implemented a0ea8e5 (implementer ac96bcf, opus) — DONE_WITH_CONCERNS; 279 tests pass; GPU 104k tok/s cold / 190k warm (threshold 30k); resume bit-exact CPU+CUDA.
Task 8: Ruling: SpikeDetector re-baselines at the finishing-mix switch — a legitimate player choice (harder finishing data) must not trigger rollbacks — cost if wrong: a true spike exactly at the switch is missed once.
Task 8: Ruling: finishing mix starts at the LR-decay start; player probe prompts sampled as chat with top_p 0.95; coaching pair dropped if either side blank — all consistent with the plan — cost if wrong: none.
Task 8: Ruling: shrinking ctx_len must be ALLOWED (spec §4.3: "Context length may go up or down freely") — implementer made it raise GrowthError; this must be fixed in Task 8's fix round — cost if wrong: n/a (spec is binding).
Task 8: NEXT STEP ON RESUME: generate review package d7… no: `review-package PLAN bbc8f99 HEAD`, dispatch task reviewer (sonnet/opus), include the ctx_len ruling as a known finding for the fix round (resume implementer ac96bcf). Minor: resume-state swap not fully atomic; result.json uses private _json_safe from events.py.
SESSION PAUSED (usage limit) after Task 8 implementation, before its review.
SESSION RESUMED. Task 8 review (opus, afa1629): Critical ctx_len shrink rejected; Important MixtureError types through train_run; Important non-finite grad norm poisons snapshot.
Task 8: Ruling: pull Minor "resume swap fragile on Windows" into the fix loop — Task 17 judge build (~1h, ~36 saves) runs on this Windows machine and a crash there is costly — cost if wrong: small extra work.
Task 8: minor (deferred): fresh run into used out_dir leaves stale model/meta/result files; events re-emitted after crash-resume (consumers dedupe by step — M2 worker); final_loss may include rolled-back steps; resume-save cadence by steps not size (~1 GB/save at 85M params — M2 time-based cadence).
Task 8: fix round 1 dispatched (resume ac96bcf) — Critical ctx_len; Important MixtureError; Important grad-norm/snapshot; Minor-ruled Windows resume swap; + make _json_safe public.
Task 8: fix round 1/5 (4 addressed, 0 open; commits 1b4351c..d7123d2)
Task 8: minor (deferred): stale undeletable resume.old.* from an earlier same-config run could be picked on resume (very unlikely); resume state.pt ~2x size (includes snapshot).
Task 8: complete (commits bbc8f99..d7123d2, review clean after 1 fix round)
Task 9: implemented 76a9e7c (sonnet a7f93ad). Review (opus a644744): no sandbox escape found in ~90 probes; Critical: def-in-loop re-walks body → steps don't bound CPU (2.5 s per 537 chars); Important: big per-step render/print/contains work; no memory cap independent of steps (268 MB at 200k steps); RecursionLimit depends on host stack (non-deterministic scores).
Task 9: Ruling: pull security hardening Minors into the loop — len(range) int-cap leak, warnings.catch_warnings thread-unsafety (M2 worker is concurrent), 20k-char source cap, wall-clock safety net max_seconds=2.0 → new fixed error "TimeLimit", charge call_function boundary copy — spec §11 requires "step, memory and time limits"; this is security-sensitive code — cost if wrong: one extra error string downstream benchmark checkers must treat as failure (they treat any error as failure).
Task 9: minor (deferred): ValueError-like cases (int('abc'), max([])) report "TypeError" (brief's error list has no ValueError) — acceptable.
Task 9: fix round 1 dispatched (resume a7f93ad).
Task 9: fix round 1/5 (5 addressed, 0 open; commits 76a9e7c..21f3f17). Re-review (opus af98a37): 38k fuzzed programs, 0 escapes, 0 internal exceptions, slowest hostile ~24 ms at 10k steps.
Task 9: Ruling: element budget 200k and MAX_NESTING 250 stand — they make legit 650-char char-by-char strings / depth-45 recursion with nested control flow fail, but benchmark & training programs are tiny; Task 12's tests require every generated program to run without error, which enforces fit — cost if wrong: some unusual but valid model programs score as failures.
Task 9: minor (deferred): _extreme/_b_sum over huge ranges uncharged copy + no clock check at very large max_steps (fine at 10k/200k); docstring caller-headroom figure understated (deterministic up to ~295 caller frames); list `+=` charged as full copy; deadline starts before parse lock; module lock + fork; host-supplied name object __hash__ escape (trusted input).
Task 9: complete (commits d7123d2..21f3f17, review clean after 1 fix round)
Task 10: implemented 0e2f221 (sonnet ac6bdca): 1478 facts, 22 relations. Review (sonnet aad0530, read all facts): Important — wrong country_language "main language" rows; fuzzy relations make MC/consistency ambiguous (alligator meat/fish etc.); disjointness test vacuous + containment/reorder leaks.
Task 10: Ruling: pull bench-validity Minors into the loop (stereotyped/contested facts, polysemous subjects, subject-contains-answer leaks, accidentally-true false swaps, shared mutable KB) — bench-v1 is generated from this KB and frozen; scores are measurements (R5) and wrong answers would make players' AIs "wrong" because of us — cost if wrong: a little extra content work.
Task 10: Ruling: MC/exact/consistency items only from relations flagged mc_safe; fuzzy relations stay in training text — cost if wrong: knowledge bench covers fewer topics.
Task 10: downstream notes: Task 14 — plan_false_facts(300) takes all of day_after/planet_order/month_after/shape_sides; 499/840 subjects have one fact so prose docs restate one fact; Task 16 — exact matching strip punctuation, case-insensitive.
Task 10: fix round 1 dispatched (resume ac6bdca).
Task 10: fix round 1/5 (3 addressed + ruled items addressed, 1 open — baby_animal puppy/pup double-correct in safe relation, confusable test can't fail on safe relations; commits 0e2f221..132a3f9)
Task 10: minor (deferred): near-paraphrases bench vs train (Jaccard ≤0.88); element items limited to 23 (hard tail, element score reads low); bench has no nature/food/other topics; turtle "four" legs.
Task 10: fix round 2 dispatched (resume ac6bdca) — confusable groups for safe relations + exact synonyms.
Task 10: fix round 2/5 (1 addressed, 0 open; commits 132a3f9..ae6fbce)
Task 10: minor (deferred): animal_legs has exactly 3 wrong objects for lizard (no headroom); no load-time check that an accepted form which is also an object has a confusable group; vacuous assert at test_kb_facts.py:902; lizard exact item accepts only four.
Task 10: complete (commits 21f3f17..ae6fbce, review clean after 2 fix rounds)
Task 11: implemented a563e27 (sonnet a4cac37). Review (sonnet a5caec7): Important — syllogism 100% solvable by "No" cue; compare solvable without chaining (chance 1/3 not 1/4); premise-level train/bench leak at scale (compare 40%); double family 2 items; bad plurals; _fair_quota verbatim duplicate.
Task 11: Ruling: supersede earlier guidance "extra 4th name not in premise" — put it in an irrelevant third premise so it is not a free elimination — cost if wrong: none.
Task 11: Ruling: widen double family AND add a minimum-n (≈10) reporting rule to Task 16 tag_breakdown (and later discoveries) — carry into Task 16 dispatch — cost if wrong: small tags hidden from players.
Task 11: Ruling: pull measurement Minors M1-M4 (uniform answer rank, sqrt family weighting, shortcut-audit test, count balance) into the loop — bench freezes; R5 — cost if wrong: extra work.
Task 11: fix round 1 dispatched (resume a4cac37).
Task 11: round-1 commit a7e762f. Ruling: syllogism decoy shares the queried class word with the opposite quantifier (forces chaining; adds "premise mentioning queried word" to audit) — cost if wrong: none. Ruling: accept world-key partition (vs listing variant prompts), compare odd-name sentence also naming a chain name, 4-7 term patterns.
Task 11: fix round 1/5 (10 addressed, 2 open — R1 count most-frequent-word shortcut 0.82; R2 syllogism one-off-word decoy detection 1.00; commits a563e27..a80c64f)
Task 11: Ruling: R2 (reviewer rated Minor) must be fixed — every nonce word appears ≥2× so the decoy can't be spotted by a one-off token; chaining must be required for the score to measure reasoning — cost if wrong: slightly longer prompts.
Task 11: Ruling: cycle "copy term k back" is the skill, not a shortcut.
Task 11: minor (deferred): word_problem "option within 5 of all others" 0.41 (just above chance+0.15); 31/50 bench syllogisms share core premises with train blocks differing in decoy/name (no answer advantage).
Task 11: fix round 2 dispatched (resume a4cac37).
Handoff: see docs/handoff/README.md (written 2026-10-07). NEXT STEP ON RESUME: Task 11 fix round 2 (R1 count shortcut, R2 syllogism one-off decoy word) — fresh implementer; old one hit rate limit with no changes.
Ruling: auto-push post-commit hook + tools/sync-handoff.sh after every task/fix round — owner asked that latest progress always be saved to GitHub for the cloud handoff — cost if wrong: none (branch only, never main, never force).
Task 11: fix round 2 implemented 4db5e74 (fresh opus implementer aac36f6). R1 count most-common 0.82→0.26; R2 once-word 1.00→0.50.
Task 11: Ruling: accept residual "two linked All premises → yes" (0.75 on 2-step syllogisms, ~0.63 family-wide) — detecting a shared middle term between universal premises is itself partial term-linking reasoning, not a surface token cue; removing it needs a 4th premise with diminishing returns — cost if wrong: 2-step syllogisms over-reward structural pattern matching somewhat.
Task 11: Ruling: accept "class of once-mentioned person is the decoy" (1.00) — applying it takes as many steps as the real chain — cost if wrong: negligible.
