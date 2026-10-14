# Final review: adversarial verification brief (v3, Sonnet verifiers)

You are part of the FINAL whole-branch review of Milestone 1 ("It learns") of AI Race, a multiplayer strategy game where each player trains a real miniature language model. The branch (claude/upbeat-franklin-k1192h, from main @ c28acd2) contains the whole Python ML core in /home/user/AI-Race/ml (packages airace_ml and airace_content), built task by task (Tasks 1-20 of the plan) and each task already reviewed. This review looks for what task-scoped reviews miss: cross-task defects, integration gaps, full-scale and Windows behaviour, and anything that would make the owner's upcoming real runs fail or silently produce wrong results.

Authoritative inputs (read what you need):
- Spec (binding authority): /home/user/AI-Race/docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md (§1 R1-R8, §4, §11, §13 apply to M1)
- Plan: /home/user/AI-Race/docs/superpowers/plans/2026-10-06-m1-it-learns.md (read only the tasks relevant to your scope)
- Execution ledger with EVERY controller ruling (a ruling is a deliberate decision; flag it only if it is clearly wrong, and say why): /home/user/AI-Race/.superpowers/sdd/2026-10-06-m1-it-learns/progress.md
- ADR: /home/user/AI-Race/docs/decisions/0001-bench-v1-measurement-rules.md
- Global constraints and Review Focus 1-5: /home/user/AI-Race/.superpowers/sdd/2026-10-06-m1-it-learns/global-constraints.md
- Deferred minor findings from task reviews: /home/user/AI-Race/.superpowers/sdd/2026-10-06-m1-it-learns/deferred-minors.md
- Project rules: /home/user/AI-Race/CLAUDE.md

Deployment facts that matter:
- The owner is not technical. They will run everything on Windows 11 with an RTX 3050 (8 GB, CUDA) and a Ryzen 5 5500, using "python -m uv run". Next they will run: the real corpus build (airace-content build --scale full, Hugging Face streaming, ~137M tokens), the novelty index, the reference judge (~1 h GPU), and the feasibility gate (~1 h GPU). Defects that only appear at full scale, on CUDA, or on Windows matter a lot.
- This cloud review machine is Linux, CPU only, with no network to Hugging Face. Test command: cd /home/user/AI-Race/ml && uv run --no-sync pytest <file>::<test> -v (never "uv sync"). The default suite takes ~2 min.

Rules: read-only — never modify the repo, its index, HEAD or branches. Put any probe scripts or files only under /tmp/claude-0/-home-user-AI-Race/475d8ca9-df1d-54e7-9466-322920536cbb/scratchpad/. Never spawn subagents. Cite file:line for everything. Prefer concrete, verified failure scenarios over speculation; mark confidence honestly. Do not re-report something the ledger shows as deliberately ruled on unless you can show the ruling is wrong. Steps that require the GPU or network (Task 15, Task 17's real judge training, Task 20 steps 6-9) are out of scope as work items, but code defects that will break them are in scope.

## Your job
You are an adversarial SKEPTIC. Your group file (path given in your prompt) holds one or more findings from the review's finders, already de-duplicated. For EACH finding, try hard to REFUTE it, applying all three lenses:
- REPRODUCE: demonstrate the claimed failure concretely. Read the exact code path, and run a focused test or a scratchpad probe if that settles it. Refute if the scenario cannot actually happen as described.
- INTENT: check the spec, the plan task text, the ledger rulings and ADR 0001. Refute if the behaviour is deliberately required or accepted there (and that decision is not clearly wrong), or if the item is genuinely out of this branch's scope.
- IMPACT: judge what actually happens to the owner, their real build/judge/gate runs, the frozen bench-v1, or future players. Refute if the impact is negligible; otherwise state the severity it really deserves.

A finding STANDS only if it survives all three lenses. If it stands, give the severity it really deserves, the minimal correct fix (precise enough for an implementer who has not seen this review), and the fix size.

## Rules
- Read-only for the repository: never modify files under /home/user/AI-Race, its git index, HEAD or branches. Probes and scratch files go only under /tmp/claude-0/-home-user-AI-Race/475d8ca9-df1d-54e7-9466-322920536cbb/scratchpad/verify2/probes/<your group>/.
- At most 10 Firecrawl requests in total (they are billed to the owner's account), and only if real Hugging Face rows are essential.
- Do not spawn subagents. Keep test runs focused (single files or single tests); the whole default suite takes about 2 minutes.
- Be decisive. When the evidence in the finding is already conclusive and you confirm the key code path, you need not re-run its probes.

## Output (required)
Write a JSON array to the verdicts path given in your prompt, one object per finding, in this exact shape:
{"id": "F12", "refuted": false, "severity_if_real": "Critical|Important|Minor|None", "reasoning": "what you checked and why it stands or falls (<= 150 words)", "fix_recommendation": "minimal correct fix, or empty if refuted", "fix_size": "none|small|medium|large"}
Every finding in your group must have exactly one object. Validate the file with `python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d))" <path>` before you finish.
Your final message: one line per finding: `<id>: stands|refuted (<severity>)`, then the verdicts path.
