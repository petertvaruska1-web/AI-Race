export const meta = {
  name: 'm1-final-review',
  description: 'Final whole-branch review of Milestone 1: subsystem x lens finders, specialists, completeness critic, dedup, adversarial verification, synthesis',
  phases: [
    { title: 'Find', detail: '5 subsystems x 3 lenses + triage, tests-health, spec-coverage' },
    { title: 'Critic', detail: 'completeness critic proposes extra focused finders' },
    { title: 'Find more', detail: 'extra finders from the critic' },
    { title: 'Dedup', detail: 'merge semantic duplicates across all findings' },
    { title: 'Verify', detail: '3 perspective-diverse skeptics per serious finding, 1 per minor' },
    { title: 'Synthesize', detail: 'final review report' },
  ],
}

// Portable copy of the final-review workflow (docs/handoff/final-review/README.md explains how to run it).
// args (all optional): { root, workspace, scratch, cacheFile, cachedLabels }
//   cacheFile + cachedLabels: replay results already saved in agent-results.json (exported by
//   export-results.py) instead of re-running those agents. cachedLabels lists the labels to replay.
const A = (typeof args === 'object' && args) ? args : {}
const ROOT = A.root || '/home/user/AI-Race'
const W = A.workspace || (ROOT + '/.superpowers/sdd/2026-10-06-m1-it-learns')
const SCRATCH = A.scratch || '/tmp/final-review-scratch'
const CACHE_FILE = A.cacheFile || (ROOT + '/docs/handoff/final-review/agent-results.json')
const CACHED = new Set(Array.isArray(A.cachedLabels) ? A.cachedLabels : [])

const CONTEXT = `You are part of the FINAL whole-branch review of Milestone 1 ("It learns") of AI Race, a multiplayer strategy game where each player trains a real miniature language model. The branch (claude/upbeat-franklin-k1192h, from main @ c28acd2) contains the whole Python ML core in ${ROOT}/ml (packages airace_ml and airace_content), built task by task (Tasks 1-20 of the plan) and each task already reviewed. This review looks for what task-scoped reviews miss: cross-task defects, integration gaps, full-scale and Windows behaviour, and anything that would make the owner's upcoming real runs fail or silently produce wrong results.

Authoritative inputs (read what you need):
- Spec (binding authority): ${ROOT}/docs/superpowers/specs/2026-10-06-ai-race-architecture-design.md (§1 R1-R8, §4, §11, §13 apply to M1)
- Plan: ${ROOT}/docs/superpowers/plans/2026-10-06-m1-it-learns.md (read only the tasks relevant to your scope)
- Execution ledger with EVERY controller ruling (a ruling is a deliberate decision; flag it only if it is clearly wrong, and say why): ${W}/progress.md
- ADR: ${ROOT}/docs/decisions/0001-bench-v1-measurement-rules.md
- Global constraints and Review Focus 1-5: ${W}/global-constraints.md
- Deferred minor findings from task reviews: ${W}/deferred-minors.md
- Project rules: ${ROOT}/CLAUDE.md

Deployment facts that matter:
- The owner is not technical. They will run everything on Windows 11 with an RTX 3050 (8 GB, CUDA) and a Ryzen 5 5500, using "python -m uv run". Next they will run: the real corpus build (airace-content build --scale full, Hugging Face streaming, ~137M tokens), the novelty index, the reference judge (~1 h GPU), and the feasibility gate (~1 h GPU). Defects that only appear at full scale, on CUDA, or on Windows matter a lot.
- This cloud review machine is Linux, CPU only, with no network to Hugging Face. Test command: cd ${ROOT}/ml && uv run --no-sync pytest <file>::<test> -v (never "uv sync"). The default suite takes ~2 min.

Rules: read-only — never modify the repo, its index, HEAD or branches. Put any probe scripts or files only under ${SCRATCH}/. Never spawn subagents. Cite file:line for everything. Prefer concrete, verified failure scenarios over speculation; mark confidence honestly. Do not re-report something the ledger shows as deliberately ruled on unless you can show the ruling is wrong. Steps that require the GPU or network (Task 15, Task 17's real judge training, Task 20 steps 6-9) are out of scope as work items, but code defects that will break them are in scope.`

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          severity: { type: 'string', enum: ['Critical', 'Important', 'Minor'] },
          file: { type: 'string' },
          line: { type: 'string' },
          scenario: { type: 'string', description: 'concrete inputs/state -> wrong result, crash, or data loss' },
          evidence: { type: 'string', description: 'what you checked: code read, probe run, test run' },
          fix: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['title', 'severity', 'file', 'line', 'scenario', 'evidence', 'fix', 'confidence'],
      },
    },
    coverage_note: { type: 'string', description: 'what you read and what you could not cover' },
  },
  required: ['findings', 'coverage_note'],
}

const SUBSYSTEMS = [
  { key: 'core', files: 'ml/src/airace_ml/{tokenizer.py,paths.py,device.py,model/,train/,data/,infer/} and tests test_tokenizer.py, test_model.py, test_growth.py, test_infer.py, test_corpus.py, test_sampler.py, test_train_config.py, test_trainer.py, test_paths_device.py, conftest.py (plan Tasks 1-8)' },
  { key: 'skills', files: 'ml/src/airace_ml/{minipy/,skills/} including skills/kb_data/*.json, and tests test_minipy.py, test_kb_facts.py, test_reasoning_patterns.py, test_code_instr_grammar.py (plan Tasks 9-12). These generate both training data and the frozen benchmark bench-v1.' },
  { key: 'content', files: 'ml/src/airace_content/ (textproc, noise, dedup, topics, sources, assemble, manifest, build, cli) and tests/content/*, tests/fixtures/sources/* (plan Tasks 13-14; Task 15 will run it for real at full scale on Windows)' },
  { key: 'evals', files: 'ml/src/airace_ml/{evals/,personality/} and tests test_bench.py, test_creativity.py, test_personality.py, fakes.py (plan Tasks 16-18; ADR 0001)' },
  { key: 'gate_cli', files: 'ml/src/airace_ml/{experiments/,cli.py}, CLAUDE.md Commands, .gitignore, and tests test_gate.py, test_cli.py, conftest.py (plan Tasks 19-20)' },
]

const LENSES = [
  { key: 'correctness', text: 'LENS = correctness and measurement validity. Does the code do what the spec, the plan tasks and the ledger rulings require? Look for logic errors, wrong formulas, off-by-one errors, misused interfaces, wrong units. For anything that produces a score, tag, trait or gate verdict: can an untrained, degenerate, constant, copying, looping or question-blind model game it? Is 0 really chance? Can a held-out/benchmark item leak into training? Is every R1 (no scripted output) and R5 (measurements, not menus) property intact?' },
  { key: 'robustness', text: 'LENS = robustness, safety and determinism. Crashes, NaN/inf, empty and degenerate inputs, Review Focus 1-5; determinism (same seed -> same result; no salted hash(), no wall-clock randomness, stable iteration order); resource use at FULL scale (corpora ~137M tokens, models up to 85M params, 8 GB VRAM, 32 GB RAM); crash-safety and atomicity of file writes and resumes; text encoding; security (MiniPy sandbox escape, path traversal via run_id/config/CLI paths, unsafe deserialization such as torch.load without weights_only); and WINDOWS behaviour specifically (os.replace/rename/unlink of open or memory-mapped files, np.memmap handles kept open, file locking, long paths, cp1252 console, CRLF, case-insensitive paths, multiprocessing spawn).' },
  { key: 'integration', text: 'LENS = cross-task integration and contracts. Callers vs callees across task boundaries (signatures, return types, units, token conventions such as bos and leading spaces, special ids, version constants); shared definitions that must agree but are duplicated (word tokenization, normalization, thresholds); dead code; error types that callers do not handle; and above all what the owner\'s next real runs will hit (full corpus build -> novelty index -> judge build -> feasibility gate): anything that would fail, take far too long, or silently produce wrong results only at full scale or on CUDA.' },
]

const SPECIALISTS = [
  { key: 'triage-deferred', prompt: `TASK = triage every deferred minor and parked finding in ${W}/deferred-minors.md (and any "minor (deferred)" or "parked" line in the ledger). For each one, check the current code: is it still present, and does it need fixing before this branch is merged and the owner runs the real build/judge/gate? Report as findings ONLY the ones that should be fixed before merge (severity Important if they can cause wrong results, data loss or a failed real run; Minor otherwise), each with a concrete scenario. Put a one-line verdict for every deferred item (still present? fix before merge / acceptable / already fixed) in coverage_note.` },
  { key: 'tests-health', prompt: `TASK = test-suite health across ml/tests. Find tests that assert nothing or assert something trivially true; tests that are flaky (timing thresholds, unseeded randomness, order dependence, reliance on the 512-vocab fixture tokenizer for properties that must hold for tok-v1); critical paths with no test at all (name them concretely); default-suite tests that are too slow (the default suite must stay well under 3 minutes on the owner's 6-core Ryzen; run "uv run --no-sync pytest --durations=25 -q" once and report); skipped tests and why; tests that would fail on Windows (path separators, file locking, line endings, encoding).` },
  { key: 'spec-coverage', prompt: `TASK = spec coverage. Walk spec §1 (R1-R8 as they apply to M1), §4.1-§4.12, §11 (security as it applies to ml/) and §13 (ML tests), plus the plan's Global Constraints and Review Focus 1-5. For each requirement, find the code that implements it and the test that pins it. Report as findings every requirement that is missing, only partly implemented, implemented differently from the spec without a recorded ruling/ADR, or untested. Also check that docs (CLAUDE.md Commands, docs/decisions/0001, spec amendments) match the code.` },
]

const VERDICT = {
  type: 'object',
  properties: { refuted: { type: 'boolean' }, severity_if_real: { type: 'string', enum: ['Critical', 'Important', 'Minor', 'None'] }, reasoning: { type: 'string' } },
  required: ['refuted', 'severity_if_real', 'reasoning'],
}

const SKEPTICS = [
  { key: 'reproduce', text: 'Your lens: REPRODUCE. Try to demonstrate the claimed failure concretely (read the exact code path; run a focused test or a scratchpad probe if feasible). Refute if the scenario cannot actually happen as described.' },
  { key: 'intent', text: 'Your lens: INTENT. Check the spec, the plan task text, the ledger rulings and ADR 0001. Refute if the behaviour is deliberately required or accepted there (and the ruling is not clearly wrong), or if the item is genuinely out of this branch\'s scope.' },
  { key: 'impact', text: 'Your lens: IMPACT. Judge what actually happens to the owner, their real build/judge/gate runs, the frozen bench-v1, or future players. Refute if the impact is negligible; otherwise state the severity it deserves.' },
]

// Replays a saved result (cheap loader agent) when the label is cached, else runs the real agent.
function runAgent(prompt, opts) {
  if (opts && CACHED.has(opts.label)) {
    return agent(`Run this exact command with the Bash tool and return its JSON output through the StructuredOutput tool, copying every field and every string verbatim (do not summarize, shorten, re-check or change anything):\npython3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1], encoding='utf-8'))[sys.argv[2]], ensure_ascii=False))" "${CACHE_FILE}" "${opts.label}"`,
      { label: `load:${opts.label}`, phase: opts.phase, schema: opts.schema, effort: 'low' })
  }
  return agent(prompt, opts)
}

function findPrompt(extra) {
  return `${CONTEXT}\n\n${extra}\n\nReturn every real defect you find with a concrete scenario. Be thorough; it is fine to return zero findings for an area that is genuinely clean, but say what you checked in coverage_note.`
}

// ---------- Find ----------
phase('Find')
const finderSpecs = []
for (const s of SUBSYSTEMS) for (const l of LENSES) finderSpecs.push({ id: `${s.key}/${l.key}`, prompt: findPrompt(`SCOPE = ${s.files}.\n${l.text}`) })
for (const sp of SPECIALISTS) finderSpecs.push({ id: sp.key, prompt: findPrompt(sp.prompt) })

const firstRound = await parallel(finderSpecs.map(f => () =>
  runAgent(f.prompt, { label: `find:${f.id}`, phase: 'Find', schema: FINDINGS }).then(r => r ? { id: f.id, ...r } : null)))
const round1 = firstRound.filter(Boolean)
log(`round 1: ${round1.length}/${finderSpecs.length} finders returned, ${round1.reduce((n, r) => n + r.findings.length, 0)} raw findings`)
if (round1.length < finderSpecs.length) {
  const missing = finderSpecs.filter((f, i) => !firstRound[i]).map(f => f.id)
  log(`INCOMPLETE: ${missing.length} finders failed (${missing.join(', ')}); stopping so a resume re-runs only them`)
  return { incomplete: 'find', missing }
}

// ---------- Critic ----------
phase('Critic')
const summary = round1.map(r => `## ${r.id}\nCoverage: ${r.coverage_note}\nFindings: ${r.findings.map(f => `[${f.severity}] ${f.file}:${f.line} ${f.title}`).join('; ') || 'none'}`).join('\n\n')
const critic = await runAgent(`${CONTEXT}\n\nYou are the COMPLETENESS CRITIC. Below is what the first round of reviewers covered and found. List what is missing: files or modules nobody read closely, failure modes nobody checked (e.g. a specific Windows behaviour, a full-scale memory path, a resume/crash window, a cross-task contract, a scoring rule that could be gamed), and claims that look unverified. Propose at most 6 extra focused finder tasks, each narrow and concrete (name files and the exact question). Propose none if coverage is genuinely complete.\n\n${summary}`,
  { label: 'critic', phase: 'Critic', schema: { type: 'object', properties: { gaps: { type: 'string' }, extra_tasks: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, task: { type: 'string' } }, required: ['id', 'task'] } } }, required: ['gaps', 'extra_tasks'] } })

if (!critic) {
  log('INCOMPLETE: the completeness critic failed; stopping so a resume re-runs it')
  return { incomplete: 'critic' }
}

// ---------- Find more ----------
phase('Find more')
const extraTasks = critic ? critic.extra_tasks.slice(0, 6) : []
if (critic && critic.extra_tasks.length > 6) log(`critic proposed ${critic.extra_tasks.length} tasks; running the first 6`)
const secondRound = await parallel(extraTasks.map(t => () =>
  runAgent(findPrompt(`TASK (from the completeness critic) = ${t.task}\n\nLimits for this task: use at most 40 Firecrawl requests in total (they are billed to the owner's account; prefer few, larger page sizes); keep every probe script and downloaded sample under ${SCRATCH}/ and never in the repo; finish with a structured result even if a probe is cut short.`), { label: `find+:${t.id}`, phase: 'Find more', schema: FINDINGS }).then(r => r ? { id: `extra/${t.id}`, ...r } : null)))
if (secondRound.some(r => !r)) {
  log('INCOMPLETE: an extra finder failed; stopping so a resume re-runs it')
  return { incomplete: 'find-more' }
}
const allRounds = round1.concat(secondRound.filter(Boolean))
const raw = []
for (const r of allRounds) r.findings.forEach((f, i) => raw.push({ ...f, source: r.id, rid: `${r.id}#${i}` }))
log(`total raw findings: ${raw.length}`)

// ---------- Dedup ----------
phase('Dedup')
const deduped = raw.length === 0 ? { merged: [] } : await runAgent(`${CONTEXT}\n\nYou are the DEDUPLICATOR. Merge findings that describe the same underlying defect (same root cause, even if reported from different lenses or files). Keep distinct defects separate. For each merged finding give the strongest severity among its sources, the clearest title/scenario/fix, and list the source rids. Do not drop any distinct defect.\n\nFINDINGS (JSON):\n${JSON.stringify(raw)}`,
  { label: 'dedup', phase: 'Dedup', schema: { type: 'object', properties: { merged: { type: 'array', items: { type: 'object', properties: {
    title: { type: 'string' }, severity: { type: 'string', enum: ['Critical', 'Important', 'Minor'] }, file: { type: 'string' }, line: { type: 'string' },
    scenario: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' }, sources: { type: 'array', items: { type: 'string' } } },
    required: ['title', 'severity', 'file', 'line', 'scenario', 'evidence', 'fix', 'sources'] } } }, required: ['merged'] } })
if (!deduped) {
  log('INCOMPLETE: dedup failed; stopping so a resume re-runs it')
  return { incomplete: 'dedup' }
}
const merged = deduped.merged
log(`after dedup: ${merged.length} findings (${merged.filter(f => f.severity !== 'Minor').length} Critical/Important)`)

// ---------- Verify ----------
phase('Verify')
async function verify(f, idx) {
  const serious = f.severity !== 'Minor'
  const lenses = serious ? SKEPTICS : [SKEPTICS[0]]
  const votes = (await parallel(lenses.map(s => () => runAgent(
    `${CONTEXT}\n\nYou are an adversarial SKEPTIC. A reviewer reports this finding; try to REFUTE it. ${s.text}\nSet refuted=false only if, under your lens, the finding stands. If it stands, give the severity it really deserves.\n\nFINDING:\n${JSON.stringify(f)}`,
    { label: `verify:${idx}:${s.key}`, phase: 'Verify', schema: VERDICT })))).filter(Boolean)
  if (votes.length < lenses.length) return { ...f, id: `F${idx + 1}`, failed: true, votes }
  const upheld = votes.filter(v => !v.refuted)
  const confirmed = serious ? upheld.length >= 2 : upheld.length >= 1
  const sevs = upheld.map(v => v.severity_if_real).filter(s => s !== 'None')
  const rank = { Critical: 3, Important: 2, Minor: 1 }
  sevs.sort((a, b) => rank[b] - rank[a])
  const agreed = sevs.length ? sevs[Math.floor(sevs.length / 2)] : f.severity
  return { ...f, id: `F${idx + 1}`, confirmed, final_severity: confirmed ? agreed : 'None', votes }
}
const verified = (await parallel(merged.map((f, i) => () => verify(f, i)))).filter(Boolean)
if (verified.length < merged.length || verified.some(v => v.failed)) {
  log('INCOMPLETE: some verification agents failed; stopping so a resume re-runs them')
  return { incomplete: 'verify', failed: verified.filter(v => v.failed).map(v => v.id) }
}
const confirmed = verified.filter(v => v.confirmed)
log(`confirmed: ${confirmed.length}/${verified.length} (${confirmed.filter(v => v.final_severity === 'Critical').length} Critical, ${confirmed.filter(v => v.final_severity === 'Important').length} Important, ${confirmed.filter(v => v.final_severity === 'Minor').length} Minor)`)

// ---------- Synthesize ----------
phase('Synthesize')
const triage = round1.find(r => r.id === 'triage-deferred')
const report = await runAgent(`${CONTEXT}\n\nYou are the SYNTHESIZER of the final review. Write the final review report as Markdown to ${W}/final-review.md (this is the one file you may write; it is in the git-ignored workspace). Structure:\n1. Verdict: is the branch ready for the owner's real runs once the confirmed Critical/Important findings are fixed? One paragraph.\n2. Confirmed Critical and Important findings, most severe first: id, title, file:line, scenario, the exact fix to make, and the test that must pin it. These will go to ONE fix implementer, so make each fix instruction precise and self-contained.\n3. Confirmed Minor findings worth fixing in the same pass (cheap, low-risk), with fixes.\n4. Remaining Minors and refuted findings: one line each with why (so nothing is silently dropped).\n5. Deferred-minor triage table (from the triage reviewer's coverage note).\n6. What the review could not verify here (GPU/Windows/network-only behaviour) and what the local session must check.\nThen return the same verdict paragraph and the list of finding ids to fix.\n\nVERIFIED FINDINGS (JSON):\n${JSON.stringify(verified)}\n\nDEFERRED TRIAGE NOTE:\n${triage ? triage.coverage_note : 'n/a'}\n\nCRITIC GAPS:\n${critic ? critic.gaps : 'n/a'}`,
  { label: 'synthesize', phase: 'Synthesize', schema: { type: 'object', properties: { verdict: { type: 'string' }, fix_ids: { type: 'array', items: { type: 'string' } }, report_path: { type: 'string' } }, required: ['verdict', 'fix_ids', 'report_path'] } })

return {
  report,
  counts: { finders: allRounds.length, raw: raw.length, merged: merged.length, confirmed: confirmed.length },
  confirmed: confirmed.map(c => ({ id: c.id, severity: c.final_severity, title: c.title, file: c.file, line: c.line })),
}
