export const meta = {
  name: 'm1-fix-wave',
  description: 'Final-review fix wave: six sequential Sonnet implementers (groups A-F), each followed by an adversarial Sonnet re-review and at most one fix round with a narrow re-check',
  phases: [
    { title: 'Implement', detail: 'one Sonnet implementer per group: reproduce with a failing test, fix, commit per finding' },
    { title: 'Review', detail: 'adversarial Sonnet review of the group commits against the binding rulings' },
    { title: 'Fix', detail: 'one fix round for blocking review issues' },
    { title: 'Recheck', detail: 'narrow re-check of the fix round' },
  ],
}

const ROOT = '/home/user/AI-Race'
const SP = '/tmp/claude-0/-home-user-AI-Race/475d8ca9-df1d-54e7-9466-322920536cbb/scratchpad'
const WS = '.superpowers/sdd/2026-10-06-m1-it-learns'
const GROUPS = (typeof args === 'object' && args && Array.isArray(args.groups)) ? args.groups : ['A', 'B', 'C', 'D', 'E', 'F']

const IMPL = { type: 'object', properties: {
  status: { type: 'string', enum: ['DONE', 'DONE_WITH_CONCERNS', 'BLOCKED'] },
  base_sha: { type: 'string', description: "the commit just before this round's first commit" },
  head_sha: { type: 'string' },
  findings: { type: 'array', items: { type: 'object', properties: {
    id: { type: 'string' }, status: { type: 'string', enum: ['FIXED', 'NOT_REPRODUCED', 'PARTIAL', 'NOT_STARTED'] },
    commits: { type: 'array', items: { type: 'string' } }, note: { type: 'string' } },
    required: ['id', 'status', 'commits', 'note'] } },
  suite: { type: 'string', description: 'default suite: passed / skipped / seconds' },
  ruff_clean: { type: 'boolean' },
  slow_suite: { type: 'string' },
  concerns: { type: 'string' } },
  required: ['status', 'base_sha', 'head_sha', 'findings', 'suite', 'ruff_clean', 'slow_suite', 'concerns'] }

const REVIEW = { type: 'object', properties: {
  issues: { type: 'array', items: { type: 'object', properties: {
    finding: { type: 'string', description: "Fxx or 'general'" }, severity: { type: 'string', enum: ['blocking', 'minor'] },
    where: { type: 'string' }, problem: { type: 'string' }, required_fix: { type: 'string' } },
    required: ['finding', 'severity', 'where', 'problem', 'required_fix'] } },
  suite: { type: 'string' },
  summary: { type: 'string' } },
  required: ['issues', 'suite', 'summary'] }

const docs = g => `${ROOT}/docs/handoff/fix-wave/fix-wave-rules.md and then ${ROOT}/docs/handoff/fix-wave/group-${g}.md`

function implPrompt(g) {
  return `You are the fix-wave implementer for group ${g} of the AI Race repository (${ROOT}). Read ${docs(g)}, and do exactly what they say. When you finish, return the structured result: status; base_sha (the commit just before this group's first fix(${g}) commit); head_sha; one entry per finding in the brief (FIXED | NOT_REPRODUCED | PARTIAL | NOT_STARTED, with commit SHAs and a one-line note); the default-suite result; whether ruff is clean; the slow-suite result; concerns.`
}
function reviewPrompt(g, impl) {
  return `You are an adversarial reviewer for fix-wave group ${g} of the AI Race repository (${ROOT}). Read ${docs(g)}: the group brief holds the controller's binding rulings. The implementer reports: ${JSON.stringify(impl)}. Its full report is ${ROOT}/${WS}/task-fixwave-${g}-report.md. Review the commits ${impl.base_sha}..${impl.head_sha} (git log -p). For every finding in the brief decide: is the ruling implemented fully and correctly; does a test genuinely pin it, i.e. would that test fail on the pre-fix code? (You may check that by creating a temporary git worktree under ${SP}/fixwave/review-${g}/ at ${impl.base_sha}, copying in only the new or changed test files, running them, and then removing the worktree with 'git worktree remove --force'.) For NOT_REPRODUCED findings: is the evidence convincing? Also check: no existing assertion was weakened; nothing outside the brief changed; determinism (no wall-clock time or unseeded randomness in measured or generated output); Windows safety (renames, open files, encodings); rule R1 (model output is never scripted or rewritten); explicit utf-8 file I/O. Run the full default suite and ruff yourself (cd ${ROOT}/ml && uv run --no-sync pytest -q && uv run --no-sync ruff check .). Never modify the repository's working tree, index, branches or commits (the temporary worktree aside). Mark an issue 'blocking' if anything is wrong, missing, untested or failing; 'minor' only for cosmetic matters. Return the structured result.`
}
function fixPrompt(g, issues) {
  return `You are the review-fix implementer for fix-wave group ${g} of the AI Race repository (${ROOT}). Read ${docs(g)}. An adversarial review of the group's commits reported these issues: ${JSON.stringify(issues)}. Fix every blocking issue, plus any minor one that is cheap and clearly right, test first as the rules say. Commit with subjects starting 'fix(${g}): review — '. Append a 'Review fix round' section to ${ROOT}/${WS}/task-fixwave-${g}-report.md. Re-run the full default suite, ruff and the slow suite. Return the structured result: base_sha is the commit before your first review-fix commit; findings has one entry per issue you handled, with id set to that issue's finding.`
}
function recheckPrompt(g, issues, fix) {
  return `You are re-checking the review-fix round of fix-wave group ${g} in the AI Race repository (${ROOT}). The review issues were: ${JSON.stringify(issues)}. The fix round reports: ${JSON.stringify(fix)}. Check commits ${fix.base_sha}..${fix.head_sha} narrowly: is each blocking issue resolved, with a test that pins it? Did the fix break or weaken anything else? Run the full default suite and ruff (cd ${ROOT}/ml && uv run --no-sync pytest -q && uv run --no-sync ruff check .). Read-only: never modify the repository. Return the structured result listing ONLY issues that remain or were newly introduced.`
}

const done = []
for (const g of GROUPS) {
  phase('Implement')
  const impl = await agent(implPrompt(g), { label: `impl:${g}`, phase: 'Implement', schema: IMPL, model: 'sonnet' })
  if (!impl) { log(`INCOMPLETE: implementer ${g} did not finish; resume re-runs it`); return { incomplete: 'implement', group: g, done } }
  if (impl.status === 'BLOCKED') { log(`BLOCKED: group ${g}`); return { blocked: g, impl, done } }
  phase('Review')
  const rev = await agent(reviewPrompt(g, impl), { label: `review:${g}`, phase: 'Review', schema: REVIEW, model: 'sonnet' })
  if (!rev) { log(`INCOMPLETE: review ${g}`); return { incomplete: 'review', group: g, done } }
  const blocking = rev.issues.filter(i => i.severity === 'blocking')
  let fix = null, recheck = null
  if (blocking.length) {
    phase('Fix')
    fix = await agent(fixPrompt(g, rev.issues), { label: `fix:${g}`, phase: 'Fix', schema: IMPL, model: 'sonnet' })
    if (!fix) { log(`INCOMPLETE: fix round ${g}`); return { incomplete: 'fix', group: g, done } }
    phase('Recheck')
    recheck = await agent(recheckPrompt(g, rev.issues, fix), { label: `recheck:${g}`, phase: 'Recheck', schema: REVIEW, model: 'sonnet' })
    if (!recheck) { log(`INCOMPLETE: recheck ${g}`); return { incomplete: 'recheck', group: g, done } }
  }
  done.push({ group: g, impl_status: impl.status, findings: impl.findings.map(f => ({ id: f.id, status: f.status })),
    review_blocking: blocking.length, review_minor: rev.issues.filter(i => i.severity !== 'blocking'),
    residual: recheck ? recheck.issues : [], suite: (fix || impl).suite, concerns: impl.concerns })
  log(`group ${g}: ${impl.findings.filter(f => f.status === 'FIXED').length}/${impl.findings.length} fixed; ${blocking.length} blocking review issues; ${recheck ? recheck.issues.length : 0} residual`)
}
return { done }
