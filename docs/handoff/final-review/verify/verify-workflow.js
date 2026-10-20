export const meta = {
  name: 'm1-final-review-verify',
  description: 'Adversarially verify the 55 de-duplicated final-review findings with Sonnet skeptics, one per subsystem group',
  phases: [{ title: 'Verify', detail: '13 Sonnet skeptics: 6 serious-finding groups first, then 7 minor groups' }],
}

const SP = '/tmp/claude-0/-home-user-AI-Race/475d8ca9-df1d-54e7-9466-322920536cbb/scratchpad'
const IDS = {
  's-bench': ['F1', 'F5'], 's-evals': ['F4', 'F6'], 's-train': ['F2', 'F7', 'F8'], 's-build': ['F3', 'F11'],
  's-data': ['F12', 'F14', 'F15'], 's-gate': ['F9', 'F10', 'F13'],
  'm-content-build': ['F16', 'F17', 'F18', 'F28', 'F29', 'F30'], 'm-content-text': ['F26', 'F27', 'F31', 'F45', 'F52'],
  'm-core-train': ['F21', 'F22', 'F23', 'F47', 'F48', 'F50'], 'm-bench-items': ['F24', 'F25', 'F44', 'F49', 'F53', 'F54', 'F55', 'F56'],
  'm-evals-infer': ['F32', 'F33', 'F34', 'F43'], 'm-gate-runs': ['F19', 'F20', 'F35', 'F36', 'F37', 'F41', 'F46'],
  'm-cli-tests': ['F38', 'F39', 'F40', 'F42'],
}
const GROUPS = Object.keys(IDS)
const SCHEMA = { type: 'object', properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, refuted: { type: 'boolean' },
  severity_if_real: { type: 'string', enum: ['Critical', 'Important', 'Minor', 'None'] },
  reasoning: { type: 'string' }, fix_recommendation: { type: 'string' },
  fix_size: { type: 'string', enum: ['none', 'small', 'medium', 'large'] } },
  required: ['id', 'refuted', 'severity_if_real', 'reasoning', 'fix_recommendation', 'fix_size'] } } }, required: ['verdicts'] }

function prompt(g) {
  return `You are an adversarial verifier in the final review of the AI Race repository (/home/user/AI-Race). First read the brief at ${SP}/verify2/brief.md and follow it exactly. Your group is '${g}' (findings ${IDS[g].join(', ')}); its findings are in ${SP}/verify2/${g}.json. Write your verdicts to ${SP}/verify2/verdicts-${g}.json as the brief says, and ALSO return the same verdicts (one per finding, same fields) through the StructuredOutput tool as {"verdicts": [...]}.`
}

phase('Verify')
const results = await parallel(GROUPS.map(g => () =>
  agent(prompt(g), { label: `verify3:${g}`, phase: 'Verify', schema: SCHEMA, model: 'sonnet' })
    .then(r => (r && IDS[g].every(id => r.verdicts.some(v => v.id === id))) ? r : null)))
const failed = GROUPS.filter((g, i) => !results[i])
if (failed.length) {
  log(`INCOMPLETE: ${failed.length} groups failed or returned incomplete verdicts (${failed.join(', ')}); resume re-runs only them`)
  return { incomplete: 'verify', failed }
}
const compact = []
for (const r of results) for (const v of r.verdicts) compact.push({ id: v.id, refuted: v.refuted, severity: v.severity_if_real, fix_size: v.fix_size })
compact.sort((a, b) => Number(a.id.slice(1)) - Number(b.id.slice(1)))
log(`stand: ${compact.filter(v => !v.refuted).length}/${compact.length}`)
return { verdicts: compact }
