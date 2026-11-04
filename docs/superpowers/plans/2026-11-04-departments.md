# Departments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The owner prefers inline execution, committing and pushing after every task.

**Goal:** Companies fund four departments (Research, Data team, Sales & marketing, Support & operations) with monthly budgets, each with a real, visible effect.

**Architecture:** Department budgets are company state in the sim. The team grows toward the budget over about a month, and a strength curve with diminishing returns scales each effect, which hooks into existing rules: the training preview, check costs, awareness, satisfaction and serving costs. Rivals fund departments by style.

**Tech Stack:** As Stage 2.

**Spec:** `docs/superpowers/specs/2026-11-04-remaining-features-design.md` §4 (decision C43).

## Global Constraints

- **Sim version:** the next free one. Upgraded worlds give every department budget 0 and staffed 0.
- **Departments:** `DEPARTMENTS = ["research", "data", "sales", "support"]`, named "Research", "Data team", "Sales & marketing", "Support & operations".
- **Budget:** $0–$500,000 a month in $10,000 steps. Refusal: "Budgets are $0–$500,000 a month, in $10,000 steps."
- **Spend:** booked daily as `budget / 30` to the new cost category `team` ("Team").
- **Growth:** `staffed += (budget − staffed) / 30` each day, before effects.
- **Strength:** `s = 1 − exp(−staffed / 200,000)`.
- **Effects:**
  - **Research:** tokens × `(1 + 0.3·s)` at the same price.
  - **Data team:** check costs × `(1 − 0.5·s)`.
  - **Sales & marketing:** daily awareness `+= 0.002·s·(1 − a)` in every region and group.
  - **Support & operations:** satisfaction drops × `(1 − 0.3·s)`, and serving compute cost × `(1 − 0.2·s)`.
- **Autopilot:** budgets pause (no spend; staffed shrinks) while the owner is offline and cash is below 0.
- **Balance:** each department changes a typical company's year profit by at most ±25% at $100,000 a month. The harness pins it.

## Review Focus

1. **A run started before a budget change keeps the tokens and price it was previewed with** (the plan's preview is recorded at start). Owner: Task 2.
2. **A budget cut to $0** shrinks the team over a month; it doesn't vanish at once. Owner: Task 1.
3. **Negative cash on autopilot** pauses spending, and spending resumes when the owner returns. Owner: Task 1.
4. **Rivals with no revenue** fund nothing. Owner: Task 3.
5. **Satisfaction rises are unchanged by Support**; only drops soften. Owner: Task 2.

---

### Task 1: Departments in the rules

**Files:**
- Create: `packages/sim/src/departments.ts`, `packages/sim/test/departments.test.ts`
- Modify:
  - `packages/protocol/src/world.ts` (`DEPARTMENTS`, `DEPARTMENT_NAMES`, command `setDepartment {department, budget}`, cost category `team`)
  - `packages/protocol/src/view.ts` (`me.company.departments`)
  - `packages/sim/src/state.ts`, `packages/sim/src/reducer.ts` (tick: `departmentsDay` before `campaignsDay`)
  - `packages/sim/src/balance.ts` (`departments: {maxBudget: 500000, step: 10000, growDays: 30, scale: 200000}`)
  - `packages/sim/src/companyExtras.ts`, `packages/sim/src/upgrade.ts`
- Test: also `packages/sim/test/properties.test.ts` (the money identity with the new cost)

**Interfaces:**
- Produces:
  - `departmentStrength(c: CompanyState, d: Department): number`.
  - View `departments: Record<Department, {budget, staffed, strength}>` (owner only).

- [ ] **Step 1: Write the failing tests:**
  - A budget of $300,000 books $10,000 a day.
  - After 30 days staffed is about 63% of the budget.
  - A cut to 0 halves the team in about 21 days.
  - An off-step budget is refused with the copy.
  - The money identity holds.
  - Offline with negative cash: no spend.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run `npm test` and `npm run check`: PASS.
- [ ] **Step 5: Commit** `feat(sim): fund four departments with monthly budgets`.

### Task 2: The four effects

**Files:**
- Modify:
  - `packages/sim/src/training.ts` (`previewTraining` passes the research strength; `previewPlan(..., researchStrength = 0)` multiplies tokens; the check costs use the data strength)
  - `packages/sim/src/market.ts` (sales awareness)
  - `packages/sim/src/feed.ts` (`updateSatisfaction` softens drops)
  - `packages/sim/src/compute.ts` or `ledger.ts` (serving cost)
  - `apps/server/src/training/config.ts` (`token_budget` = the run's recorded `preview.tokens`)
- Test: `packages/sim/test/departments.test.ts`, `apps/server/test/training.test.ts`

**Interfaces:**
- Produces:
  - Each effect is at its formula's value.
  - The trainer's token budget equals the preview's tokens, so the AI really reads more.

- [ ] **Step 1: Write the failing tests:**
  - Research at strength 0.5 gives 1.15× tokens at the same cost, and the server config's `token_budget` matches.
  - Data at 0.5 makes fact-checking 25% cheaper.
  - Sales raises awareness by the formula.
  - Support at 0.5 makes a drop 15% smaller and leaves a rise unchanged, and serving costs 10% less.
  - A run keeps its recorded preview after the budget changes.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): departments change training, checks, awareness, satisfaction and serving costs`.

### Task 3: Rivals fund departments, and the balance

**Files:**
- Modify: `packages/sim/src/rivals.ts` (monthly budgets by style from last month's revenue: premium research 6%; budget support 5%; specialist research 3% and data 3%; consumer sales 6%; rounded down to $10,000 steps, at most $500,000), `packages/sim/test/balance.test.ts` and its bots
- Test: `packages/sim/test/balance.test.ts`

- [ ] **Step 1: Write the failing tests:**
  - A harness year where the typical bot funds each department at $100,000 a month in turn: each changes its year profit by at most ±25% and by at least ±2% (no waste).
  - Rivals' budgets follow their style and revenue; no revenue means no budget.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement**, tuning the effect sizes in the balance only if a target fails. Ledger each tuned number.
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): rivals fund departments by style; departments balanced`.

### Task 4: The Team panel

**Files:**
- Create: `apps/client/src/spending/Team.tsx`
- Modify: `apps/client/src/screens/Company.tsx` (the panel), `apps/client/src/screens/Money.tsx` (the "Team" line), `apps/client/src/windows/ReportWindow.tsx` (the team's month cost)
- Test: `apps/client/test/team.test.tsx`

**Interfaces:**
- Produces: four rows, each with:
  - the department name and its effect line ("Training reads 12% more text", "Checks cost 20% less", "+0.1% awareness a day", "Bad answers sting 9% less · serving 6% cheaper") computed from `strength`
  - a budget slider in $10,000 steps (commits on release) and a strength bar

- [ ] **Step 1: Write the failing tests:**
  - The slider sends `setDepartment` with a step value.
  - The effect line shows the formula's value at the view's strength.
  - The Money screen shows the Team line once there is spend.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): the Team panel`.

### Task 5: End to end and docs

- [ ] **Step 1:** Extend the solo E2E: set Research to $100,000 and see the Team line on the Money screen.
- [ ] **Step 2:** Run everything: PASS.
- [ ] **Step 3: Play a game year on the GPU** with departments; ledger whether any felt useless or must-have.
- [ ] **Step 4: Docs:** a progress entry, the handoff resume point (the licensing plan), the ledger. Commit, push and report.
