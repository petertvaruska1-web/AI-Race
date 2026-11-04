# Licensing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The owner prefers inline execution, committing and pushing after every task.

**Goal:** Companies, players and rivals alike, rent their AIs to each other to power products for chosen customer groups, for monthly royalties.

**Architecture:**
- Offers, proposals and deals are world state.
- The market uses a licensed AI's measured scores and traits for the licensed groups of the licensee's product, and royalties move at month close.
- Customer questions in licensed groups are answered by the licensed AI, on its maker's card (Stage 2 owners).
- Rivals offer and request licenses by style.

**Tech Stack:** As Stage 2.

**Spec:** `docs/superpowers/specs/2026-11-04-remaining-features-design.md` §5 (decision C44). Builds on Stage 2 (job owners, the Directory) and departments (sim version order).

## Global Constraints

- **Sim version:** the next free one; upgraded worlds have no offers or deals.
- **Terms:**
  - price $0.10–$20.00 per customer per month, in cents
  - months 3, 6 or 12
  - one or more of the six groups
  - `exclusive` boolean
- **Money:** at each month close the licensee pays `price × its customers in the licensed groups at close` (cost category `licenses`, "Licenses"), and the licensor receives it (income category `licenseIncome`, "License income").
- **Ending:**
  - the term ends
  - the licensor retires the licensed AI's last live product, or the AI is no longer theirs to sell
  - the licensee ends it early, paying one month's royalties at the current customer count
- **Market:** for the licensee's product in licensed groups, the utility uses the licensed version's `scores` and `traits`; other groups are unchanged.
- **Posts** in licensed groups say "powered by {AI name}" and quote the licensed AI's real replies (R1).
- **Rivals:**
  - They offer their newest AI at style prices: premium $8, budget $1.50, specialist $5, consumer $3.
  - They request a license of a player's AI when its score in the rival's focus skill beats the rival's best by 10 or more, offering their style price for 6 months, at most once per 90 days per rival.
- **Opacity:** a licensee sees only the licensed AI's public facts and its replies to the licensee's customers.
- **News:** "{Licensee} licenses {Licensor}'s {AI} for its {groups}"; "{Licensee}'s license of {AI} ends".

## Review Focus

1. **A licensed AI retired mid-term:** the deal ends that day, its groups fall back to the licensee's own AI, and the royalty is pro-rated for that month. Owner: Task 2.
2. **An exclusive offer taken twice at once** (two players in the same second): the second take is refused. Owner: Task 1.
3. **A licensee with no customers in the licensed groups** pays $0, not a minimum. Owner: Task 2.
4. **The licensor's card offline (Fair off):** the licensed groups' customer questions wait like the licensor's own, and expire after 14 days. Owner: Task 3.
5. **Licensing your own AI to yourself is refused.** Owner: Task 1.

---

### Task 1: Offers, proposals and deals in the rules

**Files:**
- Create: `packages/sim/src/licensing.ts`, `packages/sim/test/licensing.test.ts`
- Modify:
  - `packages/protocol/src/world.ts` (commands `offerLicense {versionId, priceCents, months, groups, exclusive}`, `withdrawOffer {offerId}`, `takeLicense {offerId, productId}`, `proposeLicense {versionId, priceCents, months, groups}`, `answerProposal {proposalId, accept}`, `endLicense {dealId}`; the new money categories)
  - `packages/protocol/src/view.ts` (public offers; your deals and proposals)
  - `packages/sim/src/state.ts` (`WorldState.licenses: {offers, proposals, deals}`)
  - `packages/sim/src/reducer.ts`, `packages/sim/src/upgrade.ts`
- Test: `packages/sim/test/properties.test.ts` (money identity)

**Interfaces:**
- Produces:
  - `Deal = {id, licensorId, licenseeId, versionId, productId, priceCents, groups, startDay, endDay, exclusive}`.
  - Refusals:
    - "That offer has been taken."
    - "You can't license your own AI."
    - "Pick one of your AIs on sale."
    - "Prices are $0.10–$20 a customer a month."
  - `activeLicense(s, productId, group): Deal | null`.

- [ ] **Step 1: Write the failing tests:**
  - An offer is public.
  - Taking it creates a deal for the chosen product and groups; an exclusive offer closes, and a second take is refused.
  - Proposing and accepting creates a deal; declining doesn't.
  - Self-licensing is refused.
  - An upgraded world has empty licenses.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): license offers, proposals and deals`.

### Task 2: Licensed AIs in the market, royalties and endings

**Files:**
- Modify: `packages/sim/src/market.ts` (scores and traits per group via `activeLicense`), `packages/sim/src/ledger.ts` (royalties at month close, pro-rated for days active), `packages/sim/src/licensing.ts` (`licensesDay`: endings), `packages/sim/src/feed.ts` (news), `packages/sim/src/reducer.ts` (tick: `licensesDay` after `productsDay`)
- Test: `packages/sim/test/licensing.test.ts`

- [ ] **Step 1: Write the failing tests:**
  - A licensee's students choose by the licensed AI's knowledge score.
  - The month close books the price × students to both sides.
  - A retired licensed AI ends the deal with a pro-rated royalty.
  - Early ending charges one month.
  - No customers means $0.
  - The money identity holds.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): licensed AIs win customers with their own scores; royalties move monthly`.

### Task 3: Licensed answers to customers

**Files:**
- Modify: `apps/server/src/feedback/service.ts` (split a product's batch: licensed groups' questions go to a feedback job for the licensed version, owned by the licensor), `packages/sim/src/feed.ts` (`FeedbackResult.poweredBy?: string`; posts carry it), `packages/protocol/src/view.ts` (`PostView.poweredBy`)
- Test: `apps/server/test/feedback.test.ts`, `packages/sim/test/feed.test.ts`

- [ ] **Step 1: Write the failing tests:**
  - A licensee's batch makes two jobs, the licensed groups' one owned by the licensor with the licensed model.
  - The posts in those groups say powered by the AI and quote its reply byte for byte.
  - Satisfaction updates the licensee's product.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(server): customers in licensed groups are answered by the licensed AI`.

### Task 4: Rivals license

**Files:**
- Modify: `packages/sim/src/rivals.ts` (offers on each release; requests by the rule), `packages/sim/src/licensing.ts`
- Test: `packages/sim/test/licensing.test.ts`, `packages/sim/test/balance.test.ts`

- [ ] **Step 1: Write the failing tests:**
  - A rival's new release makes an offer at its style price.
  - A player's AI that beats a rival's focus skill by 10 brings one proposal within 90 days, and not a second.
  - The balance harness: a typical company that licenses a rival's AI for its weakest group for a year stays within ±25% of its unlicensed profit.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(sim): rivals offer their AIs and ask to license yours`.

### Task 5: Licensing on screen

**Files:**
- Create: `apps/client/src/spending/Licensing.tsx` (Offer window, Take window, Proposal window, Deals panel)
- Modify: `apps/client/src/windows/AiWindow.tsx` ("Offer for license"), `apps/client/src/screens/Directory.tsx` and `Rivals.tsx` (offers with "License it"), `apps/client/src/screens/Money.tsx` (the two lines), `apps/client/src/components/PostCard.tsx` ("powered by"), `apps/client/src/components/Toasts.tsx` (a rival's request)
- Test: `apps/client/test/licensing.test.tsx`

- [ ] **Step 1: Write the failing tests:**
  - The Offer window sends exact terms.
  - The Take window lists your live products and the groups.
  - A request toast opens the Proposal window with Accept and Decline.
  - A post shows "powered by Pengu".
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4:** Run: PASS.
- [ ] **Step 5: Commit** `feat(client): offer, take and answer licenses`.

### Task 6: End to end and docs

- [ ] **Step 1:** Extend the solo E2E: take a rival's offer for students; the next report card shows Licenses. Extend the friends E2E: B takes A's offer, and A sees License income.
- [ ] **Step 2:** Run everything: PASS.
- [ ] **Step 3: Play it on the GPU**; ledger.
- [ ] **Step 4: Docs:** a progress entry, the handoff resume point (the AI Hub plan), the ledger. Commit, push and report.
