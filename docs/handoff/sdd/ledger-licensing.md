# SDD ledger — plan: docs/superpowers/plans/2026-11-04-licensing.md

Executed inline (superpowers:executing-plans), committing and pushing after every task; one self-review at the end (the owner prefers no helper agents). Spec: docs/superpowers/specs/2026-11-04-remaining-features-design.md §5 (C44). Started 2026-11-07 on the owner's go ("start building licensing").

Pre-flight:
- Task 1 → Task 2: Task 1 produces `Deal` and `activeLicense(s, productId, group)`; Task 2's market reads the licensed version for a product's group through it, and the month close settles each deal. Consistent once the deal records the licensor and version.
- Task 1 → Task 3: the server splits a product's feedback batch by `activeLicense` per group. Consistent.
- Task 1 → Task 4: rivals make offers and proposals through the same functions as players' commands. Consistent.
- Task 1 → Task 5: the view's public offers and the owner's deals and proposals are what the windows read.
- Departments ended at sim-9, so this is sim-10.
Task 1: Ruling: ids come from a world counter (`offer-N`, `proposal-N`, `deal-N` in `licenses.seq`), not from the gateway, so rivals' offers and deals are named the same way as players' — cost if wrong: none.
Task 1: Ruling: `takeLicense` also names the `groups` it takes (a subset of the offer's), because the balance test licenses a rival's AI "for its weakest group" and a rival offers all six; `proposeLicense` names the proposer's `productId`, which the deal needs — cost if wrong: two schema fields.
Task 1: Ruling: an AI is "on sale" when one of its maker's products built on it is live; one open offer per AI; a running exclusive deal blocks new offers, proposals and takes of that AI ("That AI is licensed exclusively."), and an exclusive offer needs an AI nobody licenses yet; a product's group can have one license at a time ("That AI already uses a licensed AI for one of those groups."); players can't propose to rivals ("Rivals license their AIs only through their offers."); a proposal lapses after 30 days (`PROPOSAL_DAYS`) — cost if wrong: copy and small rules.
Task 1: Ruling: deals carry `aiName` (the licensor's product name when offered) and `owed` (royalties built up since the last close, Task 2); the view gives every player `licenseOffers` and the owner `me.company.licenses: {deals, proposals}` seen from its side (`role`, `otherName`, the licensee's `productName`); `endLicense` is wired but refused until Task 2 gives it its money — cost if wrong: none.
Task 1: complete (tests: sim licensing 10/10; npm test 797/797; check clean). sim-10: commands `offerLicense`, `withdrawOffer`, `takeLicense`, `proposeLicense`, `answerProposal`, `endLicense`; income `licenseIncome`, cost `licenses` (client labels "License income", "Licenses"); `WorldState.licenses {offers, proposals, deals, seq, lastRequest}` and `sim10FromSeq`; `UPGRADABLE` gains sim-9.
