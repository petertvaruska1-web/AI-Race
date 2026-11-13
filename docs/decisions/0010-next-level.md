# ADR 0010: the next level of the training system (measurements and the values they set)

Date started: 2026-11-13. Design: `docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md`. Plan A: `docs/superpowers/plans/2026-11-13-next-level-a-text-and-voice.md`. Earlier measurements: ADR 0009.

Every task of the plan that measures something writes its numbers and its decision here, kept or removed.

## 1. Measured while planning (2026-11-13, RTX 3050, engine 2)
- **Vocabulary** (`docs/handoff/upgrade-2026-11-10/tok_probe.py`): tok-v1 holds 3.22 bytes a token; a retrained 8,192 vocabulary 3.68 (conversations 3.67 against 3.53), 16,384 holds 3.99. "hello" is 3 tokens in tok-v1 and 2–3 after retraining. **Decision:** tok-v1 stays; casual typing is taught, not tokenized away.
- **Tokens a step** (`train_product.py --batch-tokens`), held-out loss and time for 12.6M tokens: Tiny at 8,192 (the product) 2.447 in 54.9 s; at 16,384: 2.502 in 49.4 s (2.478 with a bolder rate); at 4,096: 2.482 in 75.9 s. Small at 8,192: 2.352 in 82.9 s; 16,384: 2.359 in 79.8 s; 4,096: 2.357 in 107 s. **Decision:** 8,192 stays.
- **Corpus v1's conversations:** 19.8M tokens in 235,749 documents (median 49 tokens): SODA 12.5M (two people; the first speaker is the user, every other the AI), fact questions 3.4M, instructions 3.4M, everyday conversations 0.5M. 3% of the documents carry typos over the whole document, the AI's turns included. `<|sys|>` is used nowhere. The sampler draws documents with replacement.

## 2. Sources and licences for training text 2
- **`HuggingFaceTB/smol-smoltalk`** (card licence apache-2.0; 484,570 rows; columns `messages`, `source`). Its parent card (`HuggingFaceTB/smoltalk`) says: "All the new datasets (Smol-Magpie-Ultra, Smol-contraints, Smol-rewrite, Smol-summarize) are licensed under Apache 2.0" and "For the existing public datasets, please refer to the original dataset for the license". **Decision:** only rows whose `source` is `smol-magpie-ultra-short` or `smol-contraints` (the card's spelling) are used. Rows from OpenHermes, Self-OSS-Instruct, SystemChat and the other existing public datasets are left out: their licences are their originals', not checked. The rewriting and summarizing rows are Apache-2.0 but are not conversation practice for these brains.
- Everyday conversations keep coming from `HuggingFaceTB/everyday-conversations-llama3.1-2k` (apache-2.0, already in the manifest).
- Checked on the dataset cards on 2026-11-13; the revision is pinned in `airace_content/sources.py` when the source is added (plan A, Task 11).

## 3. The scorecard baseline (engine 2, text 1)
(Task 4.)

## 4. Engine 3 items
(Tasks 5–8: numbers and keep/remove for each.)

## 5. Training text 2 against text 1
(Tasks 14–15.)

## 6. A voice of its own
(Tasks 16–18.)
