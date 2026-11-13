# AI Race — Training system, next level: training text 2, a voice of its own, brain design 2

Date: 2026-11-13. Author: Claude (lead developer). Status: written at the owner's request ("develop a full plan on how to take this system to the highest level… save it for yourself"); development starts on the owner's go.

Builds on training engine 2 (`docs/superpowers/specs/2026-11-10-training-engine-2-design.md`, measurements in `docs/decisions/0009-training-engine-2.md`). Keeps the owner's decisions C49 (about half an hour is the longest run on an RTX 3050), C54 (typos stay, gentler), C55 (Careful is never a handicap), C56 (old AIs keep their scores, old worlds keep their rules) and R1–R8.

## 0. Decisions for the owner (assumed as written unless the owner says otherwise)
1. **An AI knows who it is.** Today an AI asked for its name invents one ("My name is Ruda", "I'm John") and talks about going to school, because most of its practice conversations are two people chatting. After this work the game tells each AI its name and its company at the start of a conversation, the AI has practised reading that line, and it answers as an AI. Nothing it says is scripted (R1): the line is what it reads, the answer is its own. Assumed: yes.
2. **The first AI's default reading leans a little toward conversations** (about a quarter instead of an eighth), if the measurements in §6 confirm that it makes the first chat better without hurting the rest. The player can still change every slider. Assumed: yes.
3. **Friends' download grows** from about 290 MB to about 550 MB of training text, and every friend runs the helper's setup file once more at the release (the same as after engine 2).
4. **Not assumed, asked later only if needed:** paying once (roughly 100–300 USD) to have a large model write a conversation set made exactly for these small brains. The plan reaches its targets without it if the free, openly licensed sets do what they are measured to do in §6; if they fall short of the identity and follow-up targets, this becomes a question for the owner with the measured gap attached.

## 1. What the owner gets (plain language)
- **A first AI that talks like a small AI, not like a confused person:** it knows its name, says it is an AI, answers the question it was asked more often, and stops rambling.
- **Conversations that hold together:** "And what about the cat?" is answered about the cat.
- **Later generations that are clearly better than the first**, and big brains that have enough to read.
- **A brain that learns more from the same reading** (a better internal design for brand-new AIs), if the measurements show it is worth having.
- **Lessons that stick:** what a player teaches is measured, and fixed if it does not stick.
- **Training that behaves on other people's computers:** slow cards explain themselves, weak cards do not crash.
- One release: friends update once, old worlds and old AIs keep working.

## 2. Where we stand (engine 2, RTX 3050, measured)
- First AI (Tiny, 12.6M tokens): 50–55 s; held-out loss 2.447; well-formed replies 80%; facts typed in lowercase without "?" answered right 65%.
- Gate: 5 of 7 (`docs/reports/2026-11-12-gate-engine-2.md`). Open: one lead of G3 (knowledge), G7 (cleaning).
- A grown brain ends about where a new brain of its size does on the same reading; two runs in a row cost 0.03 against one long run (ADR 0009 section 9).

## 3. What limits the AIs now (found 2026-11-13)
1. **What they read in conversations.** The conversation dataset is 19.8M tokens: 63% two-people small talk (SODA: the first speaker became "user", the other "AI", so the AI says "just hanging out with friends, going to school, and work"), 17% one-line fact questions, 17% one-line instructions, 2% assistant-style everyday talk. The middle document is 49 tokens long. Nothing in it tells the AI its name; the `<|sys|>` token exists and is used nowhere.
2. **Typos land in the AI's own answers.** 3% of conversation documents get typos over the whole document ("…the hawk is? Give a oone-wordd answer." → "Bire"): the AI is taught to misspell its replies.
3. **Pools are small for the new brains.** Each dataset holds 10–33M tokens. A Huge brain on the Marathon length with 60% on one dataset reads that dataset one and a half times in a single run, and a lineage reads it many times.
4. **Documents are drawn with replacement.** In a run that reads as much as a pool holds, about a third of the pool is never seen while other documents repeat.
5. **The optimizer is no longer the limit.** Measured today on top of engine 2: steps of 16,384 tokens instead of 8,192 make the first AI 10% faster and 0.03–0.05 worse; steps of 4,096 make it 40% slower and no better (Tiny 2.482, Small 2.357 against 2.447 and 2.352). Learning rate, schedule, iteration count, embedding rate and weight decay were swept for ADR 0009. More tuning there buys nothing.
6. **The vocabulary is not the limit either.** tok-v1 holds 3.22 bytes a token. A retrained vocabulary of 8,192 would hold 3.68 (+14%; conversations only +4%), 16,384 would hold 3.99 (+24%) — and would be a second model family, a second copy of every corpus and a larger embedding than the whole Tiny brain. tok-v1 stays frozen.
7. **The brain's internal design is the plain one** (pre-norm, RoPE, SwiGLU, tied embedding, GPT-2 start values). Known refinements exist; none has been measured here.
8. **How it talks.** Replies are drawn at temperature 0.6 / top-p 0.9 with a loop stop. A fifth of a first AI's replies are still malformed; cut-off replies, repeats and follow-up failures are not measured at all.
9. **Lessons.** Nobody has measured whether what a player teaches (coaching pairs, the notebook) is actually answered afterwards.

## 4. Approaches considered
- **A. Keep tuning the trainer.** Rejected: §3.5; the measured head-room is gone.
- **B. A new vocabulary and model family first.** Rejected: §3.6; at most +14% text per token for the largest disruption in the plan.
- **C. Teach every player's AI from a big hidden teacher model.** Rejected: it would make AIs better by putting in knowledge the player's data choices did not give them (R5: scores are measurements of what the player did), and it would flatten the differences between specialists that G3 already shows shrinking.
- **D. Text first, then voice, then brain design, each measured on one fixed scorecard, shipped as one release (chosen).** The text is the largest and surest lever and changes nothing about old AIs; the voice needs the text; the brain design is uncertain and is built only if a screening shows a real gain; a single release means friends update once.
- **E. A conversation set written to order by a large model.** Kept as the owner's option 0.4, after D's free sources have been measured.

## 5. The design

### 5.1 The measuring stick (first)
One command, `airace-ml scorecard`, trains fixed recipes (reusing finished runs) and measures every model the same way, writes `scorecard.json` and `scorecard.md`, and compares against a saved baseline.
- **Recipes:** `first` (Tiny 4×128, span 256, 12,582,912 tokens, even mix, seeds 1 and 2), `small` (Small 6×192, 12,582,912, even mix), `lineage` (first → Small → Medium 8×256, 12,582,912 each), `chatty` (Small, 60% conversations, 8,388,608: the gate's G3 recipe), `large` (Large 10×384, 25,165,824, even mix).
- **Measures per model:** held-out loss (mean and conversations); *yardstick* conversation loss on corpus v1's held-out conversations, so models trained on different text are comparable; bench-v1; and the talk probes, all drawn as the game talks:
  - well-formed share (the gate's judge rule; the judge is the yardstick root's, so it never moves);
  - casual facts (110 fact questions typed five ways);
  - follow-ups (two-turn fact questions where the second turn only makes sense with the first: "What sound does a cow make?" … "And a cat?");
  - identity (with the identity line: says the given name; says it is an AI when asked if it is a person; without the line: does not claim a person's life);
  - person-life claims (first-person family, school, job, body words in replies to small talk);
  - cut-off replies (ended by the token limit in the middle of a sentence) and loop stops;
  - lessons (ten taught question-answer pairs with made-up facts, trained the way the game trains coaching, then asked).
- **The yardstick root** is today's `data/` (corpus v1, judge v1, novelty index v1), kept unchanged as `data/yardstick/` after the release.
- **Baseline:** engine 2 on corpus v1, saved in `docs/reports/`.

### 5.2 Training text 2 (corpus v2)
Built by `airace-content build` into a second data root (`data2/`, chosen with `AIRACE_DATA`), so both corpora exist side by side until the release and every comparison is one environment variable.
- **Conversations (target 50M tokens):** assistant-style conversations from `HuggingFaceTB/smol-smoltalk` (Apache-2.0, made for models under a billion parameters; filtered to simple vocabulary, at most 6 turns, replies of at most 120 tokens, no code or mathematics) 30%; SODA 23%, keeping only dialogues in which the AI's side claims no person's life; everyday conversations 2% (all there is); fact questions 13%; instructions 13%; **follow-up chats** 12% (new generator over the knowledge base: a second question that leans on the first); **identity chats** 7% (new generator: name, maker, "are you a person", "do you have a family", with many phrasings).
- **The identity line:** `<|sys|>You are {name}, an AI made by {company}.<|end|>` (or without the maker) opens 40% of the assistant-style conversation documents and every identity chat, with names and companies drawn from large generated lists so that an AI learns to read its name, not to remember names. Never on SODA. Casual typing never touches it.
- **Typos only where people type:** in a conversation document typos fall on user turns only (C54: the mechanic stays; the AI's own answers are never misspelled by the data).
- **Bigger pools:** web 60M, books 30M, educational 40M, conversations 50M, code 20M, reasoning 20M, facts 12M, creative 40M tokens (272M in all; 139M today). Same sources and filters, more rows; noise rates of web and books unchanged.
- **Licences:** every new source is checked on its dataset card at a pinned revision and recorded in the manifest before it is used (project rule).
- **Acceptance:** on the scorecard, against corpus v1 with the same engine: first AI well-formed ≥ 88%, person-life claims ≤ 5%, follow-ups ≥ +15 points, yardstick conversation loss not worse by more than 0.03, bench-v1 overall not lower by more than the seed spread, casual facts not lower. Up to three rounds of changing shares and filters; what was tried is recorded in ADR 0010.

### 5.3 A voice of its own
- `encode_chat` and the chat prompt accept an identity line; the helper adds it only for an AI whose checkpoint says its latest run read text 2 or later (a run record carries `"text": 2`), so an AI from before never sees a token it was not trained on.
- The game sends each AI's name and company with chat, customer, keynote, interview and battle jobs. Benchmarks and the personality fingerprint stay without it (recorded scores remain comparable, C56).
- **Blocking, never rewriting (R1):** tested on the scorecard and adopted only where they raise the well-formed share by 3 points or more without lowering anything else: blocking a repeated run of four tokens; a floor on token probability relative to the best token (min-p) in place of top-p.
- **Acceptance:** identity ≥ 85% for Small and larger, ≥ 60% for Tiny; first AI well-formed ≥ 90%; cut-off replies ≤ 10%.

### 5.4 Engine 3 (small, measured items)
- Documents are drawn without replacement inside a run (a shuffled pass per dataset, resumable exactly).
- A lower loss weight on user turns in chat documents (tested at 0.3 and 0.6): adopted if it lowers the loss on replies by 0.02 or raises well-formed by 2 points with nothing else worse.
- Growth transfer, one day: a larger step for a grown brain's new parts in the first third of the run; starting the new embedding columns at small values with zeroed readers (still exactly function-preserving, R6). Adopted at ≥ 0.02 held-out gain on the `lineage` recipe. The honest ceiling is about 0.05.
- Lessons: if fewer than 7 of 10 taught pairs are answered, the coaching share and repetition are tuned until they are, without lowering the rest.
- Not done again: step size, learning rates, schedule (§3.5).

### 5.5 Brain design 2 (only if it earns its place)
- **Screening** in a scratch model class, at Tiny/12.6M, Small/12.6M and Medium/25.2M, two seeds, one change at a time and then together: output projections that start at zero; normalised queries and keys with a learned scale; a norm right after the embedding; a two-matrix squared-ReLU feed-forward of the same size in place of SwiGLU; RoPE base matched to spans of at most 512.
- **Gate:** built only if the combination is better by ≥ 0.04 held-out loss at equal time on all three sizes (or ≥ 3 bench-v1 points) — otherwise dropped and recorded.
- **If built:** `arch: 2` in the checkpoint; brand-new AIs use it everywhere, an AI that continues an earlier one keeps its family for ever; function-preserving growth, the KV cache and exact resume are proven again for it by the same tests as design 1; the helper carries both.

### 5.6 The release (one)
- In `data2/`: novelty index, judge (trained with the released engine), catalog, rival pack 3; the gate run again; `npm run e2e`; scorecard against the baseline.
- The server keeps earlier rival packs loaded for the rivals already seated in old worlds whatever data they were built from (today a pack from other data is refused, which would strand them), and seats new worlds only from a pack built on the current data.
- `ENGINE_VERSION` 3 (helpers update once); the data folders swap; the old ones become `data/yardstick/`.
- Old worlds: their AIs chat and battle as before; their next runs read text 2; their rules do not change.

### 5.7 Every computer (after the release)
- `airace-ml doctor`: the card, its driver, whether fast half precision works, the speed-test result and a real training step's speed, and a plain sentence when something is wrong (running on the wrong graphics chip, no fast half precision, thermal slow-down); the helper sends the sentence to the game and the cards window shows it. This is how the friend's "three to four times slower" card gets explained.
- Fallbacks: cards without fast bf16 use the precision that is fastest for them; out of memory halves the step in two passes instead of failing (the same result, summation order aside).
- A go/no-go spike for cards that are not NVIDIA; a Mac helper stays out of this plan.

## 6. Order, gates and plans
| Phase | What | Gate to pass | Plan file |
|---|---|---|---|
| A | The measuring stick and the baseline | baseline saved | `plans/2026-11-13-next-level-a-text-and-voice.md` |
| B | Engine 3 items | each item's own rule (§5.4) | same |
| C | Training text 2 | §5.2 acceptance | same |
| D | A voice of its own (helper and game) | §5.3 acceptance; the owner tries a first AI in a second game on `data2/` | same |
| E | Brain design 2 | §5.5 gate | written when phase D passes |
| F | The release | gate, e2e, scorecard ≥ baseline everywhere | written when E is decided |
| G | Every computer | doctor explains a slow card on a second PC | written after F |

Rough size: A half a session, B half, C two, D half, E one or two, F one, G one or two (a session being a day's work like engine 2's).

## 7. Targets (scorecard, RTX 3050)
| | engine 2 today | target |
|---|---|---|
| First AI: well-formed replies | 80% | ≥ 90% |
| First AI: facts typed casually | 65% | ≥ 80% |
| First AI: gives its name / says it is an AI | invents a name | ≥ 60% (Tiny), ≥ 85% (Small and up) |
| First AI: claims a person's life in small talk | to be measured | ≤ 5% |
| Follow-up questions answered (Small) | to be measured | baseline + 15 points, at least 60% |
| Third generation (Medium) bench-v1 overall | about 50–55 | + 5 |
| Lessons answered after one run | to be measured | ≥ 7 of 10 |
| First AI: training time | 50–55 s | no slower |
| Gate | 5 of 7 | no criterion lost; G3 and G7 measured again |

Targets marked "to be measured" are fixed in ADR 0010 when the baseline exists; a target that turns out to be unreachable is reported with its numbers, never quietly lowered.

## 8. Compatibility
- tok-v1, bench-v1 and every old checkpoint stay valid; old AIs' recorded scores are never recomputed.
- One helper update (engine 3). A helper of engine 2 gets no work after the release, as built for engine 2.
- Replay of old worlds is untouched: nothing in `packages/sim` changes its rules for a recorded balance version.
- R2: the identity line holds only an AI's public name and its company's public name.

## 9. Risks
- The new conversations are too hard for a Tiny brain → the simplicity filter and the length limits; three measured rounds; SODA's share is the fallback.
- A Tiny brain cannot copy a name it reads → the target is lower for Tiny, and the line is still an honest improvement from Small up.
- Brain design 2 gains nothing on top of the new optimizer → the gate drops it; one or two sessions lost, nothing shipped.
- Downloads fail during the big build → the build already resumes; sources are pinned by revision.
- Creativity scores shift with a new judge and novelty index → only AIs scored after the release are affected; said in the release notes.
- The rebuilt judge changes the "well-formed" rule inside the game → the scorecard keeps the old judge as the yardstick, so progress is still measured on one scale.

## 10. Out of scope
A new vocabulary; teacher models; scripted or rewritten replies; changing bench-v1; the installer, sign-in, moderation and hosting questions of a public release; a Mac helper; music.
