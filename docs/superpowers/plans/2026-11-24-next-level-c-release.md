# Next level, plan C: the release — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make training text 2, engine 3 and the "conversations 30%" first reading what the main game runs, without breaking a single old world.

**Architecture:** Code first (rival pack 3 on both sides, a new balance version for new worlds, text 2 as the default build, the measuring stick's own folder, helpers that stop downloading rival packs), then the graphics-card work on `data2/` (rival pack 3, the gate on text 2, the catalog), then one checked swap of the data folders (text 2 into `data/`, text 1 into `data/yardstick/`), then every check on the swapped folders.

**Tech Stack:** Python 3.13 (uv, torch) in `ml/`; TypeScript (npm workspaces, Vitest, Biome, Playwright) in `apps/` and `packages/`.

**Spec:** `docs/superpowers/specs/2026-11-13-ai-training-next-level-design.md` section 5.6 (the release), with `docs/handoff/next-steps.md` "What the release must do" and ADR `docs/decisions/0010-next-level.md` (every measured value this plan relies on).

## Global Constraints

- R1–R8 (`CLAUDE.md`). R5: no unlock lists; R6: runs continue from their parent; R2: the identity line holds only public names.
- Old worlds keep their rules: nothing in `packages/sim` changes for `balance-v2` or `balance-v3`; their AIs chat and battle as before; their next runs read text 2 (spec 5.6).
- New worlds: `balance-v4`, whose default reading is **conversations 30, the other seven datasets 10 each, notebook 0** (ADR 0010 §5.3).
- Rival packs: `data/rivals/v1` and `data/rivals/v2` are never deleted or built into. Text 2's pack is `rivals-v3` in `rivals/v3`, keys `{lineage}-r3-v{k}`.
- `data/server/` (the owner's real worlds) is backed up before anything moves and is never used for a test game.
- Ask the owner before any long graphics-card run, with the time it takes (rivals about 75 minutes, the gate about two hours).
- `ENGINE_VERSION` and `TRAINING_ENGINE` stay 3. Training code (`train`, `model`, `data`, `tokenizer.py`, `infer`, `engine.py` under `ml/src/airace_ml`) is not touched, so every cached run (gate, scorecard) stays reusable.
- TDD for logic; commit after each task on `m2-two-friends` with the session's trailer; push.

## Review Focus

1. **Two packs with the same AI key** (someone installs a pack whose keys another loaded pack already uses, e.g. data2's old "-e2-" pack): the server refuses the later pack with a logged error and every other pack keeps working. Test in Task 3.
2. **`build-rivals` pointed at a folder that holds another pack version** (`data/rivals/v2`): refused with exit 2 before anything is written. Test in Task 1.
3. **A new world while the current pack was built on other data, or is missing:** no rivals (a warning), never seated from an earlier pack; a world of old rules whose own pack is missing falls back to the current pack. Test in Task 3.
4. **A helper fetching from a host whose list still names rival-pack files** (a host from before the release): those files are skipped, the fetch succeeds. Test in Task 5.
5. **The swap started twice or half done** (`data/yardstick` already there, `data2/corpus` already gone): every step of Task 7 checks first and stops before moving anything.

---

### Task 1: Rival pack 3 (ML side)

**Files:**
- Modify: `ml/src/airace_ml/rivals/build.py` (module docstring, `PACK_VERSION`, `version_key`, a folder guard)
- Modify: `ml/src/airace_ml/paths.py` (`rivals_dir`)
- Modify: `ml/src/airace_ml/cli.py` (`_cmd_build_rivals`: the guard's refusal is a usage error, exit 2)
- Test: `ml/tests/test_rivals.py`, `ml/tests/test_cli.py`

**Interfaces:**
- Produces: `PACK_VERSION = "rivals-v3"`; `version_key(recipe, k) -> f"{recipe.id}-r3-v{k}"`; `rivals_dir(root) -> root / "rivals" / "v3"`; `check_pack_folder(out: Path) -> None` raising `ValueError` naming the other version when `out/rivals.json` holds a pack whose `"version"` is not `PACK_VERSION`.

- [ ] **Step 1: Failing tests** — `test_version_keys_of_pack_3_carry_r3` (`version_key(LINEAGES[0], 2) == f"{LINEAGES[0].id}-r3-v2"`, `PACK_VERSION == "rivals-v3"`); `test_rivals_dir_is_v3` (`rivals_dir(tmp) == tmp / "rivals" / "v3"`); `test_a_folder_holding_another_pack_is_refused` (write `{"version": "rivals-v2"}` as `rivals.json`; `check_pack_folder` raises `ValueError` matching `rivals-v2`; an empty folder and a `rivals-v3` index pass); in `test_cli.py`, `build-rivals --quick --out <folder with a rivals-v2 index>` returns 2 and leaves the folder's files unchanged.
- [ ] **Step 2:** Run `cd ml && python -m uv run pytest tests/test_rivals.py tests/test_cli.py -k "r3 or v3 or another_pack" -q`; expected FAIL.
- [ ] **Step 3:** Implement. `build_rival_pack` calls `check_pack_folder(out)` first. Docstrings say: `rivals-v3` is text 2's pack (engine 3); `v1` and `v2` are kept for the worlds seated from them and never built again.
- [ ] **Step 4:** Run `cd ml && python -m uv run pytest tests/test_rivals.py tests/test_cli.py -q` and `python -m uv run ruff check .`; expected PASS, clean.
- [ ] **Step 5:** Commit `feat(ml): rival pack 3 for training text 2 (rivals-v3, keys -r3-)`.

### Task 2: balance-v4 (new worlds read conversations first)

**Files:**
- Modify: `packages/sim/src/balance.ts` (`BalanceVersion`, `defaultPlan(…, mix)`, `TALK_FIRST_MIX`, `BALANCE_V4`, `getBalance`)
- Modify: `packages/protocol/src/view.ts:620` (enum gains `"balance-v4"`)
- Modify: `apps/server/src/main.ts` (default `balance-v4`; the accepted list and its error message)
- Test: `packages/sim/test/training.test.ts` (or the balance test beside it)

**Interfaces:**
- Produces: `BalanceVersion = "balance-v2" | "balance-v3" | "balance-v4" | "balance-test"`; `BALANCE_V4` = `BALANCE_V3` with `version: "balance-v4"` and `defaultPlan.mix = { web: 10, books: 10, educational: 10, conversations: 30, code: 10, reasoning: 10, facts: 10, creative: 10, notebook: 0 }`.

- [ ] **Step 1: Failing tests** — `balance-v4 differs from balance-v3 only in the default reading` (`getBalance("balance-v4").defaultPlan.mix` equals the mix above; `{...BALANCE_V4, version, defaultPlan.mix}` deep-equals `BALANCE_V3` with the same two fields); `the older rules keep the even reading` (`balance-v2` and `balance-v3` default mixes are 12 each, notebook 0).
- [ ] **Step 2:** `npx vitest run packages/sim` — FAIL.
- [ ] **Step 3:** Implement; the doc comment on `BALANCE_V4` cites ADR 0010 §5.3 (blind reading +14 to +23 points, lowercase facts 88–91% against 72–73%, bench not lower).
- [ ] **Step 4:** `npm test` and `npm run check` — PASS.
- [ ] **Step 5:** Commit `feat(game): new worlds' first reading leans on conversations (balance-v4)`.

### Task 3: The server keeps every rival pack

**Files:**
- Modify: `packages/protocol/src/rivals.ts` (`RivalPack.version` enum gains `"rivals-v3"`; export `RivalPackVersion`)
- Modify: `apps/server/src/config.ts` (`legacyRivalsDir` → `earlierRivalsDirs?: readonly string[]`)
- Modify: `apps/server/src/rivals/pack.ts` (`loadRivalPack(dir, catalog, log, { current: boolean })`: the data check only for the current pack)
- Modify: `apps/server/src/rivals/service.ts` (`current`, `earlier`, `PACK_FOR_RULES`, `packFor`, `packOf`, `peersIn` over every pack; overlapping keys refused)
- Modify: `apps/server/src/main.ts` (`rivalsDir: data/rivals/v3`, `earlierRivalsDirs: [data/rivals/v2, data/rivals/v1]`)
- Test: `apps/server/test/rivals.test.ts`, `apps/server/test/rivalPack.ts` (`writeFixturePack(identity, version)` writes v1, v2 or v3 keys)

**Interfaces:**
- Consumes: `BalanceVersion` with `balance-v4` (Task 2).
- Produces: `PACK_FOR_RULES: Record<BalanceVersion, RivalPackVersion>` = `{ "balance-v2": "rivals-v1", "balance-v3": "rivals-v2", "balance-v4": "rivals-v3", "balance-test": "rivals-v3" }`; `RivalService.packFor(balance)` = the loaded pack of the world's rules, else the current pack, else null; `RivalService.current: LoadedPack | null`, `RivalService.earlier: LoadedPack[]`.

- [ ] **Step 1: Failing tests** (replace the two tests that name `legacyRivalsDir`):
  - `every world draws from the pack of its rules` — packs v1, v2 (another data identity) and v3 (the catalog's): `packFor("balance-v2")` is v1, `"balance-v3"` is v2, `"balance-v4"` and `"balance-test"` are v3; replies and model URLs resolve for `premium-a-v1`, `premium-a-e2-v1`, `premium-a-r3-v1`; a new world is seated with `-r3-` keys only.
  - `an earlier pack built on other data still serves the rivals seated from it` — v2 on `another-identity` loads; `rivals.replies("premium-a-e2-v1")` is not empty.
  - `a new world is never seated from an earlier pack` — only v2 installed (current missing), server on `balance-test`: `installed` is false, the world gets no rivals; a world of `balance-v3` rules still gets v2.
  - `a world of older rules whose pack is missing draws from the current pack` — only v3: `packFor("balance-v2")` is v3.
  - `a pack whose keys another pack already uses is refused` — two earlier packs with the same keys: the second is not loaded, an error is logged, the first serves.
  - Keep `runs without rivals when the pack is missing or was built from other data` for the current pack.
- [ ] **Step 2:** `npx vitest run apps/server/test/rivals.test.ts` — FAIL.
- [ ] **Step 3:** Implement. `loadRivalPack` with `current: false` skips the data-identity check (it logs at info which data the pack was built on); with `current: true` it keeps today's warning and refusal.
- [ ] **Step 4:** `npm test`, `npm run check` — PASS.
- [ ] **Step 5:** Commit `feat(server): every rival pack stays loaded; new worlds only from the current data's pack`.

### Task 4: Text 2 is the default; the measuring stick gets its own folder

**Files:**
- Modify: `ml/src/airace_content/build.py` (`RELEASED_TEXT = 2`: the text of a build into a root that holds none), `ml/src/airace_content/cli.py` (`--text` help)
- Modify: `ml/src/airace_ml/catalog.py` (the Conversations blurb by text)
- Modify: `ml/src/airace_ml/paths.py` (`yardstick_root(root=None) -> Path`), `ml/src/airace_ml/cli.py` (scorecard default `--yardstick-root`)
- Modify: `tools/processes.ts` (`ensureCatalog`'s population: `ml/runs/gate-3/runs`)
- Modify: `docs/handoff/next-level/arch_lab.py`, `decode_trial.py`, `mix_lab.py` (yardstick defaults through `yardstick_root`); create `docs/handoff/next-level/README.md` (since the release: text 1 is `data/yardstick`, text 2 is `data/`; old commands' `../data2` reads `../data`, a `../data` yardstick reads `../data/yardstick`)
- Test: `ml/tests/content/test_text_version.py`, `ml/tests/test_catalog.py`, `ml/tests/test_paths_device.py`, `ml/tests/test_scorecard.py` (CLI default)

**Interfaces:**
- Produces: `yardstick_root(root)`: `root / "yardstick"` when it holds `corpus/`, else `root` (a test root without one measures against itself, as before). Conversations blurb for text 2: **"Short chats between people and a friendly AI: questions, requests and small talk."**; text 1 keeps "People talking and answering each other."

- [ ] **Step 1: Failing tests** — `test_a_build_without_a_text_is_text_2_in_an_empty_root_and_the_held_text_after` (rewrites the text-1 test; a root that holds text 1 still rebuilds text 1 without `--text`); `test_the_conversations_card_describes_the_text_the_root_holds` (tiny text-2 root's catalog blurb is the new sentence; the `tiny_data_root` (text 1) keeps the old); `test_yardstick_root_is_the_yardstick_folder_when_there` (`yardstick_root(tmp)` is `tmp` until `tmp/yardstick/corpus` exists, then `tmp/yardstick`).
- [ ] **Step 2:** `cd ml && python -m uv run pytest tests/content/test_text_version.py tests/test_catalog.py tests/test_paths_device.py -q` — FAIL.
- [ ] **Step 3:** Implement. Any other content test that builds into an empty root and checks text-1 content passes `text=1` explicitly.
- [ ] **Step 4:** `cd ml && python -m uv run pytest -q` (whole default suite), `python -m uv run ruff check .`, `npm run check` — PASS.
- [ ] **Step 5:** Commit `feat: text 2 is the text a new build makes; the measuring stick reads data/yardstick`.

### Task 5: Helpers stop downloading rival packs

A helper never reads a rival pack: rival AIs reach it as checkpoints through the blob store (`RivalService.modelUrl`). Today every friend downloads both packs (1.5 GB, with their training work folders); pack 3 would add 0.9 GB.

**Files:**
- Modify: `apps/server/src/helpers/downloads.ts` (`HELPER_DATA_DIRS` without `"rivals"`)
- Modify: `ml/src/airace_ml/worker/fetch.py` (`SKIPPED_DIRS = ("rivals",)`: listed files there are not downloaded)
- Test: the server's helper-downloads test, `ml/tests/test_fetch_data.py`

- [ ] **Step 1: Failing tests** — server: the manifest lists no `rivals/` file and `GET /helper/data/rivals/v2/rivals.json` is 404 (or refused) for a signed-in player; ML: `test_files_of_rival_packs_in_an_older_hosts_list_are_skipped` (a manifest with a `rivals/v1/rivals.json` entry: the fetch succeeds, nothing is written under `rivals/`).
- [ ] **Step 2:** run both — FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `npm test`, `cd ml && python -m uv run pytest tests/test_fetch_data.py -q` — PASS.
- [ ] **Step 5:** Commit `fix(helper): friends no longer download the rival packs`.

### Task 6: The graphics-card work on data2 (ask the owner first: about 3½ hours)

- [ ] **Step 1:** Ask the owner when the card is free (rivals about 75 minutes, the gate about two hours, both in one go).
- [ ] **Step 2:** `cd ml && AIRACE_DATA=../data2 python -m uv run airace-ml build-rivals > runs/rivals-v3.log 2>&1` (background, Monitor heartbeat). Expected: `rival pack ready in ..\data2\rivals\v3\rivals.json`; every key carries `-r3-`; `version` is `rivals-v3`.
- [ ] **Step 3:** `AIRACE_DATA=../data2 python -m uv run airace-ml gate --out runs/gate-3 --cpu-speed > runs/gate-3.log 2>&1` (nothing else on the card). Expected: a report in `runs/gate-3/report.md`; no criterion lost against gate 2 (5 of 7; G3 and G7 measured again, reported as they are).
- [ ] **Step 4:** `AIRACE_DATA=../data2 python -m uv run airace-ml export-catalog --population runs/gate-3/runs`. Expected: the Conversations blurb is the text-2 sentence; `referenceTraits` come from gate 3.
- [ ] **Step 5:** Copy the gate report to `docs/reports/2026-11-24-gate-text2.md`; commit `docs: the gate on training text 2`.

### Task 7: The swap (text 2 into data/, text 1 into data/yardstick/)

Every step checks before it moves; any surprise stops the task.

- [ ] **Step 1: Nothing runs on the folders.** No game server holds `data/server/server.lock` or `data2/server/server.lock`; no worker or build runs.
- [ ] **Step 2: Back up the owner's worlds.** Copy `data/server` to `data/server-backup-2026-11-24` (robocopy /E); compare file counts and total size.
- [ ] **Step 3: Record what each root is.** `data_identity(data)` (text 1) and `data_identity(data2)` (text 2) into the ledger; `data/tokenizer/tok-v1/tokenizer.json` and `data2/…` are byte-identical (else stop).
- [ ] **Step 4: Preconditions.** `data/yardstick` does not exist; `data2/corpus/v2`, `data2/judge/v1`, `data2/novelty/v1`, `data2/catalog/v1`, `data2/rivals/v3/rivals.json` exist; `data/rivals/v3` does not.
- [ ] **Step 5: Move.** `data/{corpus,judge,novelty,catalog}` → `data/yardstick/`, copy `data/tokenizer` → `data/yardstick/tokenizer`; `data2/{corpus,judge,novelty,catalog}` → `data/`; `data2/rivals/v3` → `data/rivals/v3`.
- [ ] **Step 6: Check.** `data_identity(data)` equals the recorded text-2 identity and `data_identity(data/yardstick)` the text-1 one; `data/rivals/v1` and `v2` untouched; the `data-check` junctions resolve (re-make them if not).
- [ ] **Step 7: The stick still measures the same.** `cd ml && python -m uv run airace-ml scorecard --out runs/scorecard-e3-text2-r3 --baseline ../docs/reports/2026-11-scorecard-engine2.json` (data root `data/`, yardstick `data/yardstick` by default): every run and measurement reused, no "unlike" refusal, numbers as in ADR 0010 §5.4 round 3.
- [ ] **Step 8: Old worlds on a copy.** `docs/handoff/release/check_worlds.ts` (run with `npx tsx`, against a copy of `data/server` in a scratch root linked to the new `data/`): every world loads; every seated rival version's key resolves to a pack with stored replies and a model; the counts per pack are printed.
- [ ] **Step 9:** Retire the second game: remove the `text2` entry of `.claude/launch.json`; `data2/` keeps its worlds and its old pack untouched. Commit `chore: the release swap (text 2 is the main game's data)` with the script.

### Task 8: Checks on the released folders

- [ ] **Step 1:** `npm test`, `npm run check`, `cd ml && python -m uv run pytest -q`, `python -m uv run pytest -m slow -q`, `python -m uv run ruff check .` — all pass.
- [ ] **Step 2:** `npm run e2e` (solo and both two-player worlds) — 3 of 3.
- [ ] **Step 3:** A played check in the `check` game (`data-check`, never `data/server`): a new world's Lab opens with conversations 30%, the others 10%; a first AI trains; a rival with an `-r3-` key is in the world; the first AI answers its name when asked. Screenshot.

### Task 9: Records

- [ ] ADR 0010 §8 "The release" (what moved, identities, gate 3, the keys, helpers, the second game retired, creativity under the new judge).
- [ ] `docs/progress.md` (top entry), `docs/handoff/sdd/ledger-next-level.md` (plan C lines), `docs/handoff/next-steps.md` (release done; next: the game document), `docs/handoff/README.md` §4, `CLAUDE.md` commands (`balance-v4`, `rivals/v3`, `data/yardstick`, the catalog population, `--text`, the retired second game), the owner's release notes `docs/reports/2026-11-24-release-notes.md` (plain language: what changed, friends run the helper's setup once, old worlds unchanged, creativity scores of new measurements use the new judge).
- [ ] Commit `docs: the release` and push.
