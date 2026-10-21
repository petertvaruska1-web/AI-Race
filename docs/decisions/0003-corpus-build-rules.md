# ADR 0003: corpus build rules from the final review

- **Date:** 2026-10-21
- **Status:** Accepted (engineering decision; no creative impact)
- **Context:** `ml/src/airace_content` (Tasks 13–14). The owner's Task 15 build streams about 137M tokens from the Hugging Face hub for about an hour, and its tokenizer is frozen as tok-v1 for good. Each rule is a ledger ruling ("FINAL REVIEW: Ruling Fxx").

## Decisions
1. **The tokenizer and shared files record their scale (F3).** A full build refuses a tokenizer, or shared vocab files, that a tiny build left behind. A tiny build refuses a root that holds a full one. A tiny check build needs its own `--out`.
2. **Hub reads retry and resume exactly (F11).**
   - Transient network and server errors are retried with backoff for about 30 minutes, with one stderr line per retry. The read resumes at the exact row, so the same seed still gives the same corpus.
   - Before anything is read, every source the build needs is opened and checked.
   - A final network failure is one plain line telling the owner to run the same command again.
3. **Progress on stderr (F17).** One line when each source opens, then lines every N rows, then lines for dedup, the tokenizer and the writes.
4. **Sources are pinned (F29).** Each source is read at one Hub commit, the one in `SourceSpec.revision`, or the head looked up once. The manifest records it.
5. **Quota sizing is a deterministic byte estimate (F16, F30).**
   - Each component stops when a running sum of estimated tokens reaches its quota.
   - Bytes per token: about 3.2 for prose, 2.7 for code and patterns, 3.6 for creative.
   - The frozen tokenizer is never used for sizing, so a second build gives the same corpus.
   - The summary warns when a component ends below 90% of plan. If that fires for the templated generators on the first real build, add their own constant.
   - This supersedes the Task 14 ruling (3.6 bytes per token, a 200-doc sample, and tokenizer counting after the first build).
6. **Source repairs, applied to source text only, never to injected noise:**
   - Real Gutenberg rows (sedthh) get their plain-text layout back before chunking, so passages keep real paragraphs (F12).
   - TinyStories' cp1252 mojibake becomes ASCII quotes and dashes, matching the clean stories (F14).
7. **Noise and quality:**
   - Copies of code, a structured dataset, are exact, so a program stays runnable (F26).
   - `quality_score` has a page-furniture factor (F27): lines with a `|` separator or a furniture phrase, matched as whole words in lines of at most 20 words. Each such line costs a third of the score.
     - Accepted consequences: a markdown table of 3 or more rows scores 0 at every cleaning level, and a short prose line with "cookies" or "read more" counts.
   - Calendar words and false friends are no longer topic keywords (F52).
8. **Kept as it was (refuted in review): the ASCII word rule of `textproc.words` (F31).** The gate deliberately reads model text as Unicode letter runs, and changing either would hide the mojibake signal G7 measures.

## Pending on the owner's machine
- **The simplicity thresholds (F15).** They keep about 97–100% of real documents. They are calibrated on real data with the real simple_vocab before the full build; see the handoff README §5.
