# ADR 0004: Task 15 source licences and simplicity thresholds

- **Date:** 2026-10-21
- **Status:** Accepted (engineering decision; no creative impact)
- **Context:** The local GPU session (handoff README §5 step 1) re-verified every source licence and calibrated the simplicity thresholds on real Hub data before the full build.

## Decisions
1. **Every licence was re-verified on the Hub (2026-10-21).** Each `SourceSpec.license` matches the dataset card's `license` field; none is gated. The commit read is pinned per build and recorded in the manifest (ADR 0003 rule 4).
2. **Simple Wikipedia is removed.**
   - Its licence is CC-BY-SA-3.0 (with GFDL), a share-alike licence. The plan's permissive list is public domain, CC0, CC-BY, ODC-By, MIT, Apache-2.0, and CDLA-Sharing used for training only. CLAUDE.md requires permissive sources.
   - CDLA-Sharing is on the list because that licence says itself that trained models are not bound by its sharing terms. CC-BY-SA says no such thing, and players see the model's text, so the exception does not carry over.
   - As the plan prescribes, its share of the Reference Facts dataset (0.40) went to the other component, so `facts` is now all `fact_prose` from the knowledge base.
   - Its adapter, helpers, fixture and tests were removed with it.
   - **Cost if wrong:** Reference Facts holds no real encyclopedia prose, only varied knowledge-base prose. A permissive real-text source can be added later as a new component without touching the tokenizer.
3. **Simplicity thresholds, calibrated (F15).**
   - Method: the full-scale simple vocabulary (the 6,000 most frequent words of 200,000 TinyStories rows, as the build makes it) and the first 2,000 prepared documents of each source. Script and numbers: `docs/handoff/calibration/`.
   - Measured keep-rates at the old thresholds: FineWeb 98% at 0.35, FineWeb-Edu 98% at 0.35, Gutenberg 99% at 0.30. The filters did almost nothing.

     | Source | q10 | q30 | q50 | q70 | New threshold | Keeps |
     |---|---|---|---|---|---|---|
     | fineweb (web) | 0.48 | 0.60 | 0.68 | 0.75 | **0.55** | 80% |
     | fineweb_edu (educational) | 0.47 | 0.57 | 0.64 | 0.70 | **0.55** | 76% |
     | gutenberg (books) | 0.49 | 0.72 | 0.83 | 0.88 | **0.50** | 89% |
     | cosmo_khan (unfiltered, for comparison) | 0.57 | 0.68 | 0.73 | 0.78 | none | 100% |

   - Rationale: drop roughly the hardest fifth of web and educational text, which a few-million-parameter model learns least from. That still leaves both sources varied and noisy, so preparation choices keep real consequences. Gutenberg's low tail is mostly contents pages, indexes and archaic verse.
   - The handoff's estimate ("0.6 keeps about 90%") was wrong on real data: 0.60 keeps 71% of FineWeb.
   - Each source has far more rows than its quota needs, so the stricter filters cost reading time, not supply.
   - **Cost if wrong:** web and educational text skews simpler than intended. The thresholds are a single number in `RECIPES`.
4. **Measured bytes per token for four components (F16).**
   - On the first full build, Reference Facts reached 87% of plan and Conversations 89%.
   - Real bytes per token with tok-v1: fact_prose 3.70, soda 3.67, fact_chat 3.57, instructions 3.34. The shared prose estimate is 3.2.
   - Those four now have their own constants: 3.7, 3.7, 3.6 and 3.3. The two datasets were rebuilt with `--datasets`, and reached 100% and 99%.
   - Every other component landed within 8% of plan and keeps the shared estimates, so no other dataset changed. FineWeb overshoots by 8% at 2.97 bytes per token.
5. **The TinyStories repair also covers ellipses and Latin-1 letters.**
   - The full build left 61 creative documents with mojibake.
   - The ellipsis (`â€¦`) was caught by the closing-quote fallback and left `"¦`. Now it becomes `...`.
   - Accented letters and Spanish marks (`cafÃ©`, `Â¡`, `Â«`) were not repaired. Now any `Â`/`Ã` pair that is valid misdecoded UTF-8 gets its letter back.
   - Creative was rebuilt and now has 0 such documents.
6. **Mojibake that stays, deliberately:**
   - Web Crawl: 93 of 33,663 documents outside the injected noise. This is natural crawl dirt, which cleaning is there to deal with.
   - Educational: 23 documents, broken math symbols inside Cosmopedia's own text.
   - Conversations: 2 Soda dialogues.
   - Books, Code, Reasoning and Reference Facts: none.
7. **Sources are pinned in code (F29).**
   - `SourceSpec.revision` holds the commit each source was read at. Tests of the head lookup use an unpinned copy.
   - A fresh simple vocabulary built with the final code equals the one in the corpus, so the corpus is reproducible from the committed code and the pinned commits.
8. **Notable but accepted:**
   - The templated datasets are heavily near-duplicated: Reference Facts 80%, Conversations 37%, Code 18% of documents are extra members of a near-duplicate cluster. A finite knowledge base and templates repeat. "Remove duplicates" therefore shrinks them a lot, which is a real preparation consequence, as the spec intends.
   - A few Gutenberg passages keep image captions such as `16-134.jpg (213K)`.
9. **A test fix found along the way.** The tiny-build duplicate check used `difflib.SequenceMatcher` with its default autojunk. On texts over 200 characters, autojunk ignores common letters, so a copy with one typo scored 0.86 instead of 0.995. The check now turns autojunk off.
