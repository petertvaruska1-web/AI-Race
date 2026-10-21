# Data sources (corpus v1, tokenizer tok-v1)

Built 2026-10-21 on the owner's PC with `airace-content build --scale full` (Task 15). The corpus itself lives under `data/` and is never committed. Its `manifest.json` records the same sources, licences and commits.

Every licence was checked against the Hugging Face dataset card on 2026-10-21. Every source is read at the pinned commit below (`SourceSpec.revision`), so a rebuild reads the same rows. Decisions: ADR 0004.

## Sources

| Source ID | Hugging Face dataset (config, split) | Licence | Commit read | Rows read | Documents kept | Tokens | Feeds game dataset |
|---|---|---|---|---|---|---|---|
| fineweb | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (sample-10BT, train) | ODC-By | `9bb295dd` | 38,416 | 30,970 | 32.3M | Web Crawl |
| gutenberg | [sedthh/gutenberg_english](https://huggingface.co/datasets/sedthh/gutenberg_english) (train) | MIT | `28973b04` | 1,745 books | 24,382 passages | 14.8M | Books |
| cosmo_khan | [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (khanacademy, train) | Apache-2.0 | `0ae6ec63` | 4,657 | 4,657 | 5.3M | Educational |
| cosmo_wikihow | [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (wikihow, train) | Apache-2.0 | `0ae6ec63` | 3,508 | 3,508 | 4.9M | Educational |
| cosmo_openstax | [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (openstax, train) | Apache-2.0 | `0ae6ec63` | 3,334 | 3,334 | 4.0M | Educational |
| fineweb_edu | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT, train) | ODC-By | `87f09149` | 5,328 | 4,047 | 6.2M | Educational |
| soda | [allenai/soda](https://huggingface.co/datasets/allenai/soda) (train) | CC-BY-4.0 | `fdc848ab` | 67,002 | 66,970 | 12.5M | Conversations |
| everyday_conv | [HuggingFaceTB/everyday-conversations-llama3.1-2k](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) (train_sft) | Apache-2.0 | `14f54321` | 2,260 (all) | 2,260 | 0.5M | Conversations |
| mbpp | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) (full, train) | CC-BY-4.0 | `4bb6404f` | 374 (all) | 374 | 0.04M | Code |
| gsm8k | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (main, train) | MIT | `740312ad` | 6,213 | 6,213 | 1.0M | Reasoning |
| tinystories | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (train) | CDLA-Sharing-1.0 (training use) | `f54c09fd` | 68,132 | 68,120 | 17.3M | Creative Writing |
| cosmo_stories | [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (stories, train) | Apache-2.0 | `0ae6ec63` | 3,975 | 3,975 | 3.2M | Creative Writing |

- **Simplicity filters (ADR 0004):** fineweb ≥ 0.55, fineweb_edu ≥ 0.55 (plus `int_score ≥ 3`), gutenberg ≥ 0.50.
- **Gutenberg:** only children's books (subjects with Juvenile, Fairy tales, Fables or Children), in passages of at most 400 words.
- **The simple vocabulary:** the 6,000 most frequent words of the first 200,000 TinyStories rows.
- **Removed (ADR 0004):** Simple Wikipedia (`wikimedia/wikipedia`, 20231101.simple), which is CC-BY-SA-3.0 and therefore not permissive.

## Procedural generators (our own code, no external licence)

| Generator | Documents | Tokens | Feeds game dataset |
|---|---|---|---|
| fact_prose (knowledge-base facts as varied prose) | 215,144 | 12.0M | Reference Facts (all of it) |
| fact_chat (knowledge-base facts as chats) | 102,017 | 3.4M | Conversations |
| instructions (instruction-following chats) | 64,502 | 3.4M | Conversations |
| code (MiniPy programs with results) | 209,409 | 10.2M | Code |
| reasoning (logic, arithmetic, counting) | 27,876 | 5.5M | Reasoning |
| patterns (sequences) | 38,049 | 3.1M | Reasoning |

## Game datasets

| Dataset | Documents | Tokens | Of target | Deliberate noise (documents) |
|---|---|---|---|---|
| Web Crawl (`web`) | 33,663 | 32.3M | 108% | typo 5,049, spam 2,020, boilerplate 3,366, garbled 1,683, duplicate 2,693, false fact 2,020 |
| Books (`books`) | 24,880 | 14.8M | 99% | typo 498, garbled 249, duplicate 498 |
| Educational (`educational`) | 15,863 | 20.4M | 102% | duplicate 317, false fact 159 |
| Conversations (`conversations`) | 235,749 | 19.8M | 99% | typo 7,072 |
| Code (`code`) | 216,271 | 10.2M | 102% | duplicate 6,488 |
| Reasoning (`reasoning`) | 72,138 | 9.7M | 97% | none |
| Reference Facts (`facts`) | 219,535 | 12.0M | 100% | duplicate 4,391 |
| Creative Writing (`creative`) | 74,325 | 20.5M | 103% | typo 1,487, duplicate 2,230 |
| **Total** | **892,424** | **139.6M** | | |

About 1% of each dataset is held out for evaluation. The check numbers (quality histograms, cleaning keep-rates, mojibake counts, reading level) are in `docs/handoff/calibration/corpus_checks.json`.

## Attribution
- FineWeb and FineWeb-Edu: Hugging Face, ODC-By 1.0, built from Common Crawl.
- Cosmopedia: Hugging Face, Apache-2.0.
- SODA: Allen Institute for AI, CC-BY-4.0.
- Everyday Conversations: Hugging Face, Apache-2.0.
- MBPP: Google Research, CC-BY-4.0.
- GSM8K: OpenAI, MIT.
- TinyStories: Eldan & Li, CDLA-Sharing-1.0.
- Project Gutenberg texts: public domain, via sedthh/gutenberg_english (MIT).
