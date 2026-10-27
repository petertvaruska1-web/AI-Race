import { z } from "zod";
import { DATASET_IDS, type DatasetId } from "./world.ts";

// The server's facts about the training data, exported by the ML project
// (ADR 0008 item 3). Players see titles, sizes, samples and what preparation
// removes; never the data itself.

export const CLEANING_LEVELS = ["light", "standard", "thorough"] as const;

/** One way to prepare a dataset: `${cleaning}|${dedup ? 1 : 0}|${factCheck ? 1 : 0}`. */
export const PREP_KEYS = CLEANING_LEVELS.flatMap((c) =>
  ["0", "1"].flatMap((d) => ["0", "1"].map((f) => `${c}|${d}|${f}`)),
) as [string, ...string[]];
export type PrepKey = (typeof PREP_KEYS)[number];

export function prepKey(
  cleaning: (typeof CLEANING_LEVELS)[number],
  dedup: boolean,
  factCheck: boolean,
): PrepKey {
  return `${cleaning}|${dedup ? 1 : 0}|${factCheck ? 1 : 0}`;
}

/** Player-facing dataset names (D2 is open; these are the temporary assumption). */
export const DATASET_TITLES: Record<DatasetId, string> = {
  web: "Web Crawl",
  books: "Books",
  educational: "Educational",
  conversations: "Conversations",
  code: "Code",
  reasoning: "Reasoning",
  facts: "Reference Facts",
  creative: "Creative Writing",
};

const Count = z.number().int().min(0);
const Sample = z.string().max(600);

export const CatalogDataset = z.object({
  id: z.enum(DATASET_IDS),
  title: z.string().min(1),
  blurb: z.string().min(1),
  tokens: Count,
  docs: Count,
  pools: z.record(z.enum(PREP_KEYS), z.object({ docs: Count, tokens: Count })),
  samples: z.array(Sample).max(5),
  removedExamples: z.object({
    cleaning: z.record(z.enum(CLEANING_LEVELS), z.array(Sample)),
    duplicates: z.array(Sample),
    factCheck: z.array(Sample),
  }),
});
export type CatalogDataset = z.infer<typeof CatalogDataset>;

export const Catalog = z
  .object({
    version: z.literal("catalog-v1"),
    dataIdentity: z.string().min(1),
    tokenizer: z.literal("tok-v1"),
    suite: z.literal("bench-v1"),
    datasets: z.array(CatalogDataset),
    referenceTraits: z.array(z.record(z.string(), z.number())),
  })
  .refine(
    (c) => {
      const ids = c.datasets.map((d) => d.id);
      return ids.length === DATASET_IDS.length && DATASET_IDS.every((id) => ids.includes(id));
    },
    { message: "the catalog must describe each dataset exactly once", path: ["datasets"] },
  );
export type Catalog = z.infer<typeof Catalog>;

/** What players may read: the catalog without the reference models' personality data. */
export type PublicCatalog = Omit<Catalog, "referenceTraits">;
