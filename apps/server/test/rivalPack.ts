import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GROUP_IDS, type RivalPack } from "@airace/protocol";

const FIXTURE = new URL("../../../packages/protocol/fixtures/rivals/rivals.json", import.meta.url);

/**
 * A rival pack in a temp folder: the protocol fixture's index (on the sample
 * catalog's data identity unless told otherwise), two stored replies per
 * group for every version, and a small stand-in tar per version.
 */
export function writeFixturePack(identity = "sample-0001"): { dir: string; pack: RivalPack } {
  const dir = mkdtempSync(join(tmpdir(), "airace-rivals-"));
  const pack = JSON.parse(readFileSync(FIXTURE, "utf8")) as RivalPack;
  pack.dataIdentity = identity;
  mkdirSync(join(dir, "replies"));
  mkdirSync(join(dir, "models"));
  for (const lineage of pack.lineages) {
    for (const v of lineage.versions) {
      const replies = GROUP_IDS.flatMap((group) =>
        [0, 1].map((k) => ({
          group,
          questionId: `${group}-q${k}`,
          question: `What would ${group} ask (${k})?`,
          reply: k === 0 ? `${v.key} answers ${group}.` : "<b>x</b> <|end|>",
          score: k === 0 ? 0.9 : 0.1,
        })),
      );
      writeFileSync(join(dir, "replies", `${v.key}.json`), JSON.stringify(replies));
      writeFileSync(join(dir, "models", `${v.key}.tar`), `fake-rival-${v.key}`);
    }
  }
  writeFileSync(join(dir, "rivals.json"), JSON.stringify(pack));
  return { dir, pack };
}
