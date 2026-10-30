import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { RivalPack, RivalReplies } from "../src/index.ts";

const fixture = () =>
  JSON.parse(readFileSync(new URL("../fixtures/rivals/rivals.json", import.meta.url), "utf8"));

describe("the rival pack", () => {
  it("parses the fixture", () => {
    const pack = RivalPack.parse(fixture());
    expect(pack.lineages).toHaveLength(8);
  });

  it("needs exactly two lineages of each style", () => {
    const p = fixture();
    p.lineages[2].style = "premium";
    expect(RivalPack.safeParse(p).success).toBe(false);
  });

  it("needs eight lineages", () => {
    const p = fixture();
    p.lineages.pop();
    expect(RivalPack.safeParse(p).success).toBe(false);
  });

  it("refuses a repeated version key", () => {
    const p = fixture();
    p.lineages[1].versions[0].key = p.lineages[0].versions[0].key;
    expect(RivalPack.safeParse(p).success).toBe(false);
  });

  it("checks replies", () => {
    const ok = [
      { group: "students", questionId: "q", question: "Q?", reply: "<b>hi</b>", score: 1 },
    ];
    expect(RivalReplies.parse(ok)[0]?.reply).toBe("<b>hi</b>");
    expect(RivalReplies.safeParse([{ ...ok[0], score: 2 }]).success).toBe(false);
  });
});
