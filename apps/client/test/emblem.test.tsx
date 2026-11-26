import { EMBLEM_FINISHES, EMBLEM_FRAMES, EMBLEM_PATTERNS, type Emblem } from "@airace/protocol";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Logo } from "../src/components/Logo.tsx";
import { emblemSvg, framePath } from "../src/emblem/art.ts";

const PALETTE = { primary: "#4f8cff", secondary: "#ffb020" };
const EMBLEM: Emblem = {
  frame: "corners",
  corners: 6,
  turn: 0,
  roundness: 0,
  rings: 2,
  ringWeight: 3,
  glow: 40,
  symbol: "spark",
  letters: "LL",
  symbolSize: 100,
  pattern: "rays",
  finish: "metal",
};

describe("the flat emblem", () => {
  it("draws a rival's old logo as an emblem with its symbol", () => {
    render(
      <Logo
        logo={{ shape: "hexagon", symbol: "rocket", initials: "" }}
        palette={PALETTE}
        title="Rival"
      />,
    );
    const svg = screen.getByRole("img", { name: "Rival" });
    expect(svg.getAttribute("data-symbol")).toBe("rocket");
    expect(svg.getAttribute("data-frame")).toBe("corners");
    expect(svg.getAttribute("data-shape")).toBe("hexagon");
    expect(svg.innerHTML).toContain(PALETTE.primary);
  });

  it("draws letters as text, never markup", () => {
    const { container } = render(
      <Logo
        logo={{
          shape: "circle",
          symbol: null,
          initials: "AB",
          emblem: { ...EMBLEM, symbol: null, letters: "XYZ" },
        }}
        palette={PALETTE}
      />,
    );
    expect(container.querySelector("text")?.textContent).toBe("XYZ");
    const hostile = emblemSvg({ ...EMBLEM, symbol: null, letters: "<b>" as never }, PALETTE);
    expect(hostile).not.toContain("<b>");
    expect(hostile).toContain("&lt;b&gt;");
  });

  it("closes every frame's path, at every corner count and roundness", () => {
    for (const frame of EMBLEM_FRAMES)
      for (const corners of [3, 6, 12])
        for (const roundness of [0, 60, 100])
          expect(framePath({ ...EMBLEM, frame, corners, roundness }).trim()).toMatch(/Z$/);
  });

  it("draws every pattern and finish without failing", () => {
    for (const pattern of EMBLEM_PATTERNS)
      for (const finish of EMBLEM_FINISHES) {
        const svg = emblemSvg({ ...EMBLEM, pattern, finish }, PALETTE);
        expect(svg.startsWith("<svg")).toBe(true);
        expect(svg).toContain(PALETTE.primary);
      }
  });

  it("keeps two emblems on one page from sharing their gradients", () => {
    const a = emblemSvg(EMBLEM, PALETTE, { idPrefix: "a" });
    const b = emblemSvg(EMBLEM, PALETTE, { idPrefix: "b" });
    expect(a).toContain('id="a-');
    expect(b).toContain('id="b-');
    expect(b).not.toContain('id="a-');
  });
});
