import { EMBLEM_FINISHES, EMBLEM_FRAMES, EMBLEM_PATTERNS, type Emblem } from "@airace/protocol";
import { render, screen } from "@testing-library/react";
import type * as THREE from "three";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Logo } from "../src/components/Logo.tsx";
import { emblemSvg, framePath } from "../src/emblem/art.ts";
import { makeBadgeStage } from "../src/emblem/badgeStage.ts";
import { EmblemBadge } from "../src/emblem/EmblemBadge.tsx";

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

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("the 3D emblem badge", () => {
  it("shows the flat emblem without WebGL", () => {
    render(
      <EmblemBadge
        logo={{ shape: "circle", symbol: "spark", initials: "", emblem: EMBLEM }}
        palette={PALETTE}
        size={120}
        label="Your logo"
      />,
    );
    expect(screen.getByRole("img", { name: "Your logo" }).tagName.toLowerCase()).toBe("svg");
  });

  it("builds a beveled badge with the emblem on its face", () => {
    const stage = makeBadgeStage(EMBLEM, PALETTE, { still: true, motion: "logo:still" });
    expect(stage.scene.getObjectByName("badge")).toBeDefined();
    expect(stage.scene.getObjectByName("badge-face")).toBeDefined();
    expect(stage.update(0, 0.016)).toBe(false);
  });

  it("changes its frame in place for another emblem", () => {
    const stage = makeBadgeStage(EMBLEM, PALETTE, { still: true, motion: "logo:still" });
    const before = (stage.scene.getObjectByName("badge-face") as unknown as { geometry: object })
      .geometry;
    stage.setLook({ ...EMBLEM, frame: "shield" }, PALETTE);
    const after = (stage.scene.getObjectByName("badge-face") as unknown as { geometry: object })
      .geometry;
    expect(after).not.toBe(before);
  });

  it("keeps the latest look on its face when drawings finish out of order, and none once gone", () => {
    // Each look is drawn into a canvas of its own; the face's drawings load in an order of
    // their own, as images do.
    const canvases: HTMLCanvasElement[] = [];
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (
      this: HTMLCanvasElement,
    ) {
      canvases.push(this);
      return { clearRect() {}, drawImage() {} } as never;
    });
    const images: { onload: (() => void) | null }[] = [];
    vi.stubGlobal(
      "Image",
      class {
        onload: (() => void) | null = null;
        src = "";
        constructor() {
          images.push(this);
        }
      },
    );
    const stage = makeBadgeStage(EMBLEM, PALETTE, { still: true, motion: "logo:still" });
    const redraw = vi.fn();
    stage.setRedraw(redraw);
    stage.setLook({ ...EMBLEM, rings: 1 }, PALETTE);
    stage.setLook({ ...EMBLEM, rings: 3 }, PALETTE);
    const face = stage.scene.getObjectByName("badge-face") as THREE.Mesh;
    const material = (face.material as THREE.MeshPhysicalMaterial[])[0];
    images[2]?.onload?.();
    expect(material?.map?.image).toBe(canvases[2]);
    images[1]?.onload?.();
    images[0]?.onload?.();
    expect(material?.map?.image).toBe(canvases[2]);
    expect(redraw).toHaveBeenCalledTimes(1);

    const gone = makeBadgeStage(EMBLEM, PALETTE, { still: true, motion: "logo:still" });
    const late = vi.fn();
    gone.setRedraw(late);
    gone.dispose();
    images[3]?.onload?.();
    expect(late).not.toHaveBeenCalled();
  });

  it("circles a spark around the badge for the orbit look", () => {
    const stage = makeBadgeStage(EMBLEM, PALETTE, { still: false, motion: "logo:orbit" });
    expect(stage.scene.getObjectByName("orbit-spark")).toBeDefined();
    expect(stage.update(0.5, 0.016)).toBe(true);
  });
});
