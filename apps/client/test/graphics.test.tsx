import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { GlassBackground } from "../src/glass/GlassBackground.tsx";
import { createFrameMeter, due, framePolicy } from "../src/graphics/frames.ts";
import { useGraphics } from "../src/store/graphics.ts";
import { useLook } from "../src/theme/glass.ts";

beforeEach(() => {
  localStorage.clear();
  useGraphics.setState({ lighter: false, auto: false });
});

describe("how often the glass and the 3D draw", () => {
  it("stops drawing when hidden, stills for reduced motion and lighter graphics", () => {
    expect(
      framePolicy({ reducedMotion: false, lighter: false, hidden: true, interacting: true }),
    ).toEqual({ fps: 0, still: true });
    expect(
      framePolicy({ reducedMotion: true, lighter: false, hidden: false, interacting: false }).still,
    ).toBe(true);
    expect(
      framePolicy({ reducedMotion: false, lighter: true, hidden: false, interacting: false }).still,
    ).toBe(true);
    expect(
      framePolicy({ reducedMotion: false, lighter: false, hidden: false, interacting: false }).fps,
    ).toBe(30);
    expect(
      framePolicy({ reducedMotion: false, lighter: false, hidden: false, interacting: true }).fps,
    ).toBe(60);
  });

  it("is due once a frame's time has passed", () => {
    expect(due(1033, 1000, 30)).toBe(true);
    expect(due(1010, 1000, 30)).toBe(false);
    expect(due(5000, 0, 0)).toBe(false);
  });

  it("asks for lighter graphics once frames stay slow", () => {
    const meter = createFrameMeter();
    const answers = Array.from({ length: 120 }, () => meter.record(55));
    expect(answers.slice(0, 89).some(Boolean)).toBe(false);
    expect(answers.filter(Boolean)).toHaveLength(1);
    const fine = createFrameMeter();
    expect(Array.from({ length: 120 }, () => fine.record(16)).some(Boolean)).toBe(false);
  });

  it("remembers lighter graphics", () => {
    useGraphics.getState().setLighter(true);
    expect(localStorage.getItem("airace.lighterGraphics")).toBe("1");
    expect(useGraphics.getState().auto).toBe(false);
    useGraphics.getState().setLighter(true, true);
    expect(useGraphics.getState().auto).toBe(true);
  });
});

describe("the glass background", () => {
  it("shows the CSS glass in the company's glass when WebGL is missing", () => {
    useLook.setState({ palette: { primary: "#7f1d3a", secondary: "#f59e0b" } });
    const { container } = render(<GlassBackground />);
    const fallback = container.querySelector("[data-glass-fallback]");
    expect(fallback?.getAttribute("data-glass")).toBe("light");
    expect(container.querySelector("canvas")).toBeNull();
    expect(container.firstElementChild?.getAttribute("aria-hidden")).toBe("true");
  });

  it("uses dark neutral glass before a company exists", () => {
    useLook.setState({ palette: null });
    const { container } = render(<GlassBackground />);
    expect(container.querySelector("[data-glass-fallback]")?.getAttribute("data-glass")).toBe(
      "dark",
    );
  });
});
