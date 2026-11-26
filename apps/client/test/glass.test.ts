import { contrastRatio, PALETTE_PRESETS } from "@airace/sim";
import { describe, expect, it } from "vitest";
import {
  applyBenchTheme,
  benchVars,
  GLASS_BASE,
  glassFor,
  glassPower,
  lampColor,
  POWER_UP_MS,
  powerUpGlass,
  readableOn,
  useLook,
} from "../src/theme/glass.ts";
import { mix } from "../src/theme/palette.ts";

describe("the glass follows the main color (C57)", () => {
  it("gives deep main colors light glass and bright ones dark glass", () => {
    expect(glassFor("#7f1d3a")).toBe("light");
    expect(glassFor("#1e3a8a")).toBe("light");
    expect(glassFor("#3b82f6")).toBe("dark");
    for (const { palette } of PALETTE_PRESETS) expect(glassFor(palette.primary)).toBe("dark");
  });

  it("lifts a deep lamp color so it glows in a dark display", () => {
    expect(
      contrastRatio(lampColor({ primary: "#7f1d3a", secondary: "#f59e0b" }), "#05070a"),
    ).toBeGreaterThanOrEqual(4.5);
    expect(lampColor({ primary: "#3b82f6", secondary: "#2dd4bf" })).toBe("#3b82f6");
  });

  it("makes any color readable on either glass and leaves readable ones alone", () => {
    for (let r = 0; r < 256; r += 32)
      for (let g = 0; g < 256; g += 32)
        for (let b = 0; b < 256; b += 32) {
          const c = `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
          for (const glass of ["dark", "light"] as const)
            expect(contrastRatio(readableOn(c, glass), GLASS_BASE[glass])).toBeGreaterThanOrEqual(
              3,
            );
        }
    expect(readableOn("#3b82f6", "dark")).toBe("#3b82f6");
  });

  it("softens the selected tint toward the company's own glass", () => {
    expect(benchVars({ primary: "#7f1d3a", secondary: "#f59e0b" })["--brand-soft"]).toBe(
      mix("#7f1d3a", "#ffffff", 0.16),
    );
    expect(benchVars({ primary: "#3b82f6", secondary: "#2dd4bf" })["--brand-soft"]).toBe(
      mix("#3b82f6", "#0f1320", 0.16),
    );
  });

  it("marks the root with its glass and sets the lamp", () => {
    const root = document.createElement("div");
    const garnet = { primary: "#7f1d3a", secondary: "#f59e0b" };
    applyBenchTheme(garnet, root);
    expect(root.dataset.glass).toBe("light");
    expect(root.style.getPropertyValue("--lamp")).toBe(lampColor(garnet));
    expect(useLook.getState().palette).toEqual(garnet);
    applyBenchTheme(null, root);
    expect(root.dataset.glass).toBe("dark");
    expect(useLook.getState().palette).toBeNull();
  });
});

describe("the lab powering up when a company is founded (C61)", () => {
  it("brightens the aurora for two and a half seconds, then settles", () => {
    expect(POWER_UP_MS).toBe(2500);
    expect(glassPower(null)).toBe(1);
    expect(glassPower(0)).toBe(1);
    expect(glassPower(400)).toBeCloseTo(1.35, 2);
    const later = glassPower(1500);
    expect(later).toBeGreaterThan(1);
    expect(later).toBeLessThan(1.35);
    expect(glassPower(POWER_UP_MS)).toBe(1);
    expect(glassPower(10_000)).toBe(1);
  });

  it("counts each power-up so the glass starts its pulse", () => {
    const before = useLook.getState().power;
    powerUpGlass();
    expect(useLook.getState().power).toBe(before + 1);
  });
});
