import { contrastRatio, PALETTE_PRESETS } from "@airace/sim";
import { describe, expect, it } from "vitest";
import {
  applyCompanyTheme,
  companyThemeVars,
  mix,
  NEUTRAL_PALETTE,
  onBrand,
} from "../src/theme/palette.ts";

describe("company theme", () => {
  it("puts readable text on every preset's main color", () => {
    for (const { id, palette } of PALETTE_PRESETS) {
      expect(contrastRatio(onBrand(palette.primary), palette.primary), id).toBeGreaterThanOrEqual(
        4.5,
      );
    }
  });

  it("sets the brand variables, and the neutral ones without a company", () => {
    const root = document.createElement("div");
    applyCompanyTheme({ primary: "#4f8cff", secondary: "#ffb020" }, root);
    expect(root.style.getPropertyValue("--brand")).toBe("#4f8cff");
    expect(root.style.getPropertyValue("--brand-2")).toBe("#ffb020");
    expect(root.style.getPropertyValue("--on-brand")).toBe("#000000");
    expect(root.style.getPropertyValue("--brand-soft")).toBe(mix("#4f8cff", "#0f1320", 0.16));
    applyCompanyTheme(null, root);
    expect(root.style.getPropertyValue("--brand")).toBe(NEUTRAL_PALETTE.primary);
  });

  it("blends the soft tint over the dark base", () => {
    expect(mix("#ffffff", "#000000", 0.5)).toBe("#808080");
    expect(mix("#4f8cff", "#0f1320", 0)).toBe("#0f1320");
    expect(companyThemeVars({ primary: "#101010", secondary: "#ffffff" })["--on-brand"]).toBe(
      "#ffffff",
    );
  });
});
