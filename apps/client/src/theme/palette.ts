import type { Palette } from "@airace/protocol";
import { BASE_BG, contrastRatio } from "@airace/sim";

/** The neutral accents used before a company is founded. */
export const NEUTRAL_PALETTE: Palette = { primary: "#8b93b8", secondary: "#c2c8e0" };

/** Black or white, whichever reads better on `color`. */
export function onBrand(color: string): "#000000" | "#ffffff" {
  return contrastRatio(color, "#000000") >= contrastRatio(color, "#ffffff") ? "#000000" : "#ffffff";
}

/** `color` laid over `base` at `alpha` (0–1), as #rrggbb. */
export function mix(color: string, base: string, alpha: number): string {
  const a = Number.parseInt(color.slice(1), 16);
  const b = Number.parseInt(base.slice(1), 16);
  const channel = (shift: number) => {
    const v = Math.round(((a >> shift) & 0xff) * alpha + ((b >> shift) & 0xff) * (1 - alpha));
    return v.toString(16).padStart(2, "0");
  };
  return `#${channel(16)}${channel(8)}${channel(0)}`;
}

/** The CSS variables a palette sets. */
export function companyThemeVars(p: Palette | null): Record<string, string> {
  const palette = p ?? NEUTRAL_PALETTE;
  return {
    "--brand": palette.primary,
    "--brand-2": palette.secondary,
    "--on-brand": onBrand(palette.primary),
    "--brand-soft": mix(palette.primary, BASE_BG, 0.16),
  };
}

/** Colors every screen with the company's palette (C5); null restores the neutral look. */
export function applyCompanyTheme(
  p: Palette | null,
  root: HTMLElement = document.documentElement,
): void {
  for (const [name, value] of Object.entries(companyThemeVars(p)))
    root.style.setProperty(name, value);
}
