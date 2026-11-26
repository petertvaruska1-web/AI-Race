import type { Palette } from "@airace/protocol";
import { contrastRatio } from "@airace/sim";
import { create } from "zustand";
import { companyThemeVars, mix, NEUTRAL_PALETTE } from "./palette.ts";

// The lab glass behind every screen (design §5.3, §6). Its tint follows the company's main
// color, and whether it is light or dark follows that color too (C57), so the main color
// always reads on its own glass.

export type Glass = "light" | "dark";

/** The glass colors text sits on, for contrast checks. */
export const GLASS_BASE: Record<Glass, string> = { dark: "#0b0e15", light: "#f2f3f6" };

/** Light glass when the main color reads better on white than on black (C57). */
export function glassFor(primary: string): Glass {
  return contrastRatio(primary, "#ffffff") >= contrastRatio(primary, "#000000") ? "light" : "dark";
}

/** The color lamps and display readouts glow in: lifted toward white for a deep main color. */
export function lampColor(p: Palette): string {
  return glassFor(p.primary) === "light" ? mix("#ffffff", p.primary, 0.5) : p.primary;
}

/**
 * `color` when it reads on `glass` (contrast at least `min`), otherwise the first step toward
 * white (dark glass) or black (light glass) that does. Used for other companies' colors (A3).
 */
export function readableOn(color: string, glass: Glass, min = 3): string {
  const base = GLASS_BASE[glass];
  if (contrastRatio(color, base) >= min) return color;
  const target = glass === "dark" ? "#ffffff" : "#000000";
  for (let k = 1; k <= 10; k++) {
    const c = mix(target, color, k / 10);
    if (contrastRatio(c, base) >= min) return c;
  }
  return target;
}

/** "r g b" for CSS's rgb(var(--x) / alpha). */
function rgbOf(hex: string): string {
  const n = Number.parseInt(hex.slice(1), 16);
  return `${(n >> 16) & 255} ${(n >> 8) & 255} ${n & 255}`;
}

/** Every CSS variable the bench reads from the company's palette. */
export function benchVars(p: Palette | null): Record<string, string> {
  const palette = p ?? NEUTRAL_PALETTE;
  const lamp = lampColor(palette);
  const light = glassFor(palette.primary) === "light";
  return {
    ...companyThemeVars(palette),
    // The tint a chosen item sits in, over its own glass.
    ...(light ? { "--brand-soft": mix(palette.primary, "#ffffff", 0.16) } : {}),
    "--lamp": lamp,
    "--lamp-rgb": rgbOf(lamp),
    "--p-rgb": rgbOf(palette.primary),
    "--s-rgb": rgbOf(palette.secondary),
    "--accent": palette.primary,
  };
}

/** The palette the glass is drawn in; null before a company exists. */
export const useLook = create<{ palette: Palette | null }>(() => ({ palette: null }));

/** Themes every screen in the company's palette (C5) and picks its glass (C57). */
export function applyBenchTheme(
  p: Palette | null,
  root: HTMLElement = document.documentElement,
): void {
  for (const [name, value] of Object.entries(benchVars(p))) root.style.setProperty(name, value);
  root.dataset.glass = p ? glassFor(p.primary) : "dark";
  useLook.setState({ palette: p });
}
