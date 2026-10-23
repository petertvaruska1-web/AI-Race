import { Hex, type Palette } from "@airace/protocol";

/** The neutral dark dashboard base that company colors sit on (C5). */
export const BASE_BG = "#0f1320";
const MIN_CONTRAST = 3;

function channel(byte: number): number {
  const s = byte / 255;
  return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
}

function luminance(hex: string): number {
  const n = Number.parseInt(hex.slice(1), 16);
  return (
    0.2126 * channel((n >> 16) & 0xff) +
    0.7152 * channel((n >> 8) & 0xff) +
    0.0722 * channel(n & 0xff)
  );
}

/** WCAG contrast ratio between two #rrggbb colors, 1 to 21. */
export function contrastRatio(a: string, b: string): number {
  const la = luminance(a);
  const lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/** Why a palette can't be used, in plain words, or null when it's fine. */
export function paletteProblem(p: Palette): string | null {
  if (!Hex.safeParse(p.primary).success || !Hex.safeParse(p.secondary).success) {
    return "Colors must be picked from the color chooser.";
  }
  if (p.primary.toLowerCase() === p.secondary.toLowerCase()) return "Pick two different colors.";
  if (contrastRatio(p.primary, BASE_BG) < MIN_CONTRAST) {
    return "Your main color is too dark to stand out on the dashboard. Pick a brighter one.";
  }
  if (contrastRatio(p.secondary, BASE_BG) < MIN_CONTRAST) {
    return "Your second color is too dark to stand out on the dashboard. Pick a brighter one.";
  }
  return null;
}

export const PALETTE_PRESETS: readonly { id: string; label: string; palette: Palette }[] = [
  { id: "ocean", label: "Ocean", palette: { primary: "#4f8cff", secondary: "#38d9c0" } },
  { id: "sunset", label: "Sunset", palette: { primary: "#ff7a45", secondary: "#ffd166" } },
  { id: "forest", label: "Forest", palette: { primary: "#4caf50", secondary: "#c5e384" } },
  { id: "royal", label: "Royal", palette: { primary: "#9b7bff", secondary: "#f5c542" } },
  { id: "ruby", label: "Ruby", palette: { primary: "#ff4d6d", secondary: "#ffb3c1" } },
  { id: "mint", label: "Mint", palette: { primary: "#3ddc97", secondary: "#a0f0ff" } },
  { id: "ember", label: "Ember", palette: { primary: "#ff9f1c", secondary: "#ff5964" } },
  { id: "sky", label: "Sky", palette: { primary: "#5cc8ff", secondary: "#ffffff" } },
  { id: "lilac", label: "Lilac", palette: { primary: "#c084fc", secondary: "#f0abfc" } },
  { id: "gold", label: "Gold", palette: { primary: "#f5b700", secondary: "#fff3b0" } },
  { id: "steel", label: "Steel", palette: { primary: "#94a3b8", secondary: "#e2e8f0" } },
  { id: "lime", label: "Lime", palette: { primary: "#a3e635", secondary: "#22d3ee" } },
];
