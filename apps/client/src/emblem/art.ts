import type { Emblem, LogoSymbol, Palette } from "@airace/protocol";
import { mix } from "../theme/palette.ts";

// A company's emblem as SVG (design §9.3, C58), in a 100 × 100 box. One drawing serves the flat
// emblem everywhere and the face of the 3D badge. Letters are written as escaped text.

const TAU = Math.PI * 2;
const C = 50;
const R = 44;

const f2 = (n: number) => n.toFixed(2);

/** A regular polygon's corners from the top, turned by `turn` degrees. */
function corners(n: number, turn: number, r = R): [number, number][] {
  return Array.from({ length: n }, (_, i) => {
    const a = -Math.PI / 2 + (i / n) * TAU + (turn * Math.PI) / 180;
    return [C + Math.cos(a) * r, C + Math.sin(a) * r];
  });
}

/** A closed path through `pts`, each corner rounded by `roundness` (0–100). */
function rounded(pts: [number, number][], roundness: number): string {
  if (roundness <= 0) return `M${pts.map(([x, y]) => `${f2(x)} ${f2(y)}`).join(" L")} Z`;
  const n = pts.length;
  const at = (a: [number, number], b: [number, number], d: number): [number, number] => {
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const k = len === 0 ? 0 : d / len;
    return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
  };
  const side = Math.hypot(
    (pts[1]?.[0] ?? 0) - (pts[0]?.[0] ?? 0),
    (pts[1]?.[1] ?? 0) - (pts[0]?.[1] ?? 0),
  );
  const cut = (side / 2) * 0.92 * (roundness / 100);
  let d = "";
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n] as [number, number];
    const here = pts[i] as [number, number];
    const next = pts[(i + 1) % n] as [number, number];
    const from = at(here, prev, cut);
    const to = at(here, next, cut);
    d += `${i === 0 ? "M" : "L"}${f2(from[0])} ${f2(from[1])} Q${f2(here[0])} ${f2(here[1])} ${f2(to[0])} ${f2(to[1])} `;
  }
  return `${d}Z`;
}

/** The emblem's outline: a closed path in the 100 × 100 box. */
export function framePath(e: Emblem): string {
  switch (e.frame) {
    case "circle":
      return `M${C - R} ${C} A${R} ${R} 0 1 0 ${C + R} ${C} A${R} ${R} 0 1 0 ${C - R} ${C} Z`;
    case "shield":
      return "M50 5 L88 17 V46 C88 72 70 88 50 96 C30 88 12 72 12 46 V17 Z";
    case "diamond":
      return rounded(corners(4, e.turn, 46), e.roundness);
    case "rosette": {
      const steps = 180;
      const turn = (e.turn * Math.PI) / 180;
      const pts = Array.from({ length: steps }, (_, i) => {
        const a = (i / steps) * TAU;
        const r = R - 4 + 4 * Math.cos(e.corners * (a - turn));
        return `${f2(C + Math.cos(a) * r)} ${f2(C + Math.sin(a) * r)}`;
      });
      return `M${pts.join(" L")} Z`;
    }
    default:
      return rounded(corners(e.corners, e.turn), e.roundness);
  }
}

/** The sixteen symbols: `S` is filled, `L` is a line, `B` is cut out in the face color. */
const SYMBOLS: Record<LogoSymbol, string> = {
  spark: '<path data-k="S" d="M50 22 L56 44 L78 50 L56 56 L50 78 L44 56 L22 50 L44 44 Z"/>',
  star: '<path data-k="S" d="M50 22 L58 42 L80 42 L62 55 L69 77 L50 63 L31 77 L38 55 L20 42 L42 42 Z"/>',
  bolt: '<path data-k="S" d="M56 20 L32 54 H48 L42 80 L68 44 H52 Z"/>',
  moon: '<path data-k="S" d="M60 24 A28 28 0 1 0 76 64 A22 22 0 1 1 60 24 Z"/>',
  sun: '<circle data-k="S" cx="50" cy="50" r="13"/><path data-k="L" d="M50 22 V30 M50 70 V78 M22 50 H30 M70 50 H78 M30 30 L36 36 M64 64 L70 70 M30 70 L36 64 M64 36 L70 30"/>',
  leaf: '<path data-k="S" d="M28 72 C28 40 48 26 74 26 C74 54 58 72 28 72 Z"/><path data-k="B" d="M32 68 L60 40"/>',
  wave: '<path data-k="L" d="M20 46 C30 32 40 32 50 46 S70 60 80 46 M20 62 C30 48 40 48 50 62 S70 76 80 62"/>',
  eye: '<path data-k="S" d="M18 50 C32 30 68 30 82 50 C68 70 32 70 18 50 Z"/><circle data-k="F" cx="50" cy="50" r="10"/>',
  flame:
    '<path data-k="S" d="M50 18 C62 34 72 46 72 60 A22 22 0 0 1 28 60 C28 48 38 42 42 30 C46 40 52 42 50 18 Z"/>',
  rocket:
    '<path data-k="S" d="M50 18 C62 28 66 44 62 62 H38 C34 44 38 28 50 18 Z"/><path data-k="S" d="M38 54 L26 70 L40 66 Z M62 54 L74 70 L60 66 Z"/><circle data-k="F" cx="50" cy="40" r="6"/>',
  brain:
    '<path data-k="S" d="M50 28 C38 22 26 30 28 42 C20 48 24 62 34 64 C36 74 48 76 50 68 C52 76 64 74 66 64 C76 62 80 48 72 42 C74 30 62 22 50 28 Z"/><path data-k="B" d="M50 30 V68"/>',
  cube: '<polygon data-k="S" points="50,20 76,34 76,64 50,78 24,64 24,34"/><path data-k="B" d="M24 34 L50 48 L76 34 M50 48 V78"/>',
  atom: '<g data-k="A"><ellipse cx="50" cy="50" rx="30" ry="11"/><ellipse cx="50" cy="50" rx="30" ry="11" transform="rotate(60 50 50)"/><ellipse cx="50" cy="50" rx="30" ry="11" transform="rotate(120 50 50)"/></g><circle data-k="S" cx="50" cy="50" r="5"/>',
  anchor:
    '<path data-k="L" d="M50 34 V74 M38 44 H62 M30 60 C32 70 42 74 50 74 C58 74 68 70 70 60 M50 22 A6 6 0 1 1 49.9 22"/>',
  crown: '<path data-k="S" d="M24 68 L28 34 L40 50 L50 28 L60 50 L72 34 L76 68 Z"/>',
  feather:
    '<path data-k="S" d="M72 22 C46 26 30 46 30 72 L36 66 C50 64 66 50 72 22 Z"/><path data-k="B" d="M30 74 L58 40"/>',
};

/** Letters as SVG text, escaped. */
function escapeText(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function pattern(e: Emblem, color: string): string {
  const s = `stroke="${color}" stroke-opacity="0.35"`;
  switch (e.pattern) {
    case "rays":
      return Array.from({ length: 24 }, (_, i) => {
        const a = (i / 24) * TAU;
        return `<line x1="50" y1="50" x2="${f2(50 + Math.cos(a) * 60)}" y2="${f2(50 + Math.sin(a) * 60)}" ${s} stroke-width="1.6"/>`;
      }).join("");
    case "grid": {
      let o = "";
      for (let i = 6; i < 100; i += 8)
        o += `<line x1="${i}" y1="0" x2="${i}" y2="100" ${s} stroke-width="0.8"/><line x1="0" y1="${i}" x2="100" y2="${i}" ${s} stroke-width="0.8"/>`;
      return o;
    }
    case "dots": {
      let o = "";
      for (let ring = 1; ring < 4; ring++) {
        const n = ring * 10;
        for (let i = 0; i < n; i++) {
          const a = (i / n) * TAU;
          o += `<circle cx="${f2(50 + Math.cos(a) * ring * 12)}" cy="${f2(50 + Math.sin(a) * ring * 12)}" r="1.4" fill="${color}" fill-opacity="0.45"/>`;
        }
      }
      return o;
    }
    case "waves": {
      let o = "";
      for (let k = 1; k < 5; k++) {
        let d = "";
        for (let i = 0; i <= 120; i++) {
          const a = (i / 120) * TAU;
          const r = k * 10 + 1.6 * Math.sin(a * 12);
          d += `${i ? "L" : "M"}${f2(50 + Math.cos(a) * r)} ${f2(50 + Math.sin(a) * r)}`;
        }
        o += `<path d="${d}Z" fill="none" ${s} stroke-width="1.2"/>`;
      }
      return o;
    }
    default:
      return "";
  }
}

/** The inner drawing of an emblem (everything inside its `<svg>`). */
export function emblemInner(e: Emblem, p: Palette, o: { idPrefix?: string } = {}): string {
  const id = o.idPrefix ?? "em";
  const fp = framePath(e);
  const dark = mix("#000000", p.primary, 0.45);
  const light = mix("#ffffff", p.primary, 0.35);
  const silver = `url(#${id}-silver)`;
  const face =
    e.finish === "light" ? "#07090c" : e.finish === "glass" ? p.primary : `url(#${id}-enamel)`;
  const faceOpacity = e.finish === "glass" ? ' fill-opacity="0.45"' : "";
  const edge =
    e.finish === "metal"
      ? `<path d="${fp}" fill="none" stroke="${silver}" stroke-width="7"/>`
      : e.finish === "enamel"
        ? `<path d="${fp}" fill="none" stroke="${dark}" stroke-width="4"/>`
        : e.finish === "glass"
          ? `<path d="${fp}" fill="none" stroke="#ffffff" stroke-opacity="0.75" stroke-width="2.5"/>`
          : `<path d="${fp}" fill="none" stroke="${p.primary}" stroke-width="3.5" style="filter:drop-shadow(0 0 3px ${p.primary})"/>`;
  const ink = e.finish === "metal" ? silver : e.finish === "glass" ? "#ffffff" : p.secondary;
  const cut = e.finish === "light" ? "#07090c" : p.primary;
  const lit = e.finish === "light" ? ` style="filter:drop-shadow(0 0 4px ${p.secondary})"` : "";
  let rings = "";
  for (let i = 0; i < e.rings; i++)
    rings += `<circle cx="50" cy="50" r="${36 - i * 7.5}" fill="none" stroke="${e.finish === "metal" ? silver : p.secondary}" stroke-width="${f2(e.ringWeight * 0.7)}"${lit}/>`;
  const letters = e.letters;
  const art =
    e.symbol === null
      ? `<text x="50" y="51" text-anchor="middle" dominant-baseline="central" font-family="Barlow Semi Condensed, sans-serif" font-weight="700" font-size="${letters.length > 2 ? 28 : letters.length > 1 ? 34 : 42}" fill="${ink}">${escapeText(letters)}</text>`
      : SYMBOLS[e.symbol]
          .replaceAll('data-k="S"', `fill="${ink}"`)
          .replaceAll('data-k="F"', `fill="${cut}"`)
          .replaceAll(
            'data-k="L"',
            `fill="none" stroke="${ink}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"`,
          )
          .replaceAll(
            'data-k="B"',
            `fill="none" stroke="${cut}" stroke-width="3.5" stroke-linecap="round"`,
          )
          .replaceAll('data-k="A"', `fill="none" stroke="${ink}" stroke-width="4"`);
  const k = (e.symbolSize / 100) * (e.symbol === null ? 1 : 0.62);
  const glow =
    e.glow > 0
      ? ` style="filter:drop-shadow(0 0 ${Math.round(2 + (e.glow / 100) * 14)}px ${p.primary})"`
      : "";
  return (
    `<defs><clipPath id="${id}-clip"><path d="${fp}"/></clipPath>` +
    `<linearGradient id="${id}-silver" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f4f6f8"/><stop offset=".45" stop-color="#8e949d"/><stop offset=".55" stop-color="#6c727b"/><stop offset="1" stop-color="#dfe2e6"/></linearGradient>` +
    `<radialGradient id="${id}-enamel" cx=".38" cy=".3" r=".85"><stop offset="0" stop-color="${light}"/><stop offset=".55" stop-color="${p.primary}"/><stop offset="1" stop-color="${dark}"/></radialGradient></defs>` +
    `<g${glow}><path d="${fp}" fill="${face}"${faceOpacity}/></g>` +
    `<g clip-path="url(#${id}-clip)">${pattern(e, p.secondary)}</g>` +
    edge +
    rings +
    `<g transform="translate(50 50) scale(${f2(k)}) translate(-50 -50)"${lit}>${art}</g>`
  );
}

/** The whole emblem as an `<svg>` document (the 3D badge's face draws it into a texture). */
export function emblemSvg(
  e: Emblem,
  p: Palette,
  o: { idPrefix?: string; size?: number } = {},
): string {
  const size = o.size ?? 100;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="${size}" height="${size}">${emblemInner(e, p, o)}</svg>`;
}
