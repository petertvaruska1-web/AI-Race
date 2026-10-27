import type { LogoShape, Logo as LogoSpec, LogoSymbol, Palette } from "@airace/protocol";
import type { ReactNode } from "react";

// A company logo drawn as SVG from its shape, symbol or initials and palette.
// Initials are SVG text nodes, so whatever a player types stays text.

const SHAPES: Record<LogoShape, ReactNode> = {
  circle: <circle cx="50" cy="50" r="46" />,
  "rounded-square": <rect x="4" y="4" width="92" height="92" rx="22" />,
  hexagon: <polygon points="50,4 90,27 90,73 50,96 10,73 10,27" />,
  shield: <path d="M50 4 L88 16 V46 C88 72 70 88 50 96 C30 88 12 72 12 46 V16 Z" />,
  diamond: <polygon points="50,4 96,50 50,96 4,50" />,
};

const LINE = {
  fill: "none",
  strokeWidth: 6,
  strokeLinecap: "round",
  strokeLinejoin: "round",
} as const;

const SYMBOLS: Record<LogoSymbol, ReactNode> = {
  spark: <path d="M50 22 L56 44 L78 50 L56 56 L50 78 L44 56 L22 50 L44 44 Z" />,
  star: <path d="M50 22 L58 42 L80 42 L62 55 L69 77 L50 63 L31 77 L38 55 L20 42 L42 42 Z" />,
  bolt: <path d="M56 20 L32 54 H48 L42 80 L68 44 H52 Z" />,
  moon: <path d="M60 24 A28 28 0 1 0 76 64 A22 22 0 1 1 60 24 Z" />,
  sun: (
    <g>
      <circle cx="50" cy="50" r="13" />
      <path
        {...LINE}
        d="M50 22 V30 M50 70 V78 M22 50 H30 M70 50 H78 M30 30 L36 36 M64 64 L70 70 M30 70 L36 64 M64 36 L70 30"
      />
    </g>
  ),
  leaf: (
    <g>
      <path d="M28 72 C28 40 48 26 74 26 C74 54 58 72 28 72 Z" />
      <path {...LINE} strokeWidth={4} d="M32 68 L60 40" style={{ stroke: "var(--logo-bg)" }} />
    </g>
  ),
  wave: (
    <path
      {...LINE}
      d="M20 46 C30 32 40 32 50 46 S70 60 80 46 M20 62 C30 48 40 48 50 62 S70 76 80 62"
    />
  ),
  eye: (
    <g>
      <path d="M18 50 C32 30 68 30 82 50 C68 70 32 70 18 50 Z" />
      <circle cx="50" cy="50" r="10" style={{ fill: "var(--logo-bg)" }} />
    </g>
  ),
  flame: (
    <path d="M50 18 C62 34 72 46 72 60 A22 22 0 0 1 28 60 C28 48 38 42 42 30 C46 40 52 42 50 18 Z" />
  ),
  rocket: (
    <g>
      <path d="M50 18 C62 28 66 44 62 62 H38 C34 44 38 28 50 18 Z" />
      <path d="M38 54 L26 70 L40 66 Z M62 54 L74 70 L60 66 Z" />
      <circle cx="50" cy="40" r="6" style={{ fill: "var(--logo-bg)" }} />
    </g>
  ),
  brain: (
    <g>
      <path d="M50 28 C38 22 26 30 28 42 C20 48 24 62 34 64 C36 74 48 76 50 68 C52 76 64 74 66 64 C76 62 80 48 72 42 C74 30 62 22 50 28 Z" />
      <path {...LINE} strokeWidth={3} d="M50 30 V68" style={{ stroke: "var(--logo-bg)" }} />
    </g>
  ),
  cube: (
    <g>
      <polygon points="50,20 76,34 76,64 50,78 24,64 24,34" />
      <path
        {...LINE}
        strokeWidth={3}
        d="M24 34 L50 48 L76 34 M50 48 V78"
        style={{ stroke: "var(--logo-bg)" }}
      />
    </g>
  ),
  atom: (
    <g {...LINE} strokeWidth={4}>
      <ellipse cx="50" cy="50" rx="30" ry="11" />
      <ellipse cx="50" cy="50" rx="30" ry="11" transform="rotate(60 50 50)" />
      <ellipse cx="50" cy="50" rx="30" ry="11" transform="rotate(120 50 50)" />
      <circle cx="50" cy="50" r="5" style={{ fill: "currentColor", stroke: "none" }} />
    </g>
  ),
  anchor: (
    <path
      {...LINE}
      d="M50 34 V74 M38 44 H62 M30 60 C32 70 42 74 50 74 C58 74 68 70 70 60 M50 22 A6 6 0 1 1 49.9 22"
    />
  ),
  crown: <path d="M24 68 L28 34 L40 50 L50 28 L60 50 L72 34 L76 68 Z" />,
  feather: (
    <g>
      <path d="M72 22 C46 26 30 46 30 72 L36 66 C50 64 66 50 72 22 Z" />
      <path {...LINE} strokeWidth={3} d="M30 74 L58 40" style={{ stroke: "var(--logo-bg)" }} />
    </g>
  ),
};

export function Logo({
  logo,
  palette,
  size = 64,
  title,
}: {
  logo: LogoSpec;
  palette: Palette;
  size?: number;
  title?: string;
}) {
  const initials = logo.initials;
  return (
    <svg
      viewBox="0 0 100 100"
      width={size}
      height={size}
      role="img"
      aria-label={title ?? "Company logo"}
      data-shape={logo.shape}
      data-symbol={logo.symbol ?? ""}
      style={{ ["--logo-bg" as string]: palette.primary, color: palette.secondary }}
    >
      <g style={{ fill: palette.primary }}>{SHAPES[logo.shape]}</g>
      {logo.symbol ? (
        <g style={{ fill: "currentColor", stroke: "currentColor" }}>{SYMBOLS[logo.symbol]}</g>
      ) : (
        <text
          x="50"
          y="50"
          dominantBaseline="central"
          textAnchor="middle"
          fontFamily="Space Grotesk, sans-serif"
          fontWeight="700"
          fontSize={initials.length > 1 ? 38 : 46}
          style={{ fill: palette.secondary }}
        >
          {initials}
        </text>
      )}
    </svg>
  );
}

export { SYMBOLS as LOGO_SYMBOL_ART };
