import {
  LOGO_SHAPES,
  LOGO_SYMBOLS,
  type LogoShape,
  type Logo as LogoSpec,
  type Palette,
} from "@airace/protocol";
import { INPUT } from "./Field.tsx";
import { Logo } from "./Logo.tsx";

const SHAPE_NAMES: Record<LogoShape, string> = {
  circle: "Circle",
  "rounded-square": "Rounded square",
  hexagon: "Hexagon",
  shield: "Shield",
  diamond: "Diamond",
};

/** Only letters and digits, at most two, in capitals (the server's rule for initials). */
export function cleanInitials(s: string): string {
  return s
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "")
    .slice(0, 2);
}

export function LogoBuilder({
  logo,
  palette,
  onChange,
}: {
  logo: LogoSpec;
  palette: Palette;
  onChange: (logo: LogoSpec) => void;
}) {
  const usesInitials = logo.symbol === null;
  return (
    <div className="space-y-4">
      <fieldset>
        <legend className="mb-2 text-sm font-semibold">Shape</legend>
        <div className="flex flex-wrap gap-2">
          {LOGO_SHAPES.map((shape) => (
            <button
              key={shape}
              type="button"
              aria-pressed={logo.shape === shape}
              aria-label={SHAPE_NAMES[shape]}
              title={SHAPE_NAMES[shape]}
              onClick={() => onChange({ ...logo, shape })}
              className={`rounded-xl border p-1 ${logo.shape === shape ? "border-brand bg-brand-soft" : "border-border"}`}
            >
              <Logo
                logo={{ ...logo, shape }}
                palette={palette}
                size={40}
                title={SHAPE_NAMES[shape]}
              />
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend className="mb-2 text-sm font-semibold">Inside it</legend>
        <div className="mb-3 flex gap-4 text-sm">
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="logo-content"
              checked={!usesInitials}
              onChange={() => onChange({ ...logo, symbol: logo.symbol ?? "spark" })}
            />
            A symbol
          </label>
          <label className="flex items-center gap-2">
            <input
              type="radio"
              name="logo-content"
              checked={usesInitials}
              onChange={() => onChange({ ...logo, symbol: null })}
            />
            Your initials
          </label>
        </div>
        {usesInitials ? (
          <input
            className={`${INPUT} max-w-24 uppercase`}
            aria-label="Initials"
            placeholder="AB"
            value={logo.initials}
            maxLength={2}
            onChange={(e) => onChange({ ...logo, initials: cleanInitials(e.target.value) })}
          />
        ) : (
          <div className="grid grid-cols-8 gap-2">
            {LOGO_SYMBOLS.map((symbol) => (
              <button
                key={symbol}
                type="button"
                aria-pressed={logo.symbol === symbol}
                aria-label={symbol}
                title={symbol}
                onClick={() => onChange({ ...logo, symbol })}
                className={`rounded-xl border p-1 ${logo.symbol === symbol ? "border-brand bg-brand-soft" : "border-border"}`}
              >
                <Logo logo={{ ...logo, symbol }} palette={palette} size={32} title={symbol} />
              </button>
            ))}
          </div>
        )}
      </fieldset>
    </div>
  );
}
