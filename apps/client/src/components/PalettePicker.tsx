import type { Palette } from "@airace/protocol";
import { BASE_BG, contrastRatio, PALETTE_PRESETS } from "@airace/sim";
import { useState } from "react";

export const HARD_TO_READ =
  "This color is hard to read on the dark background. Pick a brighter one.";

/** Shown under a custom color the dashboard can't show well (the server refuses it too). */
function readable(color: string): boolean {
  return contrastRatio(color, BASE_BG) >= 3;
}

export function PalettePicker({
  value,
  onChange,
}: {
  value: Palette;
  onChange: (p: Palette) => void;
}) {
  const preset = PALETTE_PRESETS.find(
    (p) => p.palette.primary === value.primary && p.palette.secondary === value.secondary,
  );
  const [custom, setCustom] = useState(!preset);
  const warning =
    custom && (!readable(value.primary) || !readable(value.secondary)) ? HARD_TO_READ : null;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-6 gap-2">
        {PALETTE_PRESETS.map((p) => {
          const chosen = !custom && preset?.id === p.id;
          return (
            <button
              key={p.id}
              type="button"
              aria-pressed={chosen}
              title={p.label}
              onClick={() => {
                setCustom(false);
                onChange(p.palette);
              }}
              className={`flex flex-col items-center gap-1 rounded-xl border p-2 text-xs ${chosen ? "border-brand bg-brand-soft" : "border-border"}`}
            >
              <span className="flex">
                <span className="size-5 rounded-l-full" style={{ background: p.palette.primary }} />
                <span
                  className="size-5 rounded-r-full"
                  style={{ background: p.palette.secondary }}
                />
              </span>
              {p.label}
            </button>
          );
        })}
      </div>
      <button
        type="button"
        aria-pressed={custom}
        onClick={() => setCustom(true)}
        className={`rounded-xl border px-3 py-2 text-sm ${custom ? "border-brand bg-brand-soft" : "border-border"}`}
      >
        Custom colors
      </button>
      {custom && (
        <div className="flex flex-wrap gap-4 text-sm">
          <label className="flex items-center gap-2">
            Main color
            <input
              type="color"
              aria-label="Main color"
              value={value.primary}
              onChange={(e) => onChange({ ...value, primary: e.target.value.toLowerCase() })}
            />
          </label>
          <label className="flex items-center gap-2">
            Second color
            <input
              type="color"
              aria-label="Second color"
              value={value.secondary}
              onChange={(e) => onChange({ ...value, secondary: e.target.value.toLowerCase() })}
            />
          </label>
        </div>
      )}
      {warning && (
        <p role="alert" className="text-sm text-danger">
          {warning}
        </p>
      )}
    </div>
  );
}
