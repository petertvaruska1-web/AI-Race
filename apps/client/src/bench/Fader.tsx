import type { CSSProperties, KeyboardEvent } from "react";

/**
 * A fader of a mixing desk (design §10, the AI Lab's data): a vertical slider in a recessed
 * track with a machined cap, a level meter beside it and its reading in a small display above.
 * The arrow keys step it (Page Up and Down by tenths, Home and End to the ends). A native
 * slider underneath, named by `label`, for assistive technology and the tests.
 */
export function Fader({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  level,
  readout,
  testId,
  disabled = false,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  /** How full the level meter stands, 0–1. */
  level: number;
  /** The fader's reading, shown in the display above it. */
  readout: string;
  testId?: string;
  disabled?: boolean;
}) {
  const set = (v: number) => {
    const next = Math.min(max, Math.max(min, Math.round((v - min) / step) * step + min));
    if (next !== value) onChange(next);
  };
  const big = Math.max(step, Math.round((max - min) / 10 / step) * step);
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    const by: Record<string, number> = {
      ArrowUp: step,
      ArrowRight: step,
      ArrowDown: -step,
      ArrowLeft: -step,
      PageUp: big,
      PageDown: -big,
    };
    if (e.key === "Home") set(min);
    else if (e.key === "End") set(max);
    else if (e.key in by) set(value + (by[e.key] ?? 0));
    else return;
    e.preventDefault();
  };
  const filled = ((value - min) / (max - min || 1)) * 100;
  const lit = Math.round(Math.min(1, Math.max(0, level)) * 100);
  return (
    <div className="fader" data-testid={testId}>
      <div className="display fader__readout">
        <span className="dotface dotface--xs">{readout}</span>
      </div>
      <div className="fader__body">
        <input
          type="range"
          className="fader__input"
          min={min}
          max={max}
          step={step}
          value={value}
          aria-label={label}
          aria-orientation="vertical"
          disabled={disabled}
          style={{ "--filled": `${filled}%` } as CSSProperties}
          onChange={(e) => set(Number(e.target.value))}
          onKeyDown={onKey}
        />
        <div className="fader__level" aria-hidden="true">
          <span style={{ height: `${lit}%` }} />
        </div>
      </div>
    </div>
  );
}
