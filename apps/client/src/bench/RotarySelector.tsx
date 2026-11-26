import { type KeyboardEvent, useId } from "react";

/** How far the knob turns from its first position to its last, in degrees. */
export function knobAngle(index: number, count: number): number {
  if (count < 2) return 0;
  const span = Math.min(270, 40 * (count - 1));
  return -span / 2 + (index * span) / (count - 1);
}

/**
 * A rotary selector for a choice among many (design §5.4): a knob that turns to the chosen
 * position, the positions engraved beside it as native radios. The arrow keys step and wrap.
 */
export function RotarySelector<T extends string>({
  legend,
  options,
  value,
  onChange,
  size = 64,
  className = "",
}: {
  legend: string;
  options: { label: string; value: T }[];
  value: T;
  onChange: (value: T) => void;
  size?: number;
  className?: string;
}) {
  const name = useId();
  const index = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  );
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const step =
      e.key === "ArrowRight" || e.key === "ArrowDown"
        ? 1
        : e.key === "ArrowLeft" || e.key === "ArrowUp"
          ? -1
          : 0;
    if (step === 0 || options.length === 0) return;
    e.preventDefault();
    const next = (index + step + options.length) % options.length;
    const option = options[next];
    if (!option) return;
    onChange(option.value);
    e.currentTarget.querySelectorAll("input")[next]?.focus();
  };
  return (
    <div role="radiogroup" aria-label={legend} className={`rotary ${className}`} onKeyDown={onKey}>
      <div className="rotary__knob" style={{ width: size, height: size }} aria-hidden="true">
        <div
          className="rotary__cap"
          style={{ transform: `rotate(${knobAngle(index, options.length)}deg)` }}
        />
      </div>
      <div className="rotary__marks">
        {options.map((o) => (
          <label
            key={o.value}
            className={`rotary__mark ${o.value === value ? "rotary__mark--on" : ""}`}
          >
            <input
              type="radio"
              name={name}
              className="sr-only"
              value={o.value}
              checked={o.value === value}
              onChange={() => onChange(o.value)}
            />
            <span
              className="lamp"
              data-state={o.value === value ? "on" : "off"}
              aria-hidden="true"
            />
            {o.label}
          </label>
        ))}
      </div>
    </div>
  );
}
