import { type KeyboardEvent, useRef } from "react";
import { knobAngle } from "./RotarySelector.tsx";

/**
 * A step dial (design §10, the AI Lab): a knob whose pointer turns to the step on screen, and
 * one lit key per step beside it. Each key is a button named by its step, the current one marked
 * `aria-current="step"`; the arrow keys move to the neighbouring step and its key.
 */
export function StepDial<T extends string>({
  legend,
  steps,
  value,
  onChange,
  className = "",
}: {
  legend: string;
  steps: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
}) {
  const keys = useRef<(HTMLButtonElement | null)[]>([]);
  const index = Math.max(
    0,
    steps.findIndex((s) => s.value === value),
  );
  const onKey = (e: KeyboardEvent, from: number) => {
    const by =
      e.key === "ArrowRight" || e.key === "ArrowDown"
        ? 1
        : e.key === "ArrowLeft" || e.key === "ArrowUp"
          ? -1
          : 0;
    if (by === 0) return;
    e.preventDefault();
    const to = Math.min(steps.length - 1, Math.max(0, from + by));
    const step = steps[to];
    if (!step) return;
    keys.current[to]?.focus();
    if (step.value !== value) onChange(step.value);
  };
  return (
    <nav aria-label={legend} className={`step-dial ${className}`}>
      <div className="bench-knob step-dial__knob" aria-hidden="true">
        <div
          className="bench-knob__cap"
          style={{ transform: `rotate(${knobAngle(index, steps.length)}deg)` }}
        />
      </div>
      <ol className="step-dial__keys">
        {steps.map((s, i) => (
          <li key={s.value}>
            <button
              ref={(el) => {
                keys.current[i] = el;
              }}
              type="button"
              aria-current={i === index ? "step" : undefined}
              className={`keycap step-dial__key ${i === index ? "keycap--on" : ""}`}
              onClick={() => onChange(s.value)}
              onKeyDown={(e) => onKey(e, i)}
            >
              <span className="keycap__lamp" aria-hidden="true" />
              {s.label}
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}
