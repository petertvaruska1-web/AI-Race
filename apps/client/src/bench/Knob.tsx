import { type KeyboardEvent, type PointerEvent, type ReactNode, useRef } from "react";

/**
 * A knob of the bench (design §5.4): turn it by dragging up or down, with the mouse wheel, or
 * with the arrow keys (Page Up/Down for bigger steps, Home and End for the ends). Its value shows
 * in a small display under it. A slider to assistive technology, named by `label`.
 */
export function Knob({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = String,
  size = 64,
  turn = 270,
  ring = "knurl",
  jewel,
  showValue = true,
  showLabel = true,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
  size?: number;
  /** How far the knob turns from `min` to `max`, in degrees. */
  turn?: number;
  ring?: "knurl" | "hue";
  /** A color shown in the knob's center (the hue dials). */
  jewel?: string;
  showValue?: boolean;
  showLabel?: boolean;
}) {
  const drag = useRef<{ y: number; value: number } | null>(null);
  const set = (v: number) => {
    const stepped = Math.round((v - min) / step) * step + min;
    const next = Math.min(max, Math.max(min, stepped));
    if (next !== value) onChange(next);
  };
  const big = Math.max(step, Math.round((max - min) / 10 / step) * step);
  const onKey = (e: KeyboardEvent) => {
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
  const onDown = (e: PointerEvent) => {
    drag.current = { y: e.clientY, value };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onMove = (e: PointerEvent) => {
    if (!drag.current) return;
    set(drag.current.value + ((drag.current.y - e.clientY) / 150) * (max - min));
  };
  const angle = -turn / 2 + ((value - min) / (max - min || 1)) * turn;
  let center: ReactNode = null;
  if (jewel) center = <span className="knob-jewel" style={{ background: jewel, color: jewel }} />;
  return (
    <div className="knob">
      <div
        role="slider"
        tabIndex={0}
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        aria-valuetext={format(value)}
        className={`bench-knob knob__control ${ring === "hue" ? "knob__control--hue" : ""}`}
        style={{ width: size, height: size }}
        onKeyDown={onKey}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={() => {
          drag.current = null;
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onWheel={(e) => set(value + (e.deltaY < 0 ? step : -step))}
      >
        <div className="bench-knob__cap" style={{ transform: `rotate(${angle}deg)` }} />
        {center}
      </div>
      {showValue && (
        <div className="display knob__value">
          <span className="dotface dotface--sm">{format(value)}</span>
        </div>
      )}
      {showLabel && <span className="engrave">{label}</span>}
    </div>
  );
}
