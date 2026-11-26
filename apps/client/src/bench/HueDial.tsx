import type { KeyboardEvent } from "react";
import { Knob } from "./Knob.tsx";

/**
 * A color dial (the founding bench's colors, C61): a knob over the color wheel with the chosen
 * color shining in its center. It wraps around: past 359 comes 0.
 */
export function HueDial({
  label,
  hue,
  color,
  onChange,
  size = 84,
}: {
  label: string;
  hue: number;
  color: string;
  onChange: (hue: number) => void;
  size?: number;
}) {
  // Around the wheel the arrow keys wrap; a drag or the ends stay within 0–359.
  const wrap = (e: KeyboardEvent<HTMLDivElement>) => {
    const by =
      e.key === "ArrowUp" || e.key === "ArrowRight"
        ? 1
        : e.key === "ArrowDown" || e.key === "ArrowLeft"
          ? -1
          : 0;
    if (by === 0) return;
    e.preventDefault();
    e.stopPropagation();
    onChange((hue + by + 360) % 360);
  };
  return (
    <div onKeyDownCapture={wrap}>
      <Knob
        label={label}
        value={hue}
        min={0}
        max={359}
        onChange={onChange}
        format={(h) => `${h}°`}
        size={size}
        turn={359}
        ring="hue"
        jewel={color}
        showValue={false}
      />
    </div>
  );
}
