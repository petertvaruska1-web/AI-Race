/**
 * A slider of the bench (design §5.4): a native range input under a machined cap in a lit track,
 * named by `label` (engraved above it).
 */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  showLabel = true,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  showLabel?: boolean;
}) {
  const filled = ((value - min) / (max - min || 1)) * 100;
  return (
    <label className="bench-slider">
      {showLabel && <span className="engrave">{label}</span>}
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="bench-slider__input"
        style={{ ["--filled" as string]: `${filled}%` }}
      />
    </label>
  );
}
