import { useId } from "react";

export type KeyOption<T extends string> = {
  label: string;
  value: T;
  disabled?: boolean;
  title?: string;
};

/**
 * A row of illuminated keys for a choice of a few (design §5.4): native radios underneath,
 * grouped under `legend`; the chosen key is down and lit.
 */
export function KeyRow<T extends string>({
  legend,
  options,
  value,
  onChange,
  className = "",
}: {
  legend: string;
  options: KeyOption<T>[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
}) {
  const name = useId();
  return (
    <fieldset className={`keyrow ${className}`}>
      <legend className="sr-only">{legend}</legend>
      {options.map((o) => (
        <label
          key={o.value}
          title={o.title}
          data-disabled={o.disabled || undefined}
          className={`keycap ${value === o.value ? "keycap--on" : ""}`}
        >
          <input
            type="radio"
            name={name}
            className="sr-only"
            value={o.value}
            checked={value === o.value}
            disabled={o.disabled}
            onChange={() => onChange(o.value)}
          />
          <span className="keycap__lamp" aria-hidden="true" />
          {o.label}
        </label>
      ))}
    </fieldset>
  );
}
