import { type ReactNode, useId } from "react";

/**
 * A toggle switch with its lamp (design §5.4): a native checkbox with the switch role
 * underneath, named by `label`; `hint` describes it.
 */
export function ToggleSwitch({
  label,
  checked,
  onChange,
  disabled = false,
  hint,
  className = "",
}: {
  label: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  hint?: ReactNode;
  className?: string;
}) {
  const id = useId();
  return (
    <span className={`toggle ${className}`}>
      <label className="toggle__label" htmlFor={id}>
        <input
          id={id}
          type="checkbox"
          role="switch"
          aria-checked={checked}
          className="toggle__input sr-only"
          checked={checked}
          disabled={disabled}
          aria-describedby={hint ? `${id}-hint` : undefined}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="toggle__body" aria-hidden="true">
          <span className="toggle__lamp" />
          <span className="toggle__lever" />
        </span>
        <span className="toggle__text">{label}</span>
      </label>
      {hint && (
        <span id={`${id}-hint`} className="toggle__hint">
          {hint}
        </span>
      )}
    </span>
  );
}
