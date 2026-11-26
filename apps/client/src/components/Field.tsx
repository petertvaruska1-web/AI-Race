import type { ReactNode } from "react";

/** A labelled control with a one-line "what this does" and an optional warning. */
export function Field({
  label,
  hint,
  warning,
  children,
}: {
  label: ReactNode;
  hint?: ReactNode;
  warning?: ReactNode;
  children: ReactNode;
}) {
  return (
    // biome-ignore lint/a11y/noLabelWithoutControl: the control is passed in as children
    <label className="block space-y-1">
      <span className="block text-sm font-semibold">{label}</span>
      {hint && <span className="block text-xs text-muted">{hint}</span>}
      {children}
      {warning && (
        <span role="alert" className="block text-xs text-danger">
          {warning}
        </span>
      )}
    </label>
  );
}

/** A text field, select or number box: a display you type into (bench.css). */
export const INPUT = "display-input";
