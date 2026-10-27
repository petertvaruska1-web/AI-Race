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

export const INPUT =
  "w-full rounded-lg border border-border bg-surface-2 px-3 py-2 text-text outline-none focus:border-brand";
