import type { HTMLAttributes, ReactNode } from "react";

/**
 * A module of the bench (design §5.1): a machined plate with four screws, its name engraved.
 * `heavy` for the most important module of a station, `thin` for small status plates.
 */
export function Module({
  label,
  hint,
  actions,
  weight = "plate",
  className = "",
  children,
  ...props
}: Omit<HTMLAttributes<HTMLElement>, "title"> & {
  label?: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
  weight?: "plate" | "heavy" | "thin";
  className?: string;
  children?: ReactNode;
}) {
  return (
    <section className={`module module--${weight} ${className}`} {...props}>
      {(label || hint || actions) && (
        <header className="module__head">
          <div className="min-w-0">
            {label && <h2 className="engrave">{label}</h2>}
            {hint && <p className="module__hint">{hint}</p>}
          </div>
          {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}
