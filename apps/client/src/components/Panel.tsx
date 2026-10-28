import type { HTMLAttributes, ReactNode } from "react";

/** A glass panel with an optional title, hint and actions (the game's basic surface). */
export function Panel({
  title,
  hint,
  actions,
  children,
  className = "",
  ...props
}: Omit<HTMLAttributes<HTMLElement>, "title"> & {
  title?: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className={`panel p-5 ${className}`} {...props}>
      {(title || hint || actions) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-lg font-semibold">{title}</h2>}
            {hint && <p className="mt-1 text-sm text-muted">{hint}</p>}
          </div>
          {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}
