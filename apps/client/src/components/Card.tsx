import type { HTMLAttributes, ReactNode } from "react";

export function Card({
  title,
  hint,
  children,
  className = "",
  ...props
}: HTMLAttributes<HTMLElement> & { title?: ReactNode; hint?: ReactNode }) {
  return (
    <section className={`rounded-2xl border border-border bg-surface p-5 ${className}`} {...props}>
      {title && <h2 className="mb-1 text-lg font-semibold">{title}</h2>}
      {hint && <p className="mb-4 text-sm text-muted">{hint}</p>}
      {children}
    </section>
  );
}
