import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost";

const STYLES: Record<Variant, string> = {
  primary: "bg-brand text-on-brand hover:brightness-110 disabled:opacity-40",
  secondary: "bg-surface-2 text-text border border-border hover:border-brand disabled:opacity-40",
  ghost: "text-muted hover:text-text disabled:opacity-40",
};

/** A button in the company's colors. A disabled one keeps its reason in `title`. */
export function Button({
  variant = "primary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      type="button"
      className={`rounded-lg px-4 py-2 text-sm font-semibold transition disabled:cursor-not-allowed ${STYLES[variant]} ${className}`}
      {...props}
    />
  );
}
