import type { ButtonHTMLAttributes } from "react";

export type PushVariant = "primary" | "secondary" | "quiet";

/**
 * A push button of the bench (design §5.4): a key that travels down when pressed; the primary
 * one has a lit ring in the lamp color. A disabled one keeps its reason in `title`.
 */
export function PushButton({
  variant = "primary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: PushVariant }) {
  return <button type="button" className={`push push--${variant} ${className}`} {...props} />;
}
