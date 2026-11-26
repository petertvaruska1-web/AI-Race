import type { ButtonHTMLAttributes } from "react";
import { PushButton, type PushVariant } from "../bench/PushButton.tsx";

type Variant = "primary" | "secondary" | "ghost";

const KEYS: Record<Variant, PushVariant> = {
  primary: "primary",
  secondary: "secondary",
  ghost: "quiet",
};

/** A push button of the bench in the company's lamp color. A disabled one keeps its reason in `title`. */
export function Button({
  variant = "primary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return <PushButton variant={KEYS[variant]} className={className} {...props} />;
}
