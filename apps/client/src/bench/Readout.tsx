import type { ReactNode } from "react";

/** An engraved label over a display that shows one value in the glowing dot face. */
export function Readout({
  label,
  value,
  tone = "lamp",
  size = "md",
  testId,
  className = "",
}: {
  label: ReactNode;
  value: ReactNode;
  tone?: "lamp" | "good" | "danger" | "plain";
  size?: "sm" | "md" | "lg";
  testId?: string;
  className?: string;
}) {
  return (
    <div className={`readout ${className}`}>
      <span className="engrave">{label}</span>
      <div className="display readout__display">
        <span data-testid={testId} className={`dotface dotface--${size} dotface--${tone}`}>
          {value}
        </span>
      </div>
    </div>
  );
}
