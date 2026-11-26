export type LampState = "on" | "ready" | "wait" | "problem" | "off";

/**
 * A small lamp (design §5.1): lit in the lamp color for "on", green when ready or done, amber
 * while waiting, red for a problem. `blink` for warnings that need the player.
 */
export function Lamp({
  state,
  label,
  blink = false,
}: {
  state: LampState;
  label: string;
  blink?: boolean;
}) {
  return (
    <span
      role="img"
      aria-label={label}
      data-state={state}
      className={`lamp ${blink ? "lamp--blink" : ""}`}
    />
  );
}
