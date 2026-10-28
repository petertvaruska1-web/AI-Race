import { MotionGlobalConfig } from "motion/react";

/** Whether the player asked for less motion (or motion can't be checked). */
export function reducedMotion(): boolean {
  return (
    typeof window.matchMedia !== "function" ||
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/** Every animation is instant for a player who asked for less motion (immersion design §5). */
export function setupMotion(): void {
  MotionGlobalConfig.skipAnimations = reducedMotion();
}
