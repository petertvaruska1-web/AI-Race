import { useLayoutEffect } from "react";

/** The CSS bottom of a corner panel: above a screen's bottom bar while one is shown. */
export const ABOVE_BOTTOM_BAR = "calc(1rem + var(--bottom-bar, 0px))";

/**
 * A sticky bottom bar (the AI Lab's Start training) reserves its height, so the notifications
 * and the world chat in the bottom corners sit above it instead of covering its button.
 */
export function useBottomBar(bar: HTMLElement | null): void {
  useLayoutEffect(() => {
    if (!bar) return;
    const root = document.documentElement;
    const reserve = () =>
      root.style.setProperty(
        "--bottom-bar",
        `${Math.ceil(bar.getBoundingClientRect().height) + 16}px`,
      );
    reserve();
    const watch = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(reserve);
    watch?.observe(bar);
    return () => {
      watch?.disconnect();
      root.style.removeProperty("--bottom-bar");
    };
  }, [bar]);
}
