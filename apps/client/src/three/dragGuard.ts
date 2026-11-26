/**
 * Tells a drag from a click on a 3D object that sits inside a button (the core opens its AI's
 * window): once the pointer has moved more than `threshold` CSS pixels, the click that ends the
 * drag is swallowed, so turning the object never opens the window.
 */
export function createDragGuard(threshold = 4): {
  down(x: number, y: number): void;
  move(x: number, y: number): void;
  up(): void;
  takeClick(): boolean;
} {
  let start: { x: number; y: number } | null = null;
  let dragged = false;
  return {
    down(x, y) {
      start = { x, y };
      dragged = false;
    },
    move(x, y) {
      if (start && Math.hypot(x - start.x, y - start.y) > threshold) dragged = true;
    },
    up() {
      start = null;
    },
    takeClick() {
      const was = dragged;
      dragged = false;
      return was;
    },
  };
}
