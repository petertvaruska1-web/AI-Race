/** Pixels between the pointer and its readout, and between the readout and the chamber's edge. */
const GAP = 14;
const EDGE = 6;

/**
 * Where a region's readout stands in the map's chamber (pixels from its corner): beside the
 * pointer, below and to the right, turned back where it would cross the far edges, and always
 * whole inside the chamber (pinned to the corner when the chamber is smaller than the readout).
 */
export function readoutSpot(
  at: { x: number; y: number },
  chamber: { w: number; h: number },
  box: { w: number; h: number },
): { left: number; top: number } {
  let left = at.x + GAP;
  let top = at.y + GAP;
  if (left + box.w > chamber.w - EDGE) left = at.x - GAP - box.w;
  if (top + box.h > chamber.h - EDGE) top = at.y - GAP - box.h;
  return {
    left: Math.max(EDGE, Math.min(left, chamber.w - EDGE - box.w)),
    top: Math.max(EDGE, Math.min(top, chamber.h - EDGE - box.h)),
  };
}
