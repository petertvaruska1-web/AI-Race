import { describe, expect, it } from "vitest";
import { readoutSpot } from "../src/map/readoutSpot.ts";

describe("where a region's readout stands on the map", () => {
  it("keeps it whole inside a small map wherever the pointer is", () => {
    const chamber = { w: 264, h: 150 };
    const box = { w: 200, h: 80 };
    for (let x = 0; x <= 264; x += 33)
      for (let y = 0; y <= 150; y += 25) {
        const at = readoutSpot({ x, y }, chamber, box);
        expect(at.left, `${x},${y}`).toBeGreaterThanOrEqual(6);
        expect(at.left + box.w, `${x},${y}`).toBeLessThanOrEqual(264 - 6);
        expect(at.top, `${x},${y}`).toBeGreaterThanOrEqual(6);
        expect(at.top + box.h, `${x},${y}`).toBeLessThanOrEqual(150 - 6);
      }
  });

  it("stands beside the pointer on a large map, turning back at the far edges", () => {
    const chamber = { w: 900, h: 300 };
    const box = { w: 220, h: 80 };
    expect(readoutSpot({ x: 100, y: 100 }, chamber, box)).toEqual({ left: 114, top: 114 });
    expect(readoutSpot({ x: 850, y: 250 }, chamber, box)).toEqual({ left: 616, top: 156 });
  });

  it("pins it to the corner when the map is smaller than the readout", () => {
    expect(readoutSpot({ x: 40, y: 30 }, { w: 150, h: 60 }, { w: 200, h: 80 })).toEqual({
      left: 6,
      top: 6,
    });
  });
});
