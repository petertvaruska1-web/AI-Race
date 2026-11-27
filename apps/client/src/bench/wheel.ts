/**
 * The mouse wheel as notches (the knobs and the map, design §5.4 and §8.3): one notch per
 * mouse-wheel event; a trackpad's small steps (under 50 pixels) add up to 50 per notch. A notch
 * is +1 toward the player (a value up, a map closer) and −1 away; 0 while steps add up.
 */
export function createWheel(): (e: WheelEvent) => -1 | 0 | 1 {
  let small = 0;
  return (e) => {
    if (e.deltaY === 0) return 0;
    let toward = e.deltaY < 0;
    if (e.deltaMode === 0 && Math.abs(e.deltaY) < 50) {
      small += e.deltaY;
      if (Math.abs(small) < 50) return 0;
      toward = small < 0;
    }
    small = 0;
    return toward ? 1 : -1;
  };
}
