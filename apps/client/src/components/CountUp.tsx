import { useEffect, useRef, useState } from "react";

const ROLL_MS = 700;

function reducedMotion(): boolean {
  return (
    typeof window.matchMedia !== "function" ||
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * A number that rolls to each new value (eased, about 0.7 s); the first value shows at once.
 * With reduced motion, or where motion can't be checked, it jumps.
 */
export function CountUp({ value, format }: { value: number; format: (n: number) => string }) {
  const [shown, setShown] = useState(value);
  const current = useRef(value);

  useEffect(() => {
    const from = current.current;
    if (from === value || reducedMotion()) {
      current.current = value;
      setShown(value);
      return;
    }
    const start = performance.now();
    let frame = 0;
    const roll = (now: number) => {
      const k = Math.min(1, (now - start) / ROLL_MS);
      const next = from + (value - from) * (1 - (1 - k) ** 3);
      current.current = next;
      setShown(next);
      if (k < 1) frame = requestAnimationFrame(roll);
    };
    frame = requestAnimationFrame(roll);
    return () => cancelAnimationFrame(frame);
  }, [value]);

  return <span className="tabular">{format(shown)}</span>;
}
