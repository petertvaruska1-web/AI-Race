import { motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { signed } from "../format.ts";
import { reducedMotion } from "../motion.ts";

const FLOAT_MS = 1400;

/**
 * Floats each change of `value` ("+12", "−$300") up from where it sits, then
 * fades it (immersion design). Place it inside a `relative` box. Nothing floats
 * under reduced motion.
 */
export function FloatDelta({ value, format }: { value: number; format: (n: number) => string }) {
  const last = useRef(value);
  const seq = useRef(0);
  const timers = useRef(new Set<ReturnType<typeof setTimeout>>());
  const [floats, setFloats] = useState<{ id: number; delta: number }[]>([]);

  useEffect(() => {
    const delta = value - last.current;
    last.current = value;
    if (delta === 0 || reducedMotion()) return;
    seq.current += 1;
    const id = seq.current;
    setFloats((f) => [...f.slice(-2), { id, delta }]);
    const t = setTimeout(() => {
      timers.current.delete(t);
      setFloats((f) => f.filter((x) => x.id !== id));
    }, FLOAT_MS);
    timers.current.add(t);
  }, [value]);

  useEffect(() => {
    const all = timers.current;
    return () => {
      for (const t of all) clearTimeout(t);
    };
  }, []);

  return (
    <span aria-hidden="true" className="pointer-events-none absolute inset-x-0 -top-2 h-0">
      {floats.map((f) => (
        <motion.span
          key={f.id}
          className={`absolute right-0 whitespace-nowrap text-xs font-bold ${f.delta > 0 ? "text-good" : "text-danger"}`}
          initial={{ y: 0, opacity: 1 }}
          animate={{ y: -22, opacity: 0 }}
          transition={{ duration: FLOAT_MS / 1000, ease: "easeOut" }}
        >
          {signed(format)(f.delta)}
        </motion.span>
      ))}
    </span>
  );
}
