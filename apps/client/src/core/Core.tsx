import { useEffect, useMemo, useRef } from "react";
import { reducedMotion } from "../motion.ts";
import { Orb } from "../orb/Orb.tsx";
import { type OrbInput, orbParams } from "../orb/params.ts";
import { useGraphics } from "../store/graphics.ts";
import { View3D, type ViewHandle } from "../three/View3D.tsx";
import { type CoreStage, makeCoreStage } from "./coreStage.ts";

/**
 * The AI as a 3D core (design §8.1): a crystal with a glowing heart in geared rings, its look
 * measured (R5). Drag to turn it. Without WebGL, the drawn orb stands in.
 */
export function Core({
  input,
  size = 280,
  label,
  still: held = false,
}: {
  input: OrbInput;
  size?: number;
  label?: string;
  /** Always still, drawn once (small cores in a rack). */
  still?: boolean;
}) {
  const key = JSON.stringify(input);
  // biome-ignore lint/correctness/useExhaustiveDependencies: the input is compared by value (key)
  const params = useMemo(() => orbParams(input), [key]);
  const lighter = useGraphics((s) => s.lighter);
  const still = held || reducedMotion() || lighter;
  const handle = useRef<ViewHandle | null>(null);
  const first = useRef({ params, still });

  useEffect(() => {
    const h = handle.current;
    if (!h) return;
    (h.stage as CoreStage).setParams(params);
    h.markDirty();
  }, [params]);

  useEffect(() => {
    const h = handle.current;
    if (!h) return;
    (h.stage as CoreStage).setStill(still);
    h.markDirty();
  }, [still]);

  return (
    <View3D
      make={() => makeCoreStage(first.current.params, { still: first.current.still })}
      label={label ?? "Your AI"}
      className="block"
      style={{ width: size, height: size }}
      onReady={(h) => {
        handle.current = h;
      }}
      fallback={<Orb input={input} size={size} label={label} />}
    />
  );
}
