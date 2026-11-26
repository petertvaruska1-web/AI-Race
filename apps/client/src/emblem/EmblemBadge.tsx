import { emblemOf, type Logo as LogoSpec, type Palette } from "@airace/protocol";
import { useEffect, useRef } from "react";
import { Logo } from "../components/Logo.tsx";
import { reducedMotion } from "../motion.ts";
import { useGraphics } from "../store/graphics.ts";
import { View3D, type ViewHandle } from "../three/View3D.tsx";
import { type BadgeStage, makeBadgeStage } from "./badgeStage.ts";

/**
 * A company's emblem as a 3D badge (design §8.2), for large places (64 pixels and up, A4): drag
 * to turn it; it wears the bought logo motion. Without WebGL the flat emblem stands in.
 */
export function EmblemBadge({
  logo,
  palette,
  size,
  motion = "logo:still",
  label,
}: {
  logo: LogoSpec;
  palette: Palette;
  size: number;
  motion?: string;
  label: string;
}) {
  const lighter = useGraphics((s) => s.lighter);
  const still = reducedMotion() || lighter;
  const handle = useRef<ViewHandle | null>(null);
  const first = useRef({ logo, palette, still, motion });
  const emblem = emblemOf(logo);
  const key = JSON.stringify([emblem, palette]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: the emblem and palette are compared by value (key)
  useEffect(() => {
    const h = handle.current;
    if (!h) return;
    (h.stage as BadgeStage).setLook(emblem, palette);
    h.markDirty();
  }, [key]);

  return (
    <View3D
      make={() =>
        makeBadgeStage(emblemOf(first.current.logo), first.current.palette, {
          still: first.current.still,
          motion: first.current.motion,
        })
      }
      label={label}
      className="block"
      style={{ width: size, height: size }}
      onReady={(h) => {
        handle.current = h;
        if (h) (h.stage as BadgeStage).setRedraw(h.markDirty);
      }}
      fallback={<Logo logo={logo} palette={palette} size={size} title={label} />}
    />
  );
}
