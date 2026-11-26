import { emblemOf, type Logo as LogoSpec, type Palette } from "@airace/protocol";
import { useId } from "react";
import { emblemInner } from "../emblem/art.ts";

/**
 * A company's emblem, drawn flat (design §8.2, A4): its own emblem, or the one an older logo maps
 * to so it looks like itself. Letters are escaped text; the drawing is shared with the 3D badge.
 */
export function Logo({
  logo,
  palette,
  size = 64,
  title,
}: {
  logo: LogoSpec;
  palette: Palette;
  size?: number;
  title?: string;
}) {
  const id = `e${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const emblem = emblemOf(logo);
  return (
    <svg
      viewBox="0 0 100 100"
      width={size}
      height={size}
      role="img"
      aria-label={title ?? "Company logo"}
      data-shape={logo.shape}
      data-frame={emblem.frame}
      data-symbol={emblem.symbol ?? ""}
      className="shrink-0 overflow-visible"
      // biome-ignore lint/security/noDangerouslySetInnerHtml: the emblem's art is built from validated fields, its letters escaped (emblem/art.ts)
      dangerouslySetInnerHTML={{ __html: emblemInner(emblem, palette, { idPrefix: id }) }}
    />
  );
}
