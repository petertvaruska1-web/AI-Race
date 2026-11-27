import { REGION_IDS, REGION_NAMES, type RegionId } from "@airace/protocol";
import {
  type CSSProperties,
  type MouseEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import { PushButton } from "../bench/PushButton.tsx";
import { createWheel } from "../bench/wheel.ts";
import { type MapLook, type MapStage, makeMapStage } from "../map/mapStage.ts";
import { TileMap } from "../map/TileMap.tsx";
import { reducedMotion } from "../motion.ts";
import { REGION_CENTERS, type RegionMarks, type RegionTop } from "../regions.ts";
import { useGraphics } from "../store/graphics.ts";
import { View3D, type ViewHandle } from "../three/View3D.tsx";

/**
 * The US map (design §8.3) in a dark chamber: the 3D tabletop model where WebGL works, the tile
 * map where it doesn't. Each region's top is its leader's color and height, the same for
 * everybody; the player's own marks stand on it. Drag to turn it, drag with the right button to
 * tilt, use the wheel to zoom once it is picked (clicked), and "Reset view" to go back. Hovering a
 * region shows its readout; clicking it opens it. Every region also has a real key that opens it.
 */
export function RegionMap({
  label,
  tops,
  marks = {},
  focus = null,
  onOpen,
  readout,
  small = false,
  markColor,
}: {
  label: string;
  tops: Record<RegionId, RegionTop>;
  marks?: RegionMarks;
  focus?: RegionId | null;
  onOpen?: (region: RegionId, el: Element) => void;
  readout?: (region: RegionId) => ReactNode;
  small?: boolean;
  /** The player's own color for their marks. */
  markColor: string;
}) {
  const lighter = useGraphics((s) => s.lighter);
  const still = reducedMotion() || lighter;
  const [hover, setHover] = useState<RegionId | null>(null);
  /** Where the readout stands in the 3D view (pixels from its corner). */
  const [at, setAt] = useState<{ x: number; y: number } | null>(null);
  const [three, setThree] = useState(false);
  const handle = useRef<ViewHandle | null>(null);
  const view = useRef<HTMLDivElement>(null);
  const look: MapLook = { tops, marks, focus, markColor };
  const first = useRef({ look, still, lighter });
  const key = JSON.stringify(look);
  const stage = () => handle.current?.stage as MapStage | undefined;

  // biome-ignore lint/correctness/useExhaustiveDependencies: the look is compared by value (key)
  useEffect(() => {
    stage()?.setLook(look);
    handle.current?.markDirty();
  }, [key]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: `stage` reads a ref
  useEffect(() => {
    stage()?.setHover(hover);
    handle.current?.markDirty();
  }, [hover]);

  // The wheel zooms only a map the player has picked, so scrolling the page past it changes nothing.
  useEffect(() => {
    const el = view.current;
    if (!el) return;
    const notches = createWheel();
    const onWheel = (e: WheelEvent) => {
      const s = handle.current?.stage as MapStage | undefined;
      if (!s || !el.contains(document.activeElement)) return;
      e.preventDefault();
      const by = notches(e);
      if (by === 0) return;
      s.zoom(by);
      handle.current?.markDirty();
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  /** The region under the pointer in the 3D view, and where the pointer is in it. */
  const under = (e: MouseEvent<HTMLDivElement>) => {
    const s = stage();
    const el = view.current;
    if (!s || !el) return { region: null, x: 0, y: 0 };
    const r = el.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const nx = r.width > 0 ? (x / r.width) * 2 - 1 : 0;
    const ny = r.height > 0 ? -((y / r.height) * 2 - 1) : 0;
    return { region: s.pick(nx, ny), x, y };
  };

  /** A key's region lights, its readout over the region. */
  const showKey = (region: RegionId) => {
    setHover(region);
    const s = stage();
    const el = view.current;
    if (!s || !el) {
      setAt(null);
      return;
    }
    const p = s.anchor(region).project(s.camera);
    const r = el.getBoundingClientRect();
    setAt({ x: ((p.x + 1) / 2) * r.width, y: ((1 - p.y) / 2) * r.height });
  };
  const hide = () => {
    setHover(null);
    setAt(null);
  };

  let place: CSSProperties | null = null;
  if (hover && three && at) {
    const w = view.current?.clientWidth ?? 0;
    const h = view.current?.clientHeight ?? 0;
    place = {
      left: at.x + 14,
      top: at.y + 14,
      transform: `translate(${at.x > w - 240 ? "calc(-100% - 28px)" : "0"}, ${at.y > h - 110 ? "calc(-100% - 28px)" : "0"})`,
    };
  } else if (hover) {
    const c = REGION_CENTERS[hover];
    place = {
      left: `${((c.x + 0.3) / 12.6) * 100}%`,
      top: `${((c.y - 0.4 + 0.3) / 8.6) * 100}%`,
      transform: "translate(-50%, -100%)",
    };
  }

  return (
    <fieldset className="region-map" aria-label={label}>
      {/* biome-ignore lint/a11y/noStaticElementInteractions: pointer shortcuts over the 3D map; every region also has a real key below */}
      {/* biome-ignore lint/a11y/useKeyWithClickEvents: the region keys are the keyboard's way in */}
      <div
        ref={view}
        data-testid="map-view"
        tabIndex={-1}
        className={`display region-map__view ${small ? "region-map__view--small" : ""}`}
        onPointerMove={(e) => {
          if (!three || e.buttons !== 0) return;
          const u = under(e);
          setHover(u.region);
          setAt(u.region ? { x: u.x, y: u.y } : null);
        }}
        onPointerLeave={() => {
          if (three) hide();
        }}
        onClick={(e) => {
          if (!three) return;
          const { region } = under(e);
          if (region) onOpen?.(region, e.currentTarget);
        }}
        onContextMenu={(e) => {
          if (three) e.preventDefault();
        }}
      >
        <View3D
          make={() =>
            makeMapStage(first.current.look, {
              still: first.current.still,
              lighter: first.current.lighter,
            })
          }
          label={label}
          className="block h-full w-full"
          onReady={(h) => {
            handle.current = h;
            setThree(Boolean(h));
          }}
          fallback={
            <TileMap
              tops={tops}
              marks={marks}
              focus={focus}
              hover={hover}
              onHover={setHover}
              onOpen={onOpen}
              small={small}
              markColor={markColor}
            />
          }
        />
        {hover && readout && place && (
          <div className="display region-map__readout" style={place}>
            <p className="m-0 mb-1 font-semibold">{REGION_NAMES[hover]}</p>
            {readout(hover)}
          </div>
        )}
        {three && (
          <PushButton
            variant="secondary"
            className="region-map__reset"
            onClick={(e) => {
              e.stopPropagation();
              stage()?.resetView();
              handle.current?.markDirty();
            }}
          >
            Reset view
          </PushButton>
        )}
      </div>
      <ul className={small ? "sr-only" : "region-keys"}>
        {REGION_IDS.map((r) => (
          <li key={r}>
            <button
              type="button"
              aria-label={`Open ${REGION_NAMES[r]}`}
              onClick={(e) => onOpen?.(r, e.currentTarget)}
              onPointerEnter={() => showKey(r)}
              onPointerLeave={hide}
              onFocus={() => showKey(r)}
              onBlur={hide}
              className={`region-key ${hover === r ? "region-key--on" : ""}`}
            >
              <span
                className="region-key__lamp"
                style={tops[r].color ? { background: tops[r].color ?? undefined } : undefined}
                aria-hidden="true"
              />
              {REGION_NAMES[r]}
            </button>
          </li>
        ))}
      </ul>
    </fieldset>
  );
}
