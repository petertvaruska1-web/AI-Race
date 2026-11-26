import { type CSSProperties, type ReactNode, useEffect, useRef, useState } from "react";
import { createDragGuard } from "./dragGuard.ts";
import { attachView, markInteracting, type Stage3D } from "./engine.ts";
import { webglAvailable } from "./support.ts";

export type ViewHandle = { stage: Stage3D; markDirty(): void };

/**
 * A 3D object on the page (design §8.0): its own canvas, drawn by the shared engine while it is
 * on screen. Drags on it reach the object. Without WebGL (or when the object can't be built),
 * `fallback` shows instead, so nothing depends on 3D to work.
 */
export function View3D({
  make,
  label,
  fallback,
  className = "",
  style,
  onReady,
}: {
  make: () => Stage3D;
  label: string;
  fallback: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** Called once the object is built, to change it later (then `markDirty`). */
  onReady?: (handle: ViewHandle | null) => void;
}) {
  const [ok, setOk] = useState(() => webglAvailable(2));
  const canvas = useRef<HTMLCanvasElement>(null);
  const makeRef = useRef(make);
  makeRef.current = make;
  const readyRef = useRef(onReady);
  readyRef.current = onReady;

  useEffect(() => {
    const el = canvas.current;
    if (!ok || !el) return;
    let stage: Stage3D;
    let handle: ReturnType<typeof attachView>;
    try {
      stage = makeRef.current();
      handle = attachView(el, stage);
    } catch {
      setOk(false);
      return;
    }
    readyRef.current?.({ stage, markDirty: handle.markDirty });
    const seen =
      typeof IntersectionObserver === "undefined"
        ? null
        : new IntersectionObserver(([e]) => handle.setVisible(Boolean(e?.isIntersecting)));
    seen?.observe(el);
    const sized =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(handle.markDirty);
    sized?.observe(el);
    let down: { x: number; y: number } | null = null;
    const guard = createDragGuard();
    const onDown = (e: PointerEvent) => {
      down = { x: e.clientX, y: e.clientY };
      guard.down(e.clientX, e.clientY);
      el.setPointerCapture?.(e.pointerId);
      markInteracting();
      stage.pointer?.({ type: "down", dx: 0, dy: 0 });
    };
    const onMove = (e: PointerEvent) => {
      if (!down) return;
      guard.move(e.clientX, e.clientY);
      const dx = e.clientX - down.x;
      const dy = e.clientY - down.y;
      down = { x: e.clientX, y: e.clientY };
      markInteracting();
      stage.pointer?.({ type: "move", dx, dy });
      handle.markDirty();
    };
    const onUp = () => {
      if (!down) return;
      down = null;
      guard.up();
      stage.pointer?.({ type: "up", dx: 0, dy: 0 });
      handle.markDirty();
    };
    // Turning the object must not click what it sits in (the core opens its AI's window).
    const onClick = (e: MouseEvent) => {
      if (!guard.takeClick()) return;
      e.stopPropagation();
      e.preventDefault();
    };
    el.addEventListener("click", onClick, true);
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    return () => {
      el.removeEventListener("click", onClick, true);
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      seen?.disconnect();
      sized?.disconnect();
      readyRef.current?.(null);
      handle.detach();
    };
  }, [ok]);

  if (!ok) return <>{fallback}</>;
  return (
    <canvas
      ref={canvas}
      role="img"
      aria-label={label}
      className={`touch-none ${className}`}
      style={style}
    />
  );
}
