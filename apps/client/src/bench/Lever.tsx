import { type PointerEvent, type ReactNode, useRef, useState } from "react";
import { createDragGuard } from "../three/dragGuard.ts";

/** How far the handle travels from up to down, in pixels. */
const TRAVEL = 176;

/**
 * A big lever (the founding bench's "Found company", C61): pull the handle down (drag it past
 * three quarters of its slot), click it, or press Enter or Space; `onPull` runs once. While
 * `down`, it stays pulled and does nothing more; while `disabled`, it stays up and grey.
 */
export function Lever({
  label,
  onPull,
  disabled = false,
  down = false,
  children,
}: {
  label: string;
  onPull: () => void;
  disabled?: boolean;
  down?: boolean;
  /** The status line under the lever. */
  children?: ReactNode;
}) {
  const [offset, setOffset] = useState<number | null>(null);
  const start = useRef<number | null>(null);
  const guard = useRef(createDragGuard());
  const pull = () => {
    if (!disabled && !down) onPull();
  };
  const onDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (disabled || down) return;
    start.current = e.clientY;
    guard.current.down(e.clientX, e.clientY);
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onMove = (e: PointerEvent<HTMLButtonElement>) => {
    if (start.current === null) return;
    guard.current.move(e.clientX, e.clientY);
    setOffset(Math.max(0, Math.min(TRAVEL, e.clientY - start.current)));
  };
  const onUp = () => {
    if (start.current === null) return;
    start.current = null;
    guard.current.up();
    const far = (offset ?? 0) > TRAVEL * 0.75;
    setOffset(null);
    if (far) pull();
  };
  const position = down ? TRAVEL : (offset ?? 0);
  return (
    <div className="lever">
      <div className="lever__slot">
        <button
          type="button"
          aria-label={label}
          disabled={disabled}
          className={`lever__handle ${offset !== null ? "lever__handle--dragging" : ""}`}
          style={{ transform: `translateY(${position}px)` }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={onUp}
          onClick={() => {
            // A drag already decided; a plain click (or Enter or Space) pulls.
            if (guard.current.takeClick()) return;
            pull();
          }}
        />
      </div>
      {children}
    </div>
  );
}
