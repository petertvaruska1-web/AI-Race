/** The gauge's sweep, in degrees either side of straight up. */
const SWEEP = 110;
const CX = 60;
const CY = 62;
const R = 46;

/** A point on the gauge's face at `deg` degrees from straight up, `r` from its hub. */
function at(deg: number, r: number): [number, number] {
  const a = ((deg - 90) * Math.PI) / 180;
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}

/** The arc of the face from `from` to `to` degrees. */
function arc(from: number, to: number): string {
  const [x0, y0] = at(from, R);
  const [x1, y1] = at(to, R);
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${R} ${R} 0 ${to - from > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

/**
 * A needle gauge (design §10, Benchmarks): a dark dial with ticks, a lamp-lit arc up to the
 * value and a needle pointing at it, the value in the dot face under it with its change. Says
 * "Not measured" (and rests its needle) when there is no value. A meter to assistive technology.
 */
export function NeedleGauge({
  label,
  value,
  max = 100,
  change,
  testId,
}: {
  label: string;
  value: number | null | undefined;
  max?: number;
  /** The change since the version before, shown as +n or −n. */
  change?: number | null;
  testId?: string;
}) {
  const measured = value !== null && value !== undefined;
  const v = measured ? Math.min(max, Math.max(0, value)) : 0;
  const angle = -SWEEP + (2 * SWEEP * v) / max;
  const delta = measured && change != null ? Math.round(change) : 0;
  return (
    // biome-ignore lint/a11y/useSemanticElements: the gauge draws its own dial and reading, which a native meter cannot hold
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={measured ? Math.round(value) : undefined}
      aria-valuetext={measured ? undefined : "Not measured"}
      className={`display gauge ${measured ? "" : "gauge--idle"}`}
      data-testid={testId}
    >
      <svg viewBox="0 0 120 70" aria-hidden="true" className="gauge__face">
        <path d={arc(-SWEEP, SWEEP)} className="gauge__track" />
        {measured && v > 0 && <path d={arc(-SWEEP, angle)} className="gauge__lit" />}
        {Array.from({ length: 11 }, (_, i) => {
          const deg = -SWEEP + (2 * SWEEP * i) / 10;
          const [x0, y0] = at(deg, i % 5 === 0 ? R - 11 : R - 7);
          const [x1, y1] = at(deg, R - 3);
          return (
            <line
              // biome-ignore lint/suspicious/noArrayIndexKey: the ticks are fixed
              key={i}
              x1={x0}
              y1={y0}
              x2={x1}
              y2={y1}
              className={i % 5 === 0 ? "gauge__tick gauge__tick--major" : "gauge__tick"}
            />
          );
        })}
        <g data-needle transform={`rotate(${angle.toFixed(2)} ${CX} ${CY})`}>
          <line x1={CX} y1={CY + 5} x2={CX} y2={CY - R + 8} className="gauge__needle" />
        </g>
        <circle cx={CX} cy={CY} r={4.5} className="gauge__hub" />
      </svg>
      <div className="gauge__reading">
        {measured ? (
          <span className="dotface dotface--md">{Math.round(value)}</span>
        ) : (
          <span className="gauge__none">Not measured</span>
        )}
        {delta !== 0 && (
          <span
            className={`gauge__change ${delta > 0 ? "gauge__change--up" : "gauge__change--down"}`}
          >
            {delta > 0 ? `+${delta}` : `−${-delta}`}
          </span>
        )}
      </div>
    </div>
  );
}
