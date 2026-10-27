import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export type Line = { label: string; values: (number | null)[]; color: string };

function canDraw(): boolean {
  try {
    return !!document.createElement("canvas").getContext("2d");
  } catch {
    return false;
  }
}

/**
 * A line chart over training steps. `data-points` reports how many steps it
 * shows (tests and screen readers can't read a canvas).
 */
export function LineChart({
  testId,
  label,
  x,
  lines,
  height = 220,
}: {
  testId: string;
  label: string;
  x: number[];
  lines: Line[];
  height?: number;
}) {
  const box = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const shape = lines.map((l) => `${l.label}:${l.color}`).join("|");

  // One plot per set of lines; data updates go through setData.
  // biome-ignore lint/correctness/useExhaustiveDependencies: rebuilt only when the lines change
  useEffect(() => {
    const el = box.current;
    if (!el || !canDraw()) return;
    const p = new uPlot(
      {
        width: el.clientWidth || 600,
        height,
        legend: { show: lines.length > 1 },
        cursor: { drag: { x: false, y: false } },
        axes: [
          { stroke: "#9aa3c2", grid: { stroke: "#2a3150" }, ticks: { stroke: "#2a3150" } },
          { stroke: "#9aa3c2", grid: { stroke: "#2a3150" }, ticks: { stroke: "#2a3150" } },
        ],
        series: [
          { label: "Step" },
          ...lines.map((l) => ({ label: l.label, stroke: l.color, width: 2, spanGaps: true })),
        ],
      },
      [x, ...lines.map((l) => l.values)] as uPlot.AlignedData,
      el,
    );
    plot.current = p;
    const onResize = () => p.setSize({ width: el.clientWidth || 600, height });
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      p.destroy();
      plot.current = null;
    };
  }, [shape, height]);

  useEffect(() => {
    plot.current?.setData([x, ...lines.map((l) => l.values)] as uPlot.AlignedData);
  }, [x, lines]);

  return (
    <div ref={box} data-testid={testId} data-points={x.length} role="img" aria-label={label} />
  );
}
