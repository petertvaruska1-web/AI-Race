import { DEPARTMENT_NAMES, DEPARTMENTS, type Department, type WorldView } from "@airace/protocol";
import { type Balance, getBalance } from "@airace/sim";
import { useEffect, useState } from "react";
import { Panel } from "../components/Panel.tsx";
import { formatMoney } from "../format.ts";
import { useSocket } from "../store/session.ts";

type Sizes = Balance["economy"]["departments"];

const pct = (x: number) => `${Math.round(x * 100)}%`;

/** What a team does at strength `s`, in plain words (R7), from the same numbers the game uses. */
export function effectLine(d: Department, s: number, e: Sizes): string {
  switch (d) {
    case "research":
      return `Training reads ${pct(e.research * s)} more text`;
    case "data":
      return `Checks cost ${pct(e.data * s)} less`;
    case "sales":
      return `+${(e.sales * s * 100).toFixed(2)}% awareness a day`;
    case "support":
      return `Bad answers sting ${pct(e.supportSatisfaction * s)} less · ${pct(e.supportChurn * s)} fewer unhappy customers leave · serving ${pct(e.supportServing * s)} cheaper`;
  }
}

/**
 * The Team (C43): four departments with monthly budgets. A team grows toward its budget over
 * about a month, and each extra dollar does a little less than the one before.
 */
export function TeamPanel({ view }: { view: WorldView }) {
  const company = view.me.company;
  if (!company) return null;
  const sizes = getBalance(view.world.balance).economy.departments;
  return (
    <Panel
      title="Team"
      aria-label="Team"
      hint="Fund four departments. A team takes about a month to grow or shrink, and each extra dollar does a little less."
    >
      <ul className="space-y-4">
        {DEPARTMENTS.map((d) => (
          <DepartmentRow
            key={d}
            worldId={view.world.id}
            department={d}
            sizes={sizes}
            {...company.departments[d]}
          />
        ))}
      </ul>
    </Panel>
  );
}

function DepartmentRow({
  worldId,
  department,
  sizes,
  budget,
  staffed,
  strength,
}: {
  worldId: string;
  department: Department;
  sizes: Sizes;
  budget: number;
  staffed: number;
  strength: number;
}) {
  const socket = useSocket();
  const name = DEPARTMENT_NAMES[department];
  const [value, setValue] = useState(budget);
  useEffect(() => setValue(budget), [budget]);

  function commit(raw: string) {
    const next = Number(raw);
    if (next === budget) return;
    void socket.command(worldId, { type: "setDepartment", department, budget: next });
  }

  return (
    <li>
      <label className="block">
        <span className="flex justify-between gap-3 text-sm">
          <span className="font-semibold">{name}</span>
          <span className="tabular text-muted">{`${formatMoney(value)} a month`}</span>
        </span>
        <input
          type="range"
          aria-label={`${name} budget`}
          className="mt-1 w-full accent-[var(--brand)]"
          min={0}
          max={sizes.maxBudget}
          step={sizes.step}
          value={value}
          onChange={(e) => setValue(Number(e.target.value))}
          onPointerUp={(e) => commit(e.currentTarget.value)}
          onKeyUp={(e) => commit(e.currentTarget.value)}
        />
      </label>
      <div
        role="progressbar"
        aria-label={`${name} strength`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(strength * 100)}
        className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-2"
      >
        <div
          className="h-full bg-brand transition-[width] duration-700"
          style={{ width: `${Math.round(strength * 100)}%` }}
        />
      </div>
      <p className="mt-1 text-xs text-muted">
        {budget === 0 && staffed === 0 ? "No team yet." : effectLine(department, strength, sizes)}
      </p>
    </li>
  );
}
