import { REGION_IDS, REGION_NAMES, type RegionId } from "@airace/protocol";
import { getBalance } from "@airace/sim";
import { useState } from "react";
import { Button } from "../components/Button.tsx";
import { Computers, perCustomer } from "../components/Computers.tsx";
import { INPUT } from "../components/Field.tsx";
import { formatCount, formatMoney } from "../format.ts";
import { regionCustomers } from "../regions.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { WindowFrame, type WindowProps } from "./WindowFrame.tsx";

/** A new data center: where, what it costs and gives, and the button that builds it. */
export function DataCenterWindow({ entry, view }: WindowProps) {
  const socket = useSocket();
  const back = useWindows((s) => s.back);
  const e = getBalance(view.world.balance).economy;
  const dc = e.dataCenter;
  const busiest = [...REGION_IDS].sort(
    (a, b) => regionCustomers(view, b) - regionCustomers(view, a),
  )[0];
  const [where, setWhere] = useState<RegionId>(busiest ?? "pacific");
  const [problem, setProblem] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const user = perCustomer(view);

  async function build() {
    setProblem(null);
    setSending(true);
    const r = await socket.command(view.world.id, { type: "buildDataCenter", region: where });
    setSending(false);
    if (r.ok) back();
    else setProblem(r.reason);
  }

  const facts: [string, string][] = [
    ["Building it takes", `${dc.buildDays} game days`],
    ["Upkeep", `${formatMoney(dc.upkeepPerMonth)} a month once open`],
    [
      "It serves",
      user === null
        ? "your customers once you sell an AI"
        : `about ${formatCount(dc.pflopsPerDay / user)} customers of your biggest AI`,
    ],
    [
      "Serving there costs",
      `${Math.round(e.compute.ownedServingPriceShare * 100)}% of the cloud's price`,
    ],
  ];

  return (
    <WindowFrame
      entry={entry}
      title="Build a data center"
      subtitle="Your own computers: faster answers nearby, cheaper serving, and compute to lend to training."
    >
      <label className="block text-sm">
        <span className="mb-1 block text-xs font-semibold uppercase tracking-wider text-muted">
          Where
        </span>
        <select
          aria-label="Where"
          className={INPUT}
          value={where}
          onChange={(ev) => setWhere(ev.target.value as RegionId)}
        >
          {REGION_IDS.map((r) => (
            <option key={r} value={r}>
              {`${REGION_NAMES[r]} · ${formatCount(regionCustomers(view, r))} customers`}
            </option>
          ))}
        </select>
      </label>
      <div className="mt-5 flex items-baseline justify-between gap-3 rounded-xl border border-border bg-surface-2/60 p-4">
        <span className="text-sm text-muted">It costs</span>
        <span className="font-display text-3xl font-bold">{formatMoney(dc.cost)}</span>
      </div>
      <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
        {facts.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-muted">{k}</dt>
            <dd className="m-0">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-sm text-muted">
        Customers in {REGION_NAMES[where]} and next door get faster answers once it opens.
      </p>
      {problem && (
        <p role="alert" className="mt-3 text-sm text-danger">
          {problem}
        </p>
      )}
      <div className="mt-5 flex justify-end gap-3">
        <Button variant="secondary" onClick={back}>
          Cancel
        </Button>
        <Button onClick={() => void build()} disabled={sending}>
          Build it
        </Button>
      </div>
    </WindowFrame>
  );
}

/** The computers panel in a window, for the top bar's overload warning. */
export function ComputersWindow({ entry, view }: WindowProps) {
  return (
    <WindowFrame
      entry={entry}
      title="Computers"
      subtitle="Where your AIs run: your data centers first, then rented cloud."
    >
      <Computers view={view} />
    </WindowFrame>
  );
}
