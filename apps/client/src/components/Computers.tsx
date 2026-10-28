import { PLAN_IDS, REGION_NAMES, type WorldView } from "@airace/protocol";
import { getBalance, paramCount } from "@airace/sim";
import { useEffect, useState } from "react";
import { dayLabel, formatCount } from "../format.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { Button } from "./Button.tsx";

/** Data-center compute lent to training (PFLOP a day): open capacity times the share. */
export function lentToTraining(view: WorldView): number {
  const c = view.me.company?.compute;
  return c ? c.ownedPflops * c.trainingShare : 0;
}

/** The compute (PFLOP a day) one customer of the company's biggest AI on sale uses; null with none on sale. */
export function perCustomer(view: WorldView): number | null {
  const company = view.me.company;
  if (!company) return null;
  const b = getBalance(view.world.balance);
  let biggest = 0;
  for (const plan of PLAN_IDS) {
    const product = company.products.find((p) => p.id === company.plans[plan].productId);
    const v = company.versions.find((x) => x.id === product?.versionId);
    if (v && product?.status === "live") {
      biggest = Math.max(biggest, paramCount(v.shape.nLayer, v.shape.dModel));
    }
  }
  if (biggest === 0) return null;
  return ((biggest / 1e6) * b.economy.servingCostPerCustomerDayPerMParam) / b.cloudPricePerPflop;
}

/** A ring that fills with the share of questions answered. */
function Answered({ served }: { served: number }) {
  const r = 34;
  const around = 2 * Math.PI * r;
  const color = served >= 1 ? "var(--good)" : served >= 0.9 ? "var(--brand-2)" : "var(--danger)";
  return (
    <div className="relative h-24 w-24 shrink-0">
      <svg viewBox="0 0 80 80" className="h-24 w-24 -rotate-90" aria-hidden="true">
        <circle cx={40} cy={40} r={r} fill="none" stroke="var(--surface-2)" strokeWidth={7} />
        <circle
          cx={40}
          cy={40}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={7}
          strokeLinecap="round"
          strokeDasharray={around}
          strokeDashoffset={around * (1 - served)}
          className="gauge"
        />
      </svg>
      <span className="absolute inset-0 grid place-items-center font-display text-xl font-bold">
        {`${Math.round(served * 100)}%`}
      </span>
    </div>
  );
}

/**
 * Where the company's AIs run (game-layer design §8): the share of questions
 * answered, room left, the share of data-center compute lent to training, and
 * the data centers, building or open.
 */
export function Computers({ view }: { view: WorldView }) {
  const socket = useSocket();
  const open = useWindows((s) => s.open);
  const company = view.me.company;
  const trainingShare = company?.compute.trainingShare ?? 0;
  const [share, setShare] = useState(Math.round(trainingShare * 100));
  useEffect(() => setShare(Math.round(trainingShare * 100)), [trainingShare]);
  if (!company) return null;
  const c = company.compute;
  const e = getBalance(view.world.balance).economy;
  const user = perCustomer(view);
  const ownedServing = c.ownedPflops * (1 - c.trainingShare);
  const cloudRoom = Math.max(
    e.compute.cloudStartPflops,
    c.cloudPflops * (1 + e.compute.cloudGrowthPerDay),
  );
  const room = ownedServing + cloudRoom;
  const fill = (x: number) => `${Math.min(100, (100 * x) / Math.max(room, c.loadPflops, 1e-12))}%`;

  function commit(value: string) {
    void socket.command(view.world.id, { type: "setTrainingShare", share: Number(value) / 100 });
  }

  return (
    <div>
      <div className="flex flex-wrap items-center gap-5">
        <Answered served={c.served} />
        <div className="min-w-48 flex-1">
          <p className="text-sm font-semibold">of questions answered</p>
          <p className="text-xs text-muted">
            {c.served >= 1
              ? "Everyone gets an answer."
              : "Too many questions: the cloud is still growing to meet them."}
          </p>
          <div className="mt-3 h-2.5 overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
            <div className="flex h-full">
              <div
                className="bar-x h-full bg-brand"
                style={{ width: fill(Math.min(c.loadPflops, room)) }}
              />
            </div>
          </div>
          <p className="mt-1 text-xs text-muted">
            {user === null
              ? "Nothing on sale, so nothing to serve."
              : `Room for about ${formatCount(Math.max(0, (room - c.loadPflops) / user))} more customers`}
          </p>
        </div>
      </div>

      <label className="mt-5 block">
        <span className="flex justify-between text-sm">
          <span className="font-semibold">Share for training</span>
          <span className="tabular text-muted">{share}%</span>
        </span>
        <input
          type="range"
          aria-label="Share for training"
          className="mt-1 w-full accent-[var(--brand)]"
          min={0}
          max={100}
          step={5}
          value={share}
          onChange={(e) => setShare(Number(e.target.value))}
          onPointerUp={(e) => commit(e.currentTarget.value)}
          onKeyUp={(e) => commit(e.currentTarget.value)}
        />
        <span className="block text-xs text-muted">
          {c.ownedPflops > 0
            ? "Of your data centers' compute: the rest serves customers."
            : "Lends nothing until one of your data centers opens."}
        </span>
      </label>

      <div className="mt-5">
        <div className="mb-2 flex items-center justify-between gap-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
            Data centers
          </h3>
          <Button
            variant="secondary"
            onClick={(ev) => open({ kind: "dataCenter" }, ev.currentTarget)}
          >
            Build a data center
          </Button>
        </div>
        {company.dataCenters.length === 0 ? (
          <p className="text-sm text-muted">None yet: everything runs on rented cloud.</p>
        ) : (
          <ul className="space-y-2">
            {company.dataCenters.map((d) => {
              const total = Math.max(1, d.openDay - d.startDay);
              const left = Math.max(0, d.openDay - view.world.day);
              return (
                <li
                  key={d.id}
                  className="rounded-xl border border-border bg-surface-2/50 p-3 text-sm"
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-semibold">{REGION_NAMES[d.region]}</span>
                    <span className="text-muted">
                      {d.open ? `Open since ${dayLabel(d.openDay)}` : `Opens in ${left} days`}
                    </span>
                  </div>
                  {!d.open && (
                    <div className="mt-2 h-1.5 rounded-full bg-surface">
                      <div
                        className="h-1.5 rounded-full bg-brand transition-[width] duration-700"
                        style={{ width: `${100 * (1 - left / total)}%` }}
                      />
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
