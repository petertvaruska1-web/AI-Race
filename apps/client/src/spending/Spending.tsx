import {
  APP_KINDS,
  APP_NAMES,
  type AppKind,
  type Command,
  DATASET_IDS,
  DATASET_TITLES,
  type DatasetId,
  REGION_IDS,
  REGION_NAMES,
  type RegionId,
  SPONSOR_KINDS,
  type SponsorKind,
  type TrainingPlan,
  type WorldView,
} from "@airace/protocol";
import { formatMoney } from "@airace/sim";
import { motion } from "motion/react";
import { type ReactNode, useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { dayLabel, formatCount } from "../format.ts";
import { GROUPS } from "../groups.ts";
import { usePlan } from "../store/plan.ts";
import { useSocket } from "../store/session.ts";
import type { OpenWindow } from "../store/windows.ts";
import { useWindows } from "../store/windows.ts";
import { WindowFrame } from "../windows/WindowFrame.tsx";

// The playtest pack's things to buy (§5). Every price comes from the world (`me.company.shop`).

function useCommand(view: WorldView) {
  const socket = useSocket();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async (cmd: Command) => {
    setBusy(true);
    setError(null);
    const r = await socket.command(view.world.id, cmd);
    setBusy(false);
    if (!r.ok) setError(r.reason);
    return r.ok;
  };
  return { run, error, busy };
}

function Problem({ error }: { error: string | null }) {
  return error ? (
    <p role="alert" className="mt-2 text-sm text-danger">
      {error}
    </p>
  ) : null;
}

// -- Looks ------------------------------------------------------------------------------------

export const LOOK_NAMES: Record<string, { name: string; hint: string }> = {
  "orb:classic": { name: "Classic", hint: "The orb as it was born." },
  "orb:nebula": { name: "Nebula", hint: "A soft cloud of color drifts around it." },
  "orb:crystal": { name: "Crystal", hint: "A faceted ring catches the light." },
  "orb:plasma": { name: "Plasma", hint: "It breathes with a hot, bright glow." },
  "orb:aurora": { name: "Aurora", hint: "Northern lights circle it, slowly." },
  "orb:supernova": { name: "Supernova", hint: "Rays burst from it: impossible to miss." },
  "logo:still": { name: "Still", hint: "Your logo, standing still." },
  "logo:pulse": { name: "Pulse", hint: "Your logo beats like a heart." },
  "logo:shimmer": { name: "Shimmer", hint: "Light sweeps across it." },
  "logo:orbit": { name: "Orbit", hint: "A spark circles your logo." },
};

/** The decoration a bought orb style draws around the orb (the orb itself stays a measurement). */
export function OrbFrame({
  look,
  color,
  children,
}: {
  look: string;
  color: string;
  children: ReactNode;
}) {
  const spin = (s: number) => ({
    rotate: 360,
    transition: { duration: s, repeat: Infinity, ease: "linear" as const },
  });
  const ring = (background: string, s: number, inset = -18) => (
    <motion.span
      aria-hidden
      className="pointer-events-none absolute rounded-full"
      style={{ inset, background, mask: "radial-gradient(circle, transparent 62%, #000 64%)" }}
      animate={spin(s)}
    />
  );
  return (
    <span className="relative inline-block" data-look={look}>
      {look === "orb:nebula" && (
        <motion.span
          aria-hidden
          className="pointer-events-none absolute -inset-8 rounded-full blur-2xl"
          style={{
            background: `radial-gradient(circle, ${color}66, #a855f733 50%, transparent 70%)`,
          }}
          animate={{ scale: [1, 1.08, 1] }}
          transition={{ duration: 6, repeat: Infinity }}
        />
      )}
      {look === "orb:crystal" &&
        ring(
          `repeating-conic-gradient(${color}aa 0 10deg, #ffffff55 10deg 20deg, transparent 20deg 30deg)`,
          30,
        )}
      {look === "orb:plasma" && (
        <motion.span
          aria-hidden
          className="pointer-events-none absolute -inset-3 rounded-full"
          animate={{ boxShadow: [`0 0 30px ${color}`, `0 0 70px ${color}`, `0 0 30px ${color}`] }}
          transition={{ duration: 2.2, repeat: Infinity }}
        />
      )}
      {look === "orb:aurora" &&
        ring("conic-gradient(#22c55e, #3b82f6, #a855f7, #ec4899, #22c55e)", 14, -22)}
      {look === "orb:supernova" && (
        <>
          {ring(`repeating-conic-gradient(${color} 0 4deg, transparent 4deg 18deg)`, 8, -40)}
          <motion.span
            aria-hidden
            className="pointer-events-none absolute -inset-6 rounded-full"
            animate={{ boxShadow: [`0 0 60px #fff`, `0 0 120px ${color}`, `0 0 60px #fff`] }}
            transition={{ duration: 1.6, repeat: Infinity }}
          />
        </>
      )}
      {children}
    </span>
  );
}

/** How a logo moves when its company bought a motion. */
export function LogoMotion({ motion: m, children }: { motion?: string; children: ReactNode }) {
  if (m === "logo:pulse") {
    return (
      <motion.span
        className="inline-block"
        animate={{ scale: [1, 1.08, 1] }}
        transition={{ duration: 1.4, repeat: Infinity }}
      >
        {children}
      </motion.span>
    );
  }
  if (m === "logo:shimmer") {
    return (
      <motion.span
        className="inline-block"
        animate={{ filter: ["brightness(1)", "brightness(1.6)", "brightness(1)"] }}
        transition={{ duration: 2.4, repeat: Infinity }}
      >
        {children}
      </motion.span>
    );
  }
  if (m === "logo:orbit") {
    return (
      <span className="relative inline-block">
        {children}
        <motion.span
          aria-hidden
          className="pointer-events-none absolute inset-0"
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        >
          <span className="absolute -top-1 left-1/2 h-2 w-2 -translate-x-1/2 rounded-full bg-white shadow-[0_0_8px_#fff]" />
        </motion.span>
      </span>
    );
  }
  return <>{children}</>;
}

export function CustomizeWindow({ entry, view }: { entry: OpenWindow; view: WorldView }) {
  const company = view.me.company;
  const { run, error, busy } = useCommand(view);
  if (!company) return null;
  const pub = view.companies.find((c) => c.id === company.id);
  const color = pub?.palette.primary ?? "#4f8cff";
  const group = (prefix: string) => company.shop.looks.filter((l) => l.id.startsWith(prefix));
  const section = (title: string, prefix: "orb:" | "logo:") => (
    <section aria-label={title} className="mb-5">
      <h3 className="mb-2 font-semibold">{title}</h3>
      <ul className="grid gap-2 sm:grid-cols-2">
        {group(prefix).map((l) => {
          const owned = company.looks.owned.includes(l.id);
          const worn = company.looks.orb === l.id || company.looks.logo === l.id;
          const info = LOOK_NAMES[l.id] ?? { name: l.id, hint: "" };
          return (
            <li
              key={l.id}
              className={`rounded-xl border p-3 ${worn ? "border-brand" : "border-border"}`}
            >
              <div className="flex items-center justify-between gap-2">
                <p className="font-semibold">{info.name}</p>
                <p className="text-sm text-muted">
                  {l.price === 0 ? "Free" : formatMoney(l.price)}
                </p>
              </div>
              <p className="text-xs text-muted">{info.hint}</p>
              {prefix === "orb:" && (
                <div className="my-3 flex justify-center">
                  <OrbFrame look={l.id} color={color}>
                    <span
                      className="block h-12 w-12 rounded-full"
                      style={{ background: `radial-gradient(circle at 35% 35%, #fff, ${color})` }}
                    />
                  </OrbFrame>
                </div>
              )}
              <Button
                variant={worn ? "secondary" : "primary"}
                disabled={busy || worn}
                onClick={() =>
                  void run(
                    owned ? { type: "useLook", lookId: l.id } : { type: "buyLook", lookId: l.id },
                  )
                }
              >
                {worn ? "Wearing it" : owned ? "Wear it" : `Buy for ${formatMoney(l.price)}`}
              </Button>
            </li>
          );
        })}
      </ul>
    </section>
  );
  return (
    <WindowFrame
      entry={entry}
      title="Customize"
      subtitle="Looks change only how you look, never how your AI scores."
      wide
    >
      {section("Orb style", "orb:")}
      {section("Logo motion", "logo:")}
      <Problem error={error} />
    </WindowFrame>
  );
}

// -- Partner apps -----------------------------------------------------------------------------

const APP_HINTS: Record<AppKind, string> = {
  homework: "Helps students with their questions. Lives on your AI's knowledge.",
  stories: "Bedtime stories for families. Lives on your AI's creativity.",
  code: "A coding sidekick for developers. Lives on your AI's coding.",
  puzzles: "Pattern puzzles for students. Lives on your AI's patterns.",
  office: "Writes and follows instructions for offices. Lives on how well it follows instructions.",
};

const stars = (n: number) => "★★★★★".slice(0, Math.round(n)).padEnd(5, "☆");

export function AppsPanel({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const apps = view.me.company?.apps ?? [];
  const live = apps.filter((a) => a.status === "live");
  return (
    <Panel
      title="Partner apps"
      aria-label="Partner apps"
      hint="Small apps built on your AI. They grow with your AI's real skill, and their reviews quote it."
      actions={
        <Button onClick={(e) => open({ kind: "fundApp" }, e.currentTarget)}>Fund an app</Button>
      }
    >
      {live.length === 0 ? (
        <p className="text-sm text-muted">No apps yet. Bet on the skill your AI is best at.</p>
      ) : (
        <ul className="space-y-3">
          {live.map((a) => (
            <li key={a.id} className="rounded-xl border border-border p-3">
              <div className="flex items-center justify-between">
                <p className="font-semibold">{APP_NAMES[a.kind]}</p>
                <p role="img" className="text-sm text-yellow-400" aria-label={`${a.rating} stars`}>
                  {stars(a.rating)}
                </p>
              </div>
              <p className="text-xs text-muted">{`${formatCount(a.users)} users · runs on ${a.productName}`}</p>
              {a.reviews.at(-1) && (
                <blockquote className="player-text mt-2 border-l-2 border-border pl-2 text-sm">
                  {a.reviews.at(-1)?.reply}
                </blockquote>
              )}
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export function FundAppWindow({ entry, view }: { entry: OpenWindow; view: WorldView }) {
  const company = view.me.company;
  const { run, error, busy } = useCommand(view);
  const back = useWindows((s) => s.back);
  const products = company?.products.filter((p) => p.status === "live") ?? [];
  const [productId, setProductId] = useState(products[0]?.id ?? "");
  if (!company) return null;
  const running = new Set(company.apps.filter((a) => a.status === "live").map((a) => a.kind));
  return (
    <WindowFrame
      entry={entry}
      title="Fund a partner app"
      subtitle="Users come as your AI's measured skill and the app's stars allow."
      wide
    >
      {products.length === 0 ? (
        <p className="text-sm text-muted">Put an AI on sale first: apps are built on it.</p>
      ) : (
        <>
          <label className="mb-4 block text-sm">
            <span className="mb-1 block font-semibold">Built on</span>
            <select
              className={INPUT}
              value={productId}
              onChange={(e) => setProductId(e.target.value)}
            >
              {products.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <ul className="grid gap-2 sm:grid-cols-2">
            {APP_KINDS.map((kind) => {
              const k = company.shop.apps.find((x) => x.kind === kind);
              if (!k) return null;
              return (
                <li key={kind} className="rounded-xl border border-border p-3">
                  <p className="font-semibold">{APP_NAMES[kind]}</p>
                  <p className="text-xs text-muted">{APP_HINTS[kind]}</p>
                  <p className="mt-1 text-xs">{`For ${GROUPS[k.group].many.toLowerCase()} · ${formatMoney(k.setup)} to set up, ${formatMoney(k.upkeep)} a month`}</p>
                  <Button
                    className="mt-2"
                    disabled={busy || running.has(kind)}
                    onClick={async () => {
                      if (await run({ type: "fundApp", kind, productId })) back();
                    }}
                  >
                    {running.has(kind) ? "Running" : `Fund it`}
                  </Button>
                </li>
              );
            })}
          </ul>
        </>
      )}
      <Problem error={error} />
    </WindowFrame>
  );
}

// -- Sponsorships -----------------------------------------------------------------------------

export const SPONSOR_WORDS: Record<SponsorKind, { name: string; icon: string }> = {
  stadium: { name: "A stadium", icon: "🏟️" },
  conference: { name: "A tech conference", icon: "🎤" },
  festival: { name: "An arts festival", icon: "🎪" },
};

export function SponsorPanel({ view }: { view: WorldView }) {
  const open = useWindows((s) => s.open);
  const running = (view.me.company?.sponsorships ?? []).filter((x) => view.world.day < x.endDay);
  return (
    <Panel
      title="Sponsorships"
      aria-label="Sponsorships"
      hint="Your name on a big event: its crowd hears of you every day it runs."
      actions={
        <Button onClick={(e) => open({ kind: "sponsor" }, e.currentTarget)}>
          Sponsor an event
        </Button>
      }
    >
      {running.length === 0 ? (
        <p className="text-sm text-muted">No sponsorships running.</p>
      ) : (
        <ul className="space-y-2 text-sm">
          {running.map((x) => (
            <li
              key={x.id}
            >{`${SPONSOR_WORDS[x.kind].icon} ${SPONSOR_WORDS[x.kind].name} in the ${REGION_NAMES[x.region]} · until ${dayLabel(x.endDay)}`}</li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export function SponsorWindow({ entry, view }: { entry: OpenWindow; view: WorldView }) {
  const company = view.me.company;
  const { run, error, busy } = useCommand(view);
  const back = useWindows((s) => s.back);
  const [region, setRegion] = useState<RegionId>("pacific");
  if (!company) return null;
  return (
    <WindowFrame entry={entry} title="Sponsor an event" wide>
      <label className="mb-4 block text-sm">
        <span className="mb-1 block font-semibold">Where</span>
        <select
          aria-label="Where"
          className={INPUT}
          value={region}
          onChange={(e) => setRegion(e.target.value as RegionId)}
        >
          {REGION_IDS.map((r) => (
            <option key={r} value={r}>
              {REGION_NAMES[r]}
            </option>
          ))}
        </select>
      </label>
      <ul className="grid gap-2 sm:grid-cols-3">
        {SPONSOR_KINDS.map((kind) => {
          const k = company.shop.sponsorships.find((x) => x.kind === kind);
          if (!k) return null;
          return (
            <li key={kind} className="rounded-xl border border-border p-3">
              <p className="text-3xl">{SPONSOR_WORDS[kind].icon}</p>
              <p className="font-semibold">{SPONSOR_WORDS[kind].name}</p>
              <p className="text-xs text-muted">{`Reaches ${k.groups.map((g) => GROUPS[g].many.toLowerCase()).join(" and ")} for ${k.days} days`}</p>
              <Button
                className="mt-2"
                disabled={busy}
                onClick={async () => {
                  if (await run({ type: "sponsor", region, kind })) back();
                }}
              >
                {`Sponsor for ${formatMoney(k.price)}`}
              </Button>
            </li>
          );
        })}
      </ul>
      <Problem error={error} />
    </WindowFrame>
  );
}

// -- The investment fund ----------------------------------------------------------------------

export function FundPanel({ view }: { view: WorldView }) {
  const company = view.me.company;
  const { run, error, busy } = useCommand(view);
  const [amount, setAmount] = useState(100_000);
  if (!company) return null;
  return (
    <Panel
      title="Investment fund"
      aria-label="Investment fund"
      hint="Park spare cash: it usually grows a little each month, and sometimes dips. A shortfall at a month's end is covered from it."
    >
      <p className="font-display text-2xl font-bold" data-testid="fund">
        {formatMoney(company.fund.balance)}
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <input
          type="number"
          aria-label="Amount"
          min={1}
          step={10_000}
          className={`${INPUT} w-40`}
          value={amount}
          onChange={(e) => setAmount(Math.max(1, Math.round(Number(e.target.value) || 0)))}
        />
        <Button disabled={busy} onClick={() => void run({ type: "invest", amount })}>
          Invest
        </Button>
        <Button
          variant="secondary"
          disabled={busy || company.fund.balance <= 0}
          onClick={() => void run({ type: "withdraw", amount })}
        >
          Withdraw
        </Button>
      </div>
      <Problem error={error} />
    </Panel>
  );
}

// -- Training extras --------------------------------------------------------------------------

/** Tutoring packs and Learn from customers, in the AI Lab's design step. */
export function TrainingExtras({ plan, loved }: { plan: TrainingPlan; loved: number }) {
  const update = usePlan((s) => s.update);
  const packs = plan.tutoring?.packs ?? 0;
  return (
    <Panel title="Extra practice" hint="Paid extras that change what your AI really reads.">
      <div className="space-y-4">
        <div>
          <p className="text-sm font-semibold">Tutoring packs</p>
          <p className="text-xs text-muted">
            Each pack adds 10% more reading from one dataset, on top of the length.
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <select
              aria-label="Tutoring dataset"
              className={`${INPUT} w-auto`}
              value={plan.tutoring?.dataset ?? "code"}
              onChange={(e) =>
                update({
                  tutoring: packs > 0 ? { dataset: e.target.value as DatasetId, packs } : null,
                })
              }
            >
              {DATASET_IDS.map((d) => (
                <option key={d} value={d}>
                  {DATASET_TITLES[d]}
                </option>
              ))}
            </select>
            <input
              type="range"
              min={0}
              max={5}
              step={1}
              aria-label="Tutoring packs"
              value={packs}
              onChange={(e) => {
                const n = Number(e.target.value);
                update({
                  tutoring: n > 0 ? { dataset: plan.tutoring?.dataset ?? "code", packs: n } : null,
                });
              }}
              className="accent-[var(--brand)]"
            />
            <span className="text-sm">
              {packs === 0 ? "None" : `${packs} pack${packs > 1 ? "s" : ""}`}
            </span>
          </div>
        </div>
        <label className="flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            checked={plan.learnFromCustomers}
            disabled={loved < 10}
            onChange={(e) => update({ learnFromCustomers: e.target.checked })}
          />
          <span>
            <span className="font-semibold">Learn from customers</span>
            <span className="block text-xs text-muted">
              {loved < 10
                ? `Customers have loved ${loved} of your AI's answers. It needs 10 to learn from them.`
                : `Read the ${loved} conversations customers loved most, as 5% of the run.`}
            </span>
          </span>
        </label>
      </div>
    </Panel>
  );
}
