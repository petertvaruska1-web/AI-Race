import type { DiscoveryKind, PrivateCompany } from "@airace/protocol";
import { motion } from "motion/react";
import { useState } from "react";
import { Button } from "../components/Button.tsx";
import { dayLabel } from "../format.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "../windows/WindowFrame.tsx";

type Discovery = PrivateCompany["discoveries"][number];

export const KIND_NAMES: Record<DiscoveryKind, string> = {
  strength: "Strength",
  weakness: "Weakness",
  better: "Better",
  worse: "Worse",
  changed: "Changed",
};
const GOOD: Record<DiscoveryKind, boolean> = {
  strength: true,
  better: true,
  changed: true,
  weakness: false,
  worse: false,
};

export function KindBadge({ kind }: { kind: DiscoveryKind }) {
  return (
    <span
      className="rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wider"
      style={{
        background: GOOD[kind] ? "var(--good-soft, var(--brand-soft))" : "var(--surface-2)",
        color: GOOD[kind] ? "var(--good)" : "var(--danger)",
      }}
    >
      {KIND_NAMES[kind]}
    </span>
  );
}

/** What it is compared with, in words. */
function comparedWith(d: Discovery): string {
  if (d.kind === "better" || d.kind === "worse" || d.kind === "changed") return "Before";
  return "Other AIs";
}

/** A version's discoveries as rows; each opens its window. */
export function DiscoveryList({ discoveries }: { discoveries: Discovery[] }) {
  const open = useWindows((s) => s.open);
  if (discoveries.length === 0) return null;
  return (
    <ul className="space-y-2">
      {discoveries.map((d) => (
        <li key={d.id}>
          <button
            type="button"
            onClick={(e) => open({ kind: "discovery", discoveryId: d.id }, e.currentTarget)}
            className="row-btn"
          >
            <KindBadge kind={d.kind} />
            <span className="ml-2 text-left">{d.title}</span>
            {d.published && <span className="ml-2 text-xs text-muted">Published</span>}
            <span aria-hidden="true" className="ml-auto text-muted">
              ›
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function ValueBar({ label, value, color }: { label: string; value: number; color: string }) {
  const share = Math.min(1, Math.max(0, value / 100));
  return (
    <div>
      <p className="flex justify-between text-sm">
        <span>{label}</span>
        <span className="tabular font-semibold">{Math.round(value)}</span>
      </p>
      <div className="mt-1 h-2 rounded-full bg-surface-2">
        <motion.div
          className="h-2 rounded-full"
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${100 * share}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

/**
 * One discovery: what was found, the measurement against what it was compared with, and the
 * real question and reply that prove it (verbatim, R1). Publishing makes it news.
 */
export function DiscoveryWindow({
  entry,
  view,
  discoveryId,
}: WindowProps & { discoveryId: string }) {
  const socket = useSocket();
  const [problem, setProblem] = useState<string | null>(null);
  const d = view.me.company?.discoveries.find((x) => x.id === discoveryId);
  if (!d) return <GoneWindow entry={entry} />;
  const trait = d.category === "personality";

  async function publish() {
    if (!d) return;
    setProblem(null);
    const r = await socket.command(view.world.id, {
      type: "publishDiscovery",
      discoveryId: d.id,
    });
    if (!r.ok) setProblem(r.reason);
  }

  return (
    <WindowFrame
      entry={entry}
      title={d.title}
      subtitle={`Your AI ${d.versionLabel} · ${dayLabel(d.day)}`}
    >
      <KindBadge kind={d.kind} />
      {!trait && (
        <div className="mt-4 grid gap-3">
          <ValueBar label={`Your AI ${d.versionLabel}`} value={d.value} color="var(--brand)" />
          <ValueBar label={comparedWith(d)} value={d.compared} color="var(--muted)" />
        </div>
      )}
      {d.evidence ? (
        <Section title="The proof">
          <div className="rounded-lg bg-surface-2 p-3 text-sm" data-testid="evidence">
            <p className="player-text text-muted">{d.evidence.prompt}</p>
            <p className="player-text mt-2">{d.evidence.reply}</p>
          </div>
        </Section>
      ) : null}
      {problem && (
        <p role="alert" className="mt-3 text-sm text-danger">
          {problem}
        </p>
      )}
      <div className="mt-6">
        {d.published ? (
          <p className="text-sm text-muted">Published</p>
        ) : (
          <Button onClick={() => void publish()}>Publish as news</Button>
        )}
      </div>
    </WindowFrame>
  );
}
