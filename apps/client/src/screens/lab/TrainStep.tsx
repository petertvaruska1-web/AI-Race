import type { RunView, WorldView } from "@airace/protocol";
import { useEffect, useMemo, useRef, useState } from "react";
import { Card } from "../../components/Card.tsx";
import { type Line, LineChart } from "../../components/LineChart.tsx";
import { type SampleEvent, SampleFeed } from "../../components/SampleFeed.tsx";
import { percent } from "../../format.ts";
import { useSocket } from "../../store/session.ts";
import { mix } from "../../theme/palette.ts";

type TrainEvent =
  | { type: "progress"; step: number; total_steps: number; loss: number | null }
  | { type: "heldouteval"; step: number; losses: Record<string, number | null> }
  | ({ type: "sample" } & SampleEvent)
  | { type: string; step?: number };
type Telemetry = { runId: string; totalSteps: number; events: TrainEvent[] };
type StreamData =
  | { kind: "events"; from: number; events: TrainEvent[]; totalSteps: number }
  | { kind: "reset" };

const DATASET_NAMES: Record<string, string> = {
  web: "Web Crawl",
  books: "Books",
  educational: "Educational",
  conversations: "Conversations",
  code: "Code",
  reasoning: "Reasoning",
  facts: "Reference Facts",
  creative: "Creative Writing",
};

/**
 * Everything revealed so far about a run, kept current. The backlog comes from a
 * query, then the stream adds new events; `from` lets it drop events it already
 * has (a reload, a reconnect) and notice ones it missed (then it asks again).
 */
function useTelemetry(worldId: string, runId: string) {
  const socket = useSocket();
  const [events, setEvents] = useState<TrainEvent[]>([]);
  const [totalSteps, setTotalSteps] = useState(0);
  const held = useRef<TrainEvent[]>([]);

  useEffect(() => {
    let live = true;
    let off: (() => void) | null = null;
    held.current = [];
    setEvents([]);
    const load = () =>
      socket
        .query<Telemetry>({ kind: "runTelemetry", worldId, runId })
        .then((t) => {
          if (!live) return;
          held.current = t.events;
          setEvents(t.events);
          setTotalSteps(t.totalSteps);
        })
        .catch(() => {});
    void load().then(() => {
      if (!live) return;
      off = socket.subscribe(`training:${worldId}:${runId}`, (raw) => {
        const data = raw as StreamData;
        if (data.kind === "reset") {
          held.current = [];
          setEvents([]);
          void load();
          return;
        }
        const have = held.current.length;
        if (data.from > have) {
          void load(); // missed some: fetch everything revealed so far
          return;
        }
        const fresh = data.events.slice(have - data.from);
        if (fresh.length === 0) return;
        held.current = [...held.current, ...fresh];
        setEvents(held.current);
        setTotalSteps(data.totalSteps);
      });
    });
    return () => {
      live = false;
      off?.();
    };
  }, [socket, worldId, runId]);

  return { events, totalSteps };
}

export function TrainStep({ view }: { view: WorldView }) {
  const company = view.me.company;
  const run = company?.runs.at(-1);
  if (!company || !run)
    return (
      <p className="text-muted">Nothing has trained yet. Plan a run in the first three steps.</p>
    );
  return <RunPanel view={view} run={run} />;
}

function RunPanel({ view, run }: { view: WorldView; run: RunView }) {
  const { events, totalSteps } = useTelemetry(view.world.id, run.id);
  const palette = view.companies.find((c) => c.id === view.me.company?.id)?.palette;
  const version = view.me.company?.versions.find((v) => v.id === run.versionId);

  const loss = useMemo(() => {
    const steps: number[] = [];
    const values: (number | null)[] = [];
    for (const e of events) {
      if (e.type === "progress" && "loss" in e) {
        steps.push(e.step ?? 0);
        values.push(e.loss);
      }
    }
    return {
      steps,
      lines: [{ label: "How wrong", values, color: palette?.primary ?? "#8b93b8" }] as Line[],
    };
  }, [events, palette]);

  const learning = useMemo(() => {
    const evals = events.filter(
      (e): e is Extract<TrainEvent, { type: "heldouteval" }> => e.type === "heldouteval",
    );
    const names = [...new Set(evals.flatMap((e) => Object.keys(e.losses)))].sort();
    const primary = palette?.primary ?? "#8b93b8";
    const secondary = palette?.secondary ?? "#c2c8e0";
    return {
      steps: evals.map((e) => e.step),
      lines: names.map((name, i) => ({
        label: DATASET_NAMES[name] ?? name,
        values: evals.map((e) => e.losses[name] ?? null),
        // The company's palette family, from its main to its second color.
        color: mix(secondary, primary, names.length > 1 ? i / (names.length - 1) : 0),
      })),
    };
  }, [events, palette]);

  const samples = events.filter(
    (e): e is Extract<TrainEvent, { type: "sample" }> => e.type === "sample",
  );

  return (
    <div className="space-y-4">
      <Card>
        <Status run={run} label={version?.label} />
        {(run.status === "training" || run.status === "finalizing") && (
          <div className="mt-3">
            <progress
              className="h-2 w-full accent-[var(--brand)]"
              aria-label="Game time"
              value={run.progress}
              max={1}
            />
            <p className="mt-1 text-xs text-muted">
              {percent(run.progress)} of its {run.endDay - run.startDay} game days
            </p>
          </div>
        )}
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title="How wrong its guesses are"
          hint="It reads by guessing the next word. Falling means it is learning."
        >
          <LineChart
            testId="loss-chart"
            label="How wrong its guesses are"
            x={loss.steps}
            lines={loss.lines}
          />
        </Card>
        <Card
          title="What it's learning"
          hint="How well it guesses text it has never seen, for each kind of reading."
        >
          <LineChart
            testId="learning-chart"
            label="What it's learning"
            x={learning.steps}
            lines={learning.lines}
          />
        </Card>
      </div>
      <Card
        title="What it says"
        hint="Its own words while it trains, including your test question."
      >
        <SampleFeed samples={samples} totalSteps={totalSteps} />
      </Card>
    </div>
  );
}

function Status({ run, label }: { run: RunView; label: string | undefined }) {
  if (run.status === "failed") {
    return (
      <div role="status">
        <p className="text-lg font-semibold text-danger">Failed</p>
        <p className="text-sm">{run.failure}</p>
        <p className="text-sm text-muted">Your cash was refunded.</p>
      </div>
    );
  }
  if (run.status === "completed") {
    return (
      <p role="status" className="text-lg font-semibold">
        Done: meet <span className="text-brand">{label}</span>
      </p>
    );
  }
  if (run.status === "finalizing") {
    return (
      <div role="status">
        <p className="text-lg font-semibold">Finishing up</p>
        <p className="text-sm text-muted">The lab computers are still working.</p>
      </div>
    );
  }
  if (!run.computeStarted) {
    return (
      <div role="status">
        <p className="text-lg font-semibold">Waiting for compute</p>
        <p className="text-sm text-muted">
          No lab computer is free yet; your run starts as soon as one is.
        </p>
      </div>
    );
  }
  return (
    <p role="status" className="text-lg font-semibold">
      Training
    </p>
  );
}
