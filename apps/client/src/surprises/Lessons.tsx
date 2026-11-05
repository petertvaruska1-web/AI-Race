import type { WorldView } from "@airace/protocol";
import { LESSONS_MIN } from "@airace/sim";
import { useCallback, useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { usePlan } from "../store/plan.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { WindowFrame, type WindowProps } from "../windows/WindowFrame.tsx";

const MAX_ANSWER = 500;

/**
 * Coaching (C45) under one of your own AI's replies: like it, or write the answer you wished
 * for. Each becomes a lesson the next training run can learn from.
 */
export function CoachButtons({
  worldId,
  threadId,
  idx,
}: {
  worldId: string;
  threadId: string;
  idx: number;
}) {
  const socket = useSocket();
  const [writing, setWriting] = useState(false);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save(kind: "liked" | "written") {
    setBusy(true);
    setError(null);
    const r = await socket.lesson(
      worldId,
      threadId,
      idx,
      kind,
      kind === "written" ? text : undefined,
    );
    setBusy(false);
    if (r.ok) {
      setSaved(true);
      setWriting(false);
      setText("");
    } else setError(r.reason);
  }

  return (
    <div className="mt-1 space-y-1 text-xs">
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          disabled={busy}
          onClick={() => void save("liked")}
          className="rounded-full border border-border px-2 py-0.5 hover:border-brand"
        >
          👍 Good answer
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => setWriting((w) => !w)}
          className="rounded-full border border-border px-2 py-0.5 hover:border-brand"
        >
          ✏️ Write a better answer
        </button>
        {saved && <span className="text-good">Saved as a lesson.</span>}
      </div>
      {writing && (
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void save("written");
          }}
        >
          <input
            aria-label="Your answer"
            className={`${INPUT} text-xs`}
            value={text}
            maxLength={MAX_ANSWER}
            onChange={(e) => setText(e.target.value)}
          />
          <Button type="submit" variant="secondary" disabled={busy || text.trim().length === 0}>
            Save
          </Button>
        </form>
      )}
      {error && (
        <p role="alert" className="text-danger">
          {error}
        </p>
      )}
    </div>
  );
}

/** The AI Lab's "Your coaching" card: the lesson count and the switch to train on them. */
export function LessonsCard({ view }: { view: WorldView }) {
  const plan = usePlan((s) => s.plan);
  const update = usePlan((s) => s.update);
  const open = useWindows((s) => s.open);
  const n = view.me.company?.lessons ?? 0;
  const locked = n < LESSONS_MIN;
  return (
    <Panel
      title="Your coaching"
      hint="Like your AI's good answers in the Playground, or write better ones. It learns from them."
    >
      <p className="mb-2 text-sm">
        {`${n} ${n === 1 ? "lesson" : "lessons"}`}
        {n > 0 && (
          <button
            type="button"
            className="ml-3 text-brand underline"
            onClick={(e) => open({ kind: "lessons" }, e.currentTarget)}
          >
            See your lessons
          </button>
        )}
      </p>
      <label className="flex items-start gap-2 text-sm">
        <input
          type="checkbox"
          aria-label="Train on your lessons"
          checked={!locked && (plan?.coaching ?? false)}
          disabled={locked}
          onChange={(e) => update({ coaching: e.target.checked })}
        />
        <span>
          <span className="font-semibold">Train on your lessons</span>
          <span className="block text-xs text-muted">
            {locked
              ? `Your AI needs at least ${LESSONS_MIN} lessons first. You have ${n}.`
              : "Your lessons become 5% of the run's reading. Free."}
          </span>
        </span>
      </label>
    </Panel>
  );
}

type Lesson = { id: number; kind: "liked" | "written"; question: string; answer: string };

/** Every lesson, oldest first: a written one is the player's answer, never the AI's (R1). */
export function LessonsWindow({ entry, view }: WindowProps) {
  const socket = useSocket();
  const [lessons, setLessons] = useState<Lesson[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => {
    socket
      .query<{ lessons: Lesson[] }>({ kind: "lessons", worldId: view.world.id })
      .then((r) => setLessons(r.lessons))
      .catch((err: Error) => setError(err.message));
  }, [socket, view.world.id]);
  useEffect(load, [load]);

  async function remove(id: number) {
    const r = await socket.lessonDelete(view.world.id, id);
    if (r.ok) setLessons((all) => (all ?? []).filter((l) => l.id !== id));
    else setError(r.reason);
  }

  return (
    <WindowFrame
      entry={entry}
      title="Your lessons"
      subtitle="What your AI learns from when you train on your lessons."
    >
      {lessons === null ? (
        <p className="text-sm text-muted">{error ?? "Loading…"}</p>
      ) : lessons.length === 0 ? (
        <p className="text-sm text-muted">No lessons yet.</p>
      ) : (
        <ul className="divide-y divide-border">
          {lessons.map((l) => (
            <li key={l.id} className="flex items-start gap-3 py-2 text-sm">
              <div className="mr-auto min-w-0">
                <p className="player-text text-muted">{l.question}</p>
                <p className="text-xs font-semibold">
                  {l.kind === "written" ? "Your answer" : "Its answer you liked"}
                </p>
                <p className="player-text whitespace-pre-wrap break-words">{l.answer}</p>
              </div>
              <Button variant="secondary" onClick={() => void remove(l.id)}>
                Delete
              </Button>
            </li>
          ))}
        </ul>
      )}
      {error && lessons !== null && (
        <p role="alert" className="mt-2 text-sm text-danger">
          {error}
        </p>
      )}
    </WindowFrame>
  );
}
