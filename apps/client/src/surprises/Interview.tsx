import { type NewsView, TEXT_LIMITS, type WorldView } from "@airace/protocol";
import { motion } from "motion/react";
import { type FormEvent, useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { INPUT } from "../components/Field.tsx";
import { Panel } from "../components/Panel.tsx";
import { dayLabel, formatCount } from "../format.ts";
import { useCatalog } from "../store/catalog.ts";
import { useSocket } from "../store/session.ts";
import { useWindows } from "../store/windows.ts";
import { GoneWindow, Section, WindowFrame, type WindowProps } from "../windows/WindowFrame.tsx";

const [, MAX_CHARS] = TEXT_LIMITS.chatMessage;
export const MAX_PUBLISHED = 5;

type Message = { role: "user" | "ai"; text: string };
type StreamData =
  | { kind: "chunk"; text: string }
  | { kind: "done"; text: string }
  | { kind: "failed"; reason: string };

/** A 0-1 score as a bar (interview answers). */
function ScoreBar({ score, label }: { score: number; label: string }) {
  return (
    // biome-ignore lint/a11y/useSemanticElements: a styled, animated bar; a native meter can't be drawn alike in every browser
    <div
      role="meter"
      aria-label={label}
      aria-valuenow={Math.round(100 * score)}
      aria-valuemin={0}
      aria-valuemax={100}
      className="mt-1 h-2 rounded-full bg-surface-2"
    >
      <motion.div
        className="h-2 rounded-full bg-brand"
        initial={{ width: 0 }}
        animate={{ width: `${Math.round(100 * score)}%` }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      />
    </div>
  );
}

/** The interview's story in the feed: the newest interview headline about the product. */
function headlineOf(view: WorldView, productName: string, day: number): string | null {
  const story = view.feed.find(
    (f): f is NewsView =>
      f.kind === "news" &&
      f.topic === "interview" &&
      f.day === day &&
      f.companyId === view.me.company?.id &&
      f.headline.startsWith(productName),
  );
  return story?.headline ?? null;
}

/**
 * Interview mode (game-layer design §10): the player interviews one of their AIs on sale in
 * front of the press, with prompt cards or their own questions, then ticks up to five of
 * its answers and publishes them. The server keeps the conversation, so only what the AI
 * really said is published (R1).
 */
export function InterviewMode({ view }: { view: WorldView }) {
  const socket = useSocket();
  const open = useWindows((s) => s.open);
  const catalog = useCatalog((s) => s.catalog);
  const company = view.me.company;
  const live = (company?.products ?? []).filter((p) => p.status === "live");
  const [productId, setProductId] = useState<string | null>(null);
  const product = live.find((p) => p.id === productId) ?? live[0];
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState<string | null>(null);
  const [picked, setPicked] = useState<number[]>([]);
  const [draft, setDraft] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const [published, setPublished] = useState<{ id: string; followers: number } | null>(null);
  const worldId = view.world.id;

  useEffect(() => {
    useCatalog.getState().load(socket);
  }, [socket]);

  useEffect(() => {
    if (!threadId) return;
    return socket.subscribe(`chat:${worldId}:${threadId}`, (raw) => {
      const data = raw as StreamData;
      if (data.kind === "chunk") setStreaming((s) => (s ?? "") + data.text);
      else if (data.kind === "done") {
        setStreaming(null);
        setMessages((m) => [...m, { role: "ai", text: data.text }]);
      } else {
        setStreaming(null);
        setProblem(data.reason);
      }
    });
  }, [socket, worldId, threadId]);

  if (!company || !product) {
    return (
      <Panel>
        <p className="text-sm text-muted">
          Put an AI on sale first: the press interviews AIs people can use.
        </p>
      </Panel>
    );
  }
  const version = company.versions.find((v) => v.id === product.versionId);
  const next = company.nextInterviewDay;
  const waiting = next !== null && next > view.world.day;
  const done = published ? company.interviews.find((i) => i.id === published.id) : undefined;
  const followers = view.companies.find((c) => c.id === company.id)?.followers ?? 0;

  async function ask(text: string, cardId: string | null) {
    if (!product || !version || streaming !== null) return;
    setProblem(null);
    setMessages((m) => [...m, { role: "user", text }]);
    setStreaming("");
    const r = await socket.chat(worldId, threadId, version.id, text, {
      productId: product.id,
      cardId,
    });
    if (!r.ok) {
      setStreaming(null);
      setMessages((m) => m.slice(0, -1));
      setProblem(r.reason);
      return;
    }
    const id = (r.data as { threadId?: string } | undefined)?.threadId ?? null;
    if (id && id !== threadId) setThreadId(id);
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    const text = draft.trim();
    setDraft("");
    if (text) void ask(text, null);
  }

  function toggle(idx: number) {
    setPicked((p) =>
      p.includes(idx)
        ? p.filter((x) => x !== idx)
        : p.length < MAX_PUBLISHED
          ? [...p, idx].sort((a, b) => a - b)
          : p,
    );
  }

  async function publish() {
    if (!threadId) return;
    setProblem(null);
    const r = await socket.interviewPublish(worldId, threadId, picked);
    if (!r.ok) {
      setProblem(r.reason);
      return;
    }
    const id = (r.data as { interviewId?: string } | undefined)?.interviewId;
    if (id) setPublished({ id, followers });
  }

  function restart() {
    setThreadId(null);
    setMessages([]);
    setPicked([]);
    setPublished(null);
    setProblem(null);
  }

  if (published) {
    return (
      <Panel title="Your interview" data-testid="interview-result">
        {!done ? (
          <p className="text-sm text-muted">The press is reading your answers…</p>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-muted">{done.outlet}</p>
            <p className="player-text font-display text-lg font-semibold">
              {headlineOf(view, done.productName, done.day) ??
                `${done.productName} gave an interview`}
            </p>
            <p className="text-sm">
              {`Followers: ${formatCount(Math.round(published.followers))} → ${formatCount(Math.round(followers))}`}
            </p>
            <ul className="space-y-3">
              {done.answers.map((a, k) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: published answers never change order
                <li key={k} className="rounded-lg bg-surface-2 p-3 text-sm">
                  <p className="player-text text-muted">{a.question}</p>
                  <p className="player-text mt-1">{a.reply}</p>
                  <ScoreBar score={a.score} label={`Answer ${k + 1} score`} />
                </li>
              ))}
            </ul>
            <div className="flex gap-3">
              <Button
                variant="secondary"
                onClick={(e) => open({ kind: "interview", interviewId: done.id }, e.currentTarget)}
              >
                Open the interview
              </Button>
              <Button variant="ghost" onClick={restart}>
                Back to interviews
              </Button>
            </div>
          </div>
        )}
      </Panel>
    );
  }

  const cards = catalog?.interviewCards ?? [];
  return (
    <div className="grid gap-4 md:grid-cols-[260px_1fr]">
      <aside className="space-y-3">
        <label className="block text-sm">
          <span className="mb-1 block font-semibold">Interview</span>
          <select
            className={INPUT}
            value={product.id}
            onChange={(e) => {
              setProductId(e.target.value);
              restart();
            }}
          >
            {live.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        {waiting && next !== null && (
          <p className="rounded-lg bg-brand-soft p-3 text-sm">{`Your next interview: ${dayLabel(next)}`}</p>
        )}
        <p className="text-xs font-semibold uppercase tracking-wider text-muted">Prompt cards</p>
        <ul className="flex flex-wrap gap-2">
          {cards.map((c) => (
            <li key={c.id}>
              <button
                type="button"
                disabled={streaming !== null}
                onClick={() => void ask(c.prompt, c.id)}
                className="player-text rounded-full border border-border bg-surface-2 px-3 py-1 text-left text-xs hover:border-brand"
              >
                {c.prompt}
                {c.checked && <span className="ml-1 text-muted">· has a right answer</span>}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <Panel>
        <p className="mb-3 text-xs text-muted">
          The press is listening. Tick the answers you want published, up to five.
        </p>
        <ol className="space-y-3" aria-live="polite" data-testid="interview-log">
          {messages.map((m, idx) => (
            <li
              // biome-ignore lint/suspicious/noArrayIndexKey: messages are only appended, so a position never changes hands
              key={idx}
              className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}
            >
              <span className="mb-1 text-xs text-muted">
                {m.role === "user" ? "You" : product.name}
              </span>
              <p
                className={`player-text max-w-[80%] rounded-2xl px-4 py-2 text-sm ${m.role === "user" ? "bg-brand text-on-brand" : "border border-border bg-surface-2"}`}
              >
                {m.text}
              </p>
              {m.role === "ai" && (
                <label className="mt-1 flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={picked.includes(idx)}
                    disabled={!picked.includes(idx) && picked.length >= MAX_PUBLISHED}
                    onChange={() => toggle(idx)}
                  />
                  Publish this
                </label>
              )}
            </li>
          ))}
          {streaming !== null && (
            <li className="flex flex-col items-start">
              <span className="mb-1 text-xs text-muted">{product.name}</span>
              <p className="player-text max-w-[80%] rounded-2xl border border-border bg-surface-2 px-4 py-2 text-sm">
                {streaming === "" ? <span className="text-muted">Thinking…</span> : streaming}
              </p>
            </li>
          )}
        </ol>
        {problem && (
          <p role="alert" className="mt-3 text-sm text-danger">
            {problem}
          </p>
        )}
        <form onSubmit={submit} className="mt-4 flex gap-2">
          <input
            className={INPUT}
            aria-label="Your question"
            placeholder="Ask your own question"
            value={draft}
            maxLength={MAX_CHARS}
            onChange={(e) => setDraft(e.target.value)}
          />
          <Button type="submit" variant="secondary" disabled={streaming !== null || !draft.trim()}>
            Ask
          </Button>
        </form>
        <div className="mt-4 flex items-center gap-3">
          <Button
            disabled={picked.length === 0 || streaming !== null}
            onClick={() => void publish()}
          >
            Publish interview
          </Button>
          <span className="text-xs text-muted">{`${picked.length} of ${MAX_PUBLISHED} picked`}</span>
        </div>
      </Panel>
    </div>
  );
}

/** One published interview: the outlet, its story and every answer with its score. */
export function InterviewWindow({
  entry,
  view,
  interviewId,
}: WindowProps & { interviewId: string }) {
  const done = view.me.company?.interviews.find((i) => i.id === interviewId);
  if (!done) return <GoneWindow entry={entry} />;
  return (
    <WindowFrame
      entry={entry}
      title={`${done.productName} on ${done.outlet}`}
      subtitle={dayLabel(done.day)}
    >
      <p className="player-text font-display text-lg font-semibold">
        {headlineOf(view, done.productName, done.day) ?? "Interview published"}
      </p>
      <Section title="The answers">
        <ul className="space-y-3">
          {done.answers.map((a, k) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: published answers never change order
            <li key={k} className="rounded-lg bg-surface-2 p-3 text-sm">
              <p className="player-text text-muted">{a.question}</p>
              <p className="player-text mt-1">{a.reply}</p>
              <ScoreBar score={a.score} label={`Answer ${k + 1} score`} />
              {a.checked && (
                <p className="mt-1 text-xs text-muted">This question had a right answer.</p>
              )}
            </li>
          ))}
        </ul>
      </Section>
    </WindowFrame>
  );
}
