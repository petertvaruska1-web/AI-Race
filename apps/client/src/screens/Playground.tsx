import { TEXT_LIMITS, type VersionView, type WorldView } from "@airace/protocol";
import { type FormEvent, useCallback, useEffect, useState } from "react";
import { Button } from "../components/Button.tsx";
import { Card } from "../components/Card.tsx";
import { ChatLog, type ChatMessage } from "../components/ChatLog.tsx";
import { INPUT } from "../components/Field.tsx";
import { usePlayground } from "../store/playground.ts";
import { useSocket } from "../store/session.ts";

type Thread = { id: string; versionId: string; createdAt: number };
type ThreadData = {
  threadId: string;
  versionId: string;
  waiting: boolean;
  messages: ChatMessage[];
};
type StreamData =
  | { kind: "chunk"; text: string }
  | { kind: "done"; text: string }
  | { kind: "failed"; reason: string };

const [, MAX_CHARS] = TEXT_LIMITS.chatMessage;
export const TOO_LONG = `Keep it under ${MAX_CHARS} characters.`;

/** True when this version and every version before it read no conversations at all. */
export function neverSawConversation(view: WorldView, version: VersionView): boolean {
  const company = view.me.company;
  if (!company) return false;
  for (
    let v: VersionView | undefined = version;
    v;
    v = company.versions.find((x) => x.id === v?.parentId)
  ) {
    const run = company.runs.find((r) => r.id === v?.runId);
    if (!run || run.plan.mix.conversations > 0) return false;
  }
  return true;
}

export function Playground({ view }: { view: WorldView }) {
  const socket = useSocket();
  const versions = view.me.company?.versions ?? [];
  const chosen = usePlayground((s) => s.versionId);
  const setVersion = usePlayground((s) => s.setVersion);
  const version = versions.find((v) => v.id === chosen) ?? versions.at(-1);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [lastSent, setLastSent] = useState("");
  const worldId = view.world.id;

  const loadThreads = useCallback(() => {
    socket
      .query<{ threads: Thread[] }>({ kind: "chatThreads", worldId })
      .then((r) => setThreads(r.threads))
      .catch(() => {});
  }, [socket, worldId]);
  useEffect(loadThreads, [loadThreads]);

  // Follow the open conversation: its stored messages, then the reply as it streams in.
  useEffect(() => {
    if (!threadId) return;
    const off = socket.subscribe(`chat:${worldId}:${threadId}`, (raw) => {
      const data = raw as StreamData;
      if (data.kind === "chunk") setStreaming((s) => (s ?? "") + data.text);
      else if (data.kind === "done") {
        setStreaming(null);
        setMessages((m) => [...m, { role: "ai", text: data.text }]);
      } else {
        setStreaming(null);
        setFailure(data.reason);
      }
    });
    socket
      .query<ThreadData>({ kind: "chatThread", worldId, threadId })
      .then((t) => {
        setMessages(t.messages.map((m) => ({ role: m.role, text: m.text })));
        setStreaming((s) => (t.waiting ? (s ?? "") : null));
      })
      .catch(() => {});
    return off;
  }, [socket, worldId, threadId]);

  async function send(text: string) {
    if (!version) return;
    setFailure(null);
    setLastSent(text);
    setMessages((m) => [...m, { role: "user", text }]);
    setStreaming("");
    const r = await socket.chat(worldId, threadId, version.id, text);
    if (!r.ok) {
      setStreaming(null);
      setMessages((m) => m.slice(0, -1));
      setFailure(r.reason);
      return;
    }
    const id = (r.data as { threadId?: string } | undefined)?.threadId ?? null;
    if (id && id !== threadId) {
      setThreadId(id);
      loadThreads();
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    const text = draft;
    setDraft("");
    void send(text);
  }

  function openThread(t: Thread) {
    setFailure(null);
    setStreaming(null);
    setMessages([]);
    setThreadId(t.id);
  }

  if (!version) return <p className="text-muted">Train your first AI to talk to it here.</p>;
  const tooLong = draft.length > MAX_CHARS;
  const mine = threads.filter((t) => t.versionId === version.id);

  return (
    <div className="grid gap-4 md:grid-cols-[220px_1fr]">
      <aside className="space-y-3">
        <label className="block text-sm">
          <span className="mb-1 block font-semibold">Talk to</span>
          <select
            className={INPUT}
            value={version.id}
            onChange={(e) => {
              setVersion(e.target.value);
              setThreadId(null);
              setMessages([]);
            }}
          >
            {[...versions].reverse().map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </label>
        <Button
          variant="secondary"
          className="w-full"
          onClick={() => {
            setThreadId(null);
            setMessages([]);
            setStreaming(null);
          }}
        >
          New conversation
        </Button>
        <ul className="space-y-1 text-sm">
          {mine.map((t, i) => (
            <li key={t.id}>
              <button
                type="button"
                onClick={() => openThread(t)}
                aria-current={t.id === threadId ? "true" : undefined}
                className={`w-full rounded-lg px-3 py-1.5 text-left ${t.id === threadId ? "bg-brand-soft text-text" : "text-muted hover:text-text"}`}
              >
                Conversation {i + 1}
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <Card>
        {neverSawConversation(view, version) && (
          <p className="mb-3 rounded-lg bg-brand-soft p-3 text-sm">
            This AI never saw a conversation, so it may just continue your text.
          </p>
        )}
        {messages.length === 0 && streaming === null && (
          <p className="mb-3 text-sm text-muted">
            Say something to {version.label}. It answers in its own words.
          </p>
        )}
        <ChatLog messages={messages} streaming={streaming} aiName={version.label} />
        {failure && (
          <div role="alert" className="mt-3 flex items-center gap-3 text-sm text-danger">
            {failure}
            {lastSent && (
              <Button variant="secondary" onClick={() => void send(lastSent)}>
                Try again
              </Button>
            )}
          </div>
        )}
        <form onSubmit={submit} className="mt-4 space-y-2">
          <textarea
            className={`${INPUT} min-h-20`}
            aria-label="Your message"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                e.currentTarget.form?.requestSubmit();
              }
            }}
          />
          {tooLong && <p className="text-xs text-danger">{TOO_LONG}</p>}
          <Button
            type="submit"
            disabled={streaming !== null || tooLong || draft.trim().length === 0}
          >
            Send
          </Button>
        </form>
      </Card>
    </div>
  );
}
