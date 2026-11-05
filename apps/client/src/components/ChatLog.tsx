import type { ReactNode } from "react";

export type ChatMessage = { role: "user" | "ai"; text: string };

/**
 * A conversation. Model text is shown exactly as written, as text with its line
 * breaks (R1, spec §11); `streaming` is the reply arriving now ("" while it thinks).
 */
export function ChatLog({
  messages,
  streaming,
  aiName,
  coach,
}: {
  messages: ChatMessage[];
  streaming: string | null;
  aiName: string;
  /** Coaching under each stored AI reply that answers a message (your own AIs only). */
  coach?: (idx: number) => ReactNode;
}) {
  return (
    <ol className="space-y-3" aria-live="polite" data-testid="chat-log">
      {messages.map((m, i) => (
        <Bubble
          // biome-ignore lint/suspicious/noArrayIndexKey: messages are only ever appended, so a position never changes hands
          key={`${i}:${m.role}`}
          who={m.role === "user" ? "You" : aiName}
          mine={m.role === "user"}
          below={
            coach && m.role === "ai" && messages[i - 1]?.role === "user" ? coach(i) : undefined
          }
        >
          {m.text}
        </Bubble>
      ))}
      {streaming !== null && (
        <Bubble who={aiName} mine={false}>
          {streaming === "" ? <span className="text-muted">Thinking…</span> : streaming}
        </Bubble>
      )}
    </ol>
  );
}

function Bubble({
  who,
  mine,
  children,
  below,
}: {
  who: string;
  mine: boolean;
  children: ReactNode;
  below?: ReactNode;
}) {
  return (
    <li className={`flex flex-col ${mine ? "items-end" : "items-start"}`}>
      <span className="mb-1 text-xs text-muted">{who}</span>
      <p
        className={`player-text max-w-[80%] rounded-2xl px-4 py-2 text-sm ${mine ? "bg-brand text-on-brand" : "border border-border bg-surface-2"}`}
      >
        {children}
      </p>
      {below}
    </li>
  );
}
