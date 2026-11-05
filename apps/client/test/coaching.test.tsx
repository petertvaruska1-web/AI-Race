import { BALANCE_V2, type CompanyState } from "@airace/sim";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { ChatLog } from "../src/components/ChatLog.tsx";
import { usePlan } from "../src/store/plan.ts";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { CoachButtons, LessonsCard } from "../src/surprises/Lessons.tsx";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { runningWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.setState({ stack: [], seq: 0 });
  usePlan.setState({ plan: null, worldId: null });
});

const THREAD = "00000000-0000-4000-8000-0000000000aa";

describe("coaching in the Playground", () => {
  it("offers coaching only under the AI's replies", () => {
    render(
      <ChatLog
        messages={[
          { role: "user", text: "What is 2+2?" },
          { role: "ai", text: "Four." },
        ]}
        streaming={null}
        aiName="v1"
        coach={(idx) => <span data-testid={`coach-${idx}`} />}
      />,
    );
    expect(screen.queryByTestId("coach-0")).toBeNull();
    expect(screen.getByTestId("coach-1")).toBeTruthy();
  });

  it("a like sends exactly that reply as a lesson", async () => {
    render(<CoachButtons worldId="w1" threadId={THREAD} idx={1} />);
    fireEvent.click(screen.getByRole("button", { name: "👍 Good answer" }));
    await waitFor(() => expect(game.ws.sentOf("lesson")).toHaveLength(1));
    const sent = game.ws.sentOf("lesson")[0];
    expect(sent).toMatchObject({ worldId: "w1", threadId: THREAD, idx: 1, kind: "liked" });
    expect(sent && "text" in sent ? sent.text : undefined).toBeUndefined();
    game.ws.receive({ v: 1, type: "ack", cid: sent?.cid, duplicate: false });
    expect(await screen.findByText("Saved as a lesson.")).toBeTruthy();
  });

  it("a written answer sends its text, and a refusal shows", async () => {
    render(<CoachButtons worldId="w1" threadId={THREAD} idx={1} />);
    fireEvent.click(screen.getByRole("button", { name: "✏️ Write a better answer" }));
    fireEvent.change(screen.getByLabelText("Your answer"), { target: { value: "It is 4." } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(game.ws.sentOf("lesson").at(-1)).toMatchObject({ kind: "written", text: "It is 4." }),
    );
    game.ws.receive({
      v: 1,
      type: "reject",
      cid: game.ws.sentOf("lesson").at(-1)?.cid,
      reason: "Answers can't hold links, email addresses or phone numbers.",
    });
    expect(
      await screen.findByText("Answers can't hold links, email addresses or phone numbers."),
    ).toBeTruthy();
  });
});

describe("your coaching in the AI Lab", () => {
  it("locks the switch under 5 lessons and turns it on from 5", () => {
    const s = runningWorld("balance-v2");
    (s.companies.c1 as CompanyState).lessons = 4;
    usePlan.getState().reset(s.worldId, BALANCE_V2, null);
    const { rerender } = render(<LessonsCard view={viewOf(s)} />);
    const toggle = screen.getByLabelText("Train on your lessons") as HTMLInputElement;
    expect(toggle.disabled).toBe(true);
    expect(screen.getByText("Your AI needs at least 5 lessons first. You have 4.")).toBeTruthy();
    expect(screen.getByText("4 lessons")).toBeTruthy();
    (s.companies.c1 as CompanyState).lessons = 5;
    rerender(<LessonsCard view={viewOf(s)} />);
    fireEvent.click(screen.getByLabelText("Train on your lessons"));
    expect(usePlan.getState().plan?.coaching).toBe(true);
  });

  it("lists the lessons, labels a written one as yours, as plain text, and deletes", async () => {
    const view = viewOf(runningWorld("balance-v2"));
    useWindows.getState().open({ kind: "lessons" });
    const { container } = render(<WindowHost view={view} />);
    const q = await waitFor(() => {
      const m = game.ws.sentOf("query").find((x) => x.q.kind === "lessons");
      if (!m) throw new Error("no query yet");
      return m;
    });
    game.ws.receive({
      v: 1,
      type: "queryResult",
      id: q.id,
      ok: true,
      data: {
        lessons: [
          { id: 1, kind: "liked", question: "Hi?", answer: "Hello.", at: 0 },
          { id: 2, kind: "written", question: "What is 2+2?", answer: "<b>It is 4.</b>", at: 1 },
        ],
      },
    });
    const written = within(
      (await screen.findByText("<b>It is 4.</b>")).closest("li") as HTMLElement,
    );
    expect(written.getByText("Your answer")).toBeTruthy();
    expect(container.querySelector("li b")).toBeNull();
    fireEvent.click(written.getByRole("button", { name: "Delete" }));
    await waitFor(() =>
      expect(game.ws.sentOf("lessonDelete").at(-1)).toMatchObject({ lessonId: 2 }),
    );
  });
});
