import type { Catalog, Discovery } from "@airace/protocol";
import { BALANCE_TEST, type WorldState } from "@airace/sim";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import sampleCatalog from "../../../packages/protocol/fixtures/catalog.sample.json";
import { Toasts } from "../src/components/Toasts.tsx";
import { Company } from "../src/screens/Company.tsx";
import { DataStep } from "../src/screens/lab/DataStep.tsx";
import { useCatalog } from "../src/store/catalog.ts";
import { usePlan } from "../src/store/plan.ts";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { InterviewMode } from "../src/surprises/Interview.tsx";
import { PARROT_WARNING } from "../src/surprises/Notebook.tsx";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { apply, launchedWorld, SCORES, trainedWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;
beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.getState().closeAll();
  useCatalog.setState({ catalog: sampleCatalog as Catalog, error: null, loading: false });
  usePlan.setState({
    plan: { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } },
  } as never);
});

const THREAD = "22222222-2222-4222-8222-222222222222";
const HOSTILE = '<img src=x onerror="alert(1)"> <b>bold</b>';

const sys = (s: WorldState, event: unknown) => apply(s, { kind: "system", event } as never);

describe("interview mode", () => {
  it("a prompt card asks the AI on sale, then ticked answers are published exactly", async () => {
    const view = viewOf(launchedWorld());
    render(<InterviewMode view={view} />);
    const card = sampleCatalog.interviewCards[1];
    await userEvent.click(screen.getByRole("button", { name: new RegExp(card?.prompt ?? "") }));
    await waitFor(() => expect(game.ws.sentOf("chat")).toHaveLength(1));
    const chat = game.ws.sentOf("chat")[0];
    expect(chat).toMatchObject({
      versionId: "v1",
      text: card?.prompt,
      interview: { productId: "p1", cardId: card?.id },
    });
    game.ws.receive({
      v: 1,
      type: "ack",
      cid: chat?.cid,
      duplicate: false,
      data: { threadId: THREAD },
    });
    await waitFor(() =>
      expect(game.ws.sentOf("subscribe").map((s) => s.sub)).toContain(`chat:w1:${THREAD}`),
    );
    act(() =>
      game.ws.receive({
        v: 1,
        type: "stream",
        sub: `chat:w1:${THREAD}`,
        data: { kind: "done", text: HOSTILE },
      }),
    );
    // The AI's answer is shown as text, markup and all (R1).
    expect(await screen.findByText(HOSTILE)).toBeTruthy();
    expect(document.querySelector("img")).toBeNull();
    await userEvent.click(screen.getByRole("checkbox", { name: "Publish this" }));
    await userEvent.click(screen.getByRole("button", { name: "Publish interview" }));
    await waitFor(() => expect(game.ws.sentOf("interviewPublish")).toHaveLength(1));
    expect(game.ws.sentOf("interviewPublish")[0]).toMatchObject({
      worldId: "w1",
      threadId: THREAD,
      idxs: [1],
    });
  });

  it("shows a refusal, and the result card once the world has the interview", async () => {
    let s = launchedWorld();
    const { rerender } = render(<InterviewMode view={viewOf(s)} />);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "Hi?" } });
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    const chat = game.ws.sentOf("chat")[0];
    game.ws.receive({
      v: 1,
      type: "ack",
      cid: chat?.cid,
      duplicate: false,
      data: { threadId: THREAD },
    });
    await waitFor(() => expect(game.ws.sentOf("subscribe")).toHaveLength(1));
    act(() =>
      game.ws.receive({
        v: 1,
        type: "stream",
        sub: `chat:w1:${THREAD}`,
        data: { kind: "done", text: "Hello!" },
      }),
    );
    await userEvent.click(await screen.findByRole("checkbox", { name: "Publish this" }));
    await userEvent.click(screen.getByRole("button", { name: "Publish interview" }));
    await waitFor(() => expect(game.ws.sentOf("interviewPublish")).toHaveLength(1));
    const first = game.ws.sentOf("interviewPublish")[0];
    game.ws.receive({ v: 1, type: "reject", cid: first?.cid, reason: "That AI isn't on sale." });
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      "That AI isn't on sale.",
    );

    await userEvent.click(screen.getByRole("button", { name: "Publish interview" }));
    await waitFor(() => expect(game.ws.sentOf("interviewPublish")).toHaveLength(2));
    const second = game.ws.sentOf("interviewPublish")[1];
    game.ws.receive({
      v: 1,
      type: "ack",
      cid: second?.cid,
      duplicate: false,
      data: { interviewId: "i1" },
    });
    expect(await screen.findByText("The press is reading your answers…")).toBeTruthy();
    s = sys(s, {
      type: "interviewScored",
      interviewId: "i1",
      productId: "p1",
      answers: [{ question: "Hi?", reply: "Hello!", score: 0.9, checked: false }],
    });
    rerender(<InterviewMode view={viewOf(s)} />);
    const result = await screen.findByTestId("interview-result");
    expect(within(result).getByText(/^Spark wows in its interview with /)).toBeTruthy();
    expect(within(result).getByText("Hello!")).toBeTruthy();
  });
});

describe("an interview that can't be scored", () => {
  it("tells the player, and lets them publish again", async () => {
    render(<InterviewMode view={viewOf(launchedWorld())} />);
    fireEvent.change(screen.getByLabelText("Your question"), { target: { value: "Hi?" } });
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    const chat = game.ws.sentOf("chat")[0];
    game.ws.receive({
      v: 1,
      type: "ack",
      cid: chat?.cid,
      duplicate: false,
      data: { threadId: THREAD },
    });
    await waitFor(() => expect(game.ws.sentOf("subscribe")).toHaveLength(1));
    act(() =>
      game.ws.receive({
        v: 1,
        type: "stream",
        sub: `chat:w1:${THREAD}`,
        data: { kind: "done", text: "Hello!" },
      }),
    );
    await userEvent.click(await screen.findByRole("checkbox", { name: "Publish this" }));
    await userEvent.click(screen.getByRole("button", { name: "Publish interview" }));
    await waitFor(() => expect(game.ws.sentOf("interviewPublish")).toHaveLength(1));
    const sent = game.ws.sentOf("interviewPublish")[0];
    game.ws.receive({
      v: 1,
      type: "ack",
      cid: sent?.cid,
      duplicate: false,
      data: { interviewId: "i1" },
    });
    expect(await screen.findByText("The press is reading your answers…")).toBeTruthy();
    act(() =>
      game.ws.receive({
        v: 1,
        type: "event",
        worldId: "w1",
        kind: "interview",
        text: "The interview couldn't be scored. Try publishing again.",
      }),
    );
    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      "The interview couldn't be scored. Try publishing again.",
    );
    expect(screen.getByRole("button", { name: "Publish interview" })).toBeTruthy();
  });
});

const finding = (over: Partial<Discovery> = {}): Discovery => ({
  key: "knowledge/topic:animals",
  kind: "strength",
  about: "facts about animals",
  title: "Surprisingly good at facts about animals",
  category: "knowledge",
  tag: "topic:animals",
  value: 80,
  compared: 30,
  evidence: { prompt: "Which group does the newt belong to?", reply: HOSTILE },
  ...over,
});

describe("discoveries, the notebook and the trophy shelf", () => {
  it("a finding opens with its evidence verbatim, and Publish sends the command", async () => {
    const s = sys(trainedWorld(SCORES), {
      type: "discoveryFound",
      versionId: "v1",
      discoveries: [finding()],
    });
    render(<WindowHost view={viewOf(s)} />);
    act(() => useWindows.getState().open({ kind: "ai", versionId: "v1" }));
    const ai = await screen.findByRole("dialog", { name: /Your AI v1/ });
    await userEvent.click(
      within(ai).getByRole("button", { name: /Surprisingly good at facts about animals/ }),
    );
    const w = await screen.findByRole("dialog", { name: /Surprisingly good/ });
    expect(within(w).getByTestId("evidence").textContent).toContain(HOSTILE);
    expect(document.querySelector("img")).toBeNull();
    await userEvent.click(within(w).getByRole("button", { name: "Publish as news" }));
    await waitFor(() =>
      expect(game.ws.sentOf("command").map((c) => c.cmd)).toContainEqual({
        type: "publishDiscovery",
        discoveryId: "v1-d0",
      }),
    );
  });

  it("a new finding arrives as a toast", async () => {
    const before = trainedWorld(SCORES);
    const { rerender } = render(<Toasts view={viewOf(before)} />);
    const after = sys(before, {
      type: "discoveryFound",
      versionId: "v1",
      discoveries: [finding()],
    });
    rerender(<Toasts view={viewOf(after)} />);
    expect(
      await screen.findByText("Your AI has a surprise: Surprisingly good at facts about animals"),
    ).toBeTruthy();
  });

  it("the notebook card locks its slider at 0 words, warns about a big share, and saves", async () => {
    const empty = viewOf(trainedWorld(SCORES));
    const plan = usePlan.getState().plan;
    if (!plan) throw new Error("no plan");
    const { rerender } = render(
      <DataStep plan={plan} catalog={sampleCatalog as Catalog} view={empty} />,
    );
    const card = screen.getByTestId("dataset-notebook");
    expect(within(card).getByLabelText("Your notebook share")).toHaveProperty("disabled", true);
    const written = viewOf(
      sys(trainedWorld(SCORES), {
        type: "notebookSaved",
        companyId: "c1",
        hash: "a".repeat(64),
        words: 42,
      }),
    );
    usePlan.getState().setWeight("notebook", 60);
    rerender(
      <DataStep
        plan={usePlan.getState().plan ?? plan}
        catalog={sampleCatalog as Catalog}
        view={written}
      />,
    );
    expect(within(card).getByTestId("notebook-words").textContent).toBe("42 / 5,000 words");
    expect(within(card).getByLabelText("Your notebook share")).toHaveProperty("disabled", false);
    expect(within(card).getByText(PARROT_WARNING)).toBeTruthy();

    render(<WindowHost view={written} />);
    act(() => useWindows.getState().open({ kind: "notebook" }));
    await waitFor(() => expect(game.ws.sentOf("query").map((q) => q.q.kind)).toContain("notebook"));
    const q = game.ws.sentOf("query").find((x) => x.q.kind === "notebook");
    game.ws.receive({ v: 1, type: "queryResult", id: q?.id, ok: true, data: { text: "Old text" } });
    const area = await screen.findByLabelText("Notebook text");
    fireEvent.change(area, { target: { value: "Cats are mammals.\n\nDogs too." } });
    expect(screen.getByText("5 / 5,000 words")).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(game.ws.sentOf("notebookSave")).toHaveLength(1));
    const saved = game.ws.sentOf("notebookSave")[0];
    expect(saved).toMatchObject({ worldId: "w1", text: "Cats are mammals.\n\nDogs too." });
    game.ws.receive({
      v: 1,
      type: "reject",
      cid: saved?.cid,
      reason: "Notebooks hold up to 5,000 words.",
    });
    expect(await screen.findByText("Notebooks hold up to 5,000 words.")).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(game.ws.sentOf("notebookSave")).toHaveLength(2));
    game.ws.receive({
      v: 1,
      type: "ack",
      cid: game.ws.sentOf("notebookSave")[1]?.cid,
      duplicate: false,
    });
    expect(await screen.findByText("Saved. Only your company's AI reads this.")).toBeTruthy();
  });

  it("the shelf shows only earned milestones, and a new one bursts once", async () => {
    const before = launchedWorld();
    render(<Company view={viewOf(trainedWorld())} />);
    expect(screen.queryByTestId("trophy-shelf")).toBeNull(); // nothing earned: no shelf at all
    const earned = structuredClone(before);
    earned.companies.c1?.milestones.push({ id: "dataCenter", day: earned.day, detail: null });
    const { rerender } = render(<Toasts view={viewOf(before)} />);
    rerender(<Toasts view={viewOf(earned)} />);
    expect(await screen.findAllByText(/A data center opened/)).toHaveLength(1);
    rerender(<Toasts view={viewOf(earned)} />);
    expect(screen.getAllByText(/A data center opened/)).toHaveLength(1);
    render(<Company view={viewOf(earned)} />);
    const shelf = screen.getByTestId("trophy-shelf");
    const shown = within(shelf)
      .getAllByRole("listitem")
      .map((li) => li.textContent);
    expect(shown).toHaveLength(viewOf(earned).me.company?.milestones.length ?? 0);
    expect(shown.some((t) => t?.includes("A data center opened"))).toBe(true);
  });
});
