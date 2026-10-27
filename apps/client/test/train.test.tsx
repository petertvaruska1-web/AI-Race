import { BALANCE_TEST } from "@airace/sim";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { TrainStep } from "../src/screens/lab/TrainStep.tsx";
import { useSession } from "../src/store/session.ts";
import { liveGame, type MockWs } from "./mockSocket.ts";
import { apply, command, runningWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
});

/** A running world where c1 has started training run r1. */
function training() {
  const plan = { ...BALANCE_TEST.defaultPlan, mix: { ...BALANCE_TEST.defaultPlan.mix } };
  return apply(runningWorld(), command("u1", { type: "startTraining", plan }, "r1"));
}

const progress = (step: number) => ({
  type: "progress",
  step,
  total_steps: 8,
  tokens: step * 8192,
  loss: 6 - step * 0.4,
  lr: 0.001,
});
const sample = (step: number, text: string) => ({
  type: "sample",
  step,
  prompt: "Tell me a story",
  text,
});

/** Answers the view's telemetry query with `events`, then waits for its stream subscription. */
async function serve(ws: MockWs, events: unknown[], nth = 1) {
  await waitFor(() =>
    expect(ws.sentOf("query").filter((q) => q.q.kind === "runTelemetry")).toHaveLength(nth),
  );
  const q = ws.sentOf("query").filter((m) => m.q.kind === "runTelemetry")[nth - 1];
  ws.receive({
    v: 1,
    type: "queryResult",
    id: q?.id,
    ok: true,
    data: { runId: "r1", totalSteps: 8, events },
  });
}

const points = (id: string) => Number(screen.getByTestId(id).getAttribute("data-points"));

describe("live training view", () => {
  it("shows the backlog, then adds live events", async () => {
    render(<TrainStep view={viewOf(training())} />);
    await serve(game.ws, [progress(1), progress(2), sample(2, "Once upon"), progress(3)]);
    await waitFor(() => expect(points("loss-chart")).toBe(3));
    await waitFor(() =>
      expect(game.ws.sentOf("subscribe").map((s) => s.sub)).toEqual(["training:w1:r1"]),
    );
    game.ws.receive({
      v: 1,
      type: "stream",
      sub: "training:w1:r1",
      data: {
        kind: "events",
        from: 4,
        totalSteps: 8,
        events: [progress(4), sample(4, "Once upon a time"), progress(5)],
      },
    });
    await waitFor(() => expect(points("loss-chart")).toBe(5));
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    // Newest first.
    expect(screen.getAllByRole("listitem")[0]?.textContent).toContain("Once upon a time");
  });

  it("shows the same backlog after a reload, without repeats", async () => {
    const backlog = [progress(1), progress(2), sample(2, "Hi"), progress(3)];
    const first = render(<TrainStep view={viewOf(training())} />);
    await serve(game.ws, backlog);
    await waitFor(() => expect(points("loss-chart")).toBe(3));
    first.unmount();

    render(<TrainStep view={viewOf(training())} />);
    await serve(game.ws, backlog, 2);
    await waitFor(() => expect(points("loss-chart")).toBe(3));
    await waitFor(() => expect(game.ws.sentOf("subscribe").length).toBeGreaterThan(1));
    // A stream message overlapping what the backlog had adds only the new event.
    game.ws.receive({
      v: 1,
      type: "stream",
      sub: "training:w1:r1",
      data: { kind: "events", from: 3, totalSteps: 8, events: [progress(3), progress(4)] },
    });
    await waitFor(() => expect(points("loss-chart")).toBe(4));
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
  });

  it("fetches again when it missed some events", async () => {
    render(<TrainStep view={viewOf(training())} />);
    await serve(game.ws, [progress(1)]);
    await waitFor(() => expect(game.ws.sentOf("subscribe")).toHaveLength(1));
    game.ws.receive({
      v: 1,
      type: "stream",
      sub: "training:w1:r1",
      data: { kind: "events", from: 3, totalSteps: 8, events: [progress(4)] },
    });
    await serve(game.ws, [progress(1), progress(2), progress(3), progress(4)], 2);
    await waitFor(() => expect(points("loss-chart")).toBe(4));
  });

  it("names each stage of a run", () => {
    let s = training();
    const waiting = render(<TrainStep view={viewOf(s)} />);
    expect(screen.getByText("Waiting for compute")).toBeTruthy();
    waiting.unmount();
    s = apply(s, { kind: "system", event: { type: "computeStarted", runId: "r1" } });
    const run = s.companies.c1?.runs.r1;
    for (let d = s.day; d < (run?.endDay ?? 0); d++) s = apply(s, { kind: "tick" });
    render(<TrainStep view={viewOf(s)} />);
    expect(screen.getByText("Finishing up")).toBeTruthy();
    expect(screen.getByText("The lab computers are still working.")).toBeTruthy();
  });

  it("shows the AI's words as text, markup included", async () => {
    const { container } = render(<TrainStep view={viewOf(training())} />);
    await serve(game.ws, [sample(1, "<script>alert(1)</script> hello")]);
    expect(await screen.findByText("<script>alert(1)</script> hello")).toBeTruthy();
    expect(container.querySelector("script")).toBeNull();
  });
});
