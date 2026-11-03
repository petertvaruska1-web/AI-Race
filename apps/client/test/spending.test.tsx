import type { Catalog } from "@airace/protocol";
import { BALANCE_TEST } from "@airace/sim";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import sampleCatalog from "../../../packages/protocol/fixtures/catalog.sample.json";
import { LaunchDialog } from "../src/screens/LaunchDialog.tsx";
import { FundPanel, TrainingExtras } from "../src/spending/Spending.tsx";
import { useCatalog } from "../src/store/catalog.ts";
import { usePlan } from "../src/store/plan.ts";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, SCORES, trainedWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;
beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.getState().closeAll();
  useCatalog.setState({ catalog: sampleCatalog as Catalog, error: null, loading: false });
});

const commands = () => game.ws.sentOf("command").map((c) => c.cmd);

describe("things to buy", () => {
  it("Customize: every orb style and logo motion with its price; buying one sends it", async () => {
    const view = viewOf(trainedWorld(SCORES));
    render(<WindowHost view={view} />);
    useWindows.getState().open({ kind: "customize" });
    const w = await screen.findByRole("dialog", { name: /Customize/ });
    const orbs = within(w).getByRole("region", { name: "Orb style" });
    expect(within(orbs).getByText("Supernova")).toBeTruthy();
    expect(within(orbs).getAllByText("Wearing it")).toHaveLength(1);
    await userEvent.click(within(orbs).getByRole("button", { name: "Buy for $100,000" }));
    await waitFor(() =>
      expect(commands()).toContainEqual({ type: "buyLook", lookId: "orb:nebula" }),
    );
  });

  it("the fund: shows its balance and invests the amount typed", async () => {
    render(<FundPanel view={viewOf(trainedWorld(SCORES))} />);
    expect(screen.getByTestId("fund").textContent).toBe("$0");
    fireEvent.change(screen.getByLabelText("Amount"), { target: { value: "25000" } });
    await userEvent.click(screen.getByRole("button", { name: "Invest" }));
    await waitFor(() => expect(commands()).toContainEqual({ type: "invest", amount: 25_000 }));
  });

  it("partner apps need an AI on sale, then fund the chosen kind on it", async () => {
    render(<WindowHost view={viewOf(trainedWorld(SCORES))} />);
    useWindows.getState().open({ kind: "fundApp" });
    expect(await screen.findByText("Put an AI on sale first: apps are built on it.")).toBeTruthy();
  });

  it("partner apps: a live AI can be bet on", async () => {
    const view = viewOf(launchedWorld());
    render(<WindowHost view={view} />);
    useWindows.getState().open({ kind: "fundApp" });
    const w = await screen.findByRole("dialog", { name: /Fund a partner app/ });
    await userEvent.click(within(w).getAllByRole("button", { name: "Fund it" })[0] as HTMLElement);
    await waitFor(() => expect(commands()[0]).toMatchObject({ type: "fundApp", kind: "homework" }));
  });

  it("sponsorships: pick a region and an event", async () => {
    render(<WindowHost view={viewOf(trainedWorld(SCORES))} />);
    useWindows.getState().open({ kind: "sponsor" });
    const w = await screen.findByRole("dialog", { name: /Sponsor an event/ });
    fireEvent.change(within(w).getByLabelText("Where"), { target: { value: "mountain" } });
    await userEvent.click(within(w).getByRole("button", { name: "Sponsor for $150,000" }));
    await waitFor(() =>
      expect(commands()).toContainEqual({
        type: "sponsor",
        region: "mountain",
        kind: "conference",
      }),
    );
  });

  const current = () => {
    const p = usePlan.getState().plan;
    if (!p) throw new Error("no plan");
    return p;
  };

  it("training extras: tutoring packs and Learn from customers (only with enough loved answers)", async () => {
    usePlan.setState({ plan: { ...BALANCE_TEST.defaultPlan } } as never);
    const { rerender } = render(<TrainingExtras plan={current()} loved={3} />);
    expect(screen.getByRole("checkbox")).toHaveProperty("disabled", true);
    expect(screen.getByText(/Customers have loved 3/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText("Tutoring packs"), { target: { value: "2" } });
    expect(usePlan.getState().plan?.tutoring).toEqual({ dataset: "code", packs: 2 });
    rerender(<TrainingExtras plan={current()} loved={40} />);
    expect(screen.getByRole("checkbox")).toHaveProperty("disabled", false);
  });

  it("a launch can book a keynote with a live demo question", async () => {
    render(<LaunchDialog view={viewOf(trainedWorld(SCORES))} onClose={() => {}} />);
    await userEvent.type(screen.getByLabelText(/name/i), "Spark");
    await userEvent.click(screen.getByRole("radio", { name: "What is the capital of France?" }));
    await userEvent.click(screen.getByRole("button", { name: "Launch" }));
    await waitFor(() =>
      expect(commands()[0]).toMatchObject({
        type: "launchProduct",
        keynote: { questionId: "knowledge-customer-0003" },
      }),
    );
  });
});
