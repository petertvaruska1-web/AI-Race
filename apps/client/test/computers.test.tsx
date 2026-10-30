import type { Catalog, Command } from "@airace/protocol";
import { BALANCE_V2, finance, previewPlan, type WorldState } from "@airace/sim";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import sampleCatalog from "../../../packages/protocol/fixtures/catalog.sample.json";
import { formatMoney } from "../src/format.ts";
import { Lab } from "../src/screens/lab/LabLayout.tsx";
import { Money } from "../src/screens/Money.tsx";
import { useCatalog } from "../src/store/catalog.ts";
import { usePlan } from "../src/store/plan.ts";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, runningWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useCatalog.setState({ catalog: sampleCatalog as Catalog, error: null, loading: false });
  usePlan.setState({ plan: null, worldId: null });
  useWindows.getState().closeAll();
});

const commands = () => game.ws.sentOf("command").map((m) => m.cmd as Command);

function money(s: WorldState) {
  const view = viewOf(s);
  render(
    <>
      <Money view={view} />
      <WindowHost view={view} />
    </>,
  );
  return view;
}

/** `s` with c1 owning `centers` (building unless `open`) and lending `share` to training. */
function withCenters(s: WorldState, open: boolean, share = 0): WorldState {
  const next = structuredClone(s);
  const c = next.companies.c1;
  if (!c) throw new Error("no company");
  c.dataCenters.dc1 = {
    id: "dc1",
    region: "pacific",
    startDay: next.day,
    openDay: open ? next.day : next.day + 40,
    pflopsPerDay: BALANCE_V2.economy.dataCenter.pflopsPerDay,
  };
  c.compute.trainingShare = share;
  return next;
}

describe("computers", () => {
  it("show the share of questions answered", () => {
    const view = money(launchedWorld());
    const served = Math.round((view.me.company?.compute.served ?? -1) * 100);
    const panel = screen.getByRole("region", { name: "Computers" });
    expect(within(panel).getByText(`${served}%`)).toBeTruthy();
    expect(within(panel).getByText("of questions answered")).toBeTruthy();
  });

  it("send the share for training when the slider is let go", async () => {
    money(withCenters(launchedWorld(), true));
    const slider = screen.getByLabelText("Share for training");
    fireEvent.change(slider, { target: { value: "40" } });
    fireEvent.pointerUp(slider);
    await waitFor(() => expect(commands()).toEqual([{ type: "setTrainingShare", share: 0.4 }]));
  });

  it("build a data center from its window, showing a refusal there", async () => {
    money(launchedWorld());
    await userEvent.click(screen.getByRole("button", { name: "Build a data center" }));
    const w = await screen.findByRole("dialog", { name: "Build a data center" });
    expect(within(w).getByText(formatMoney(BALANCE_V2.economy.dataCenter.cost))).toBeTruthy();
    await userEvent.selectOptions(within(w).getByLabelText("Where"), "pacific");
    await userEvent.click(within(w).getByRole("button", { name: "Build it" }));
    await waitFor(() =>
      expect(commands()).toEqual([{ type: "buildDataCenter", region: "pacific" }]),
    );
    const cid = game.ws.sentOf("command")[0]?.cid;
    game.ws.receive({ v: 1, type: "reject", cid, reason: "A data center costs $1,500,000." });
    expect((await within(w).findByRole("alert")).textContent).toBe(
      "A data center costs $1,500,000.",
    );
  });

  it("show a data center being built with its days left", () => {
    money(withCenters(launchedWorld(), false));
    const panel = screen.getByRole("region", { name: "Computers" });
    expect(within(panel).getByText("Pacific")).toBeTruthy();
    expect(within(panel).getByText("Opens in 40 days")).toBeTruthy();
  });

  it("price training with the compute open data centers lend it", async () => {
    const s = withCenters(runningWorld("balance-v2"), true, 0.5);
    render(<Lab view={viewOf(s)} />);
    fireEvent.click(screen.getByRole("button", { name: "3 Design" }));
    const cost = await screen.findByTestId("plan-cost");
    const plan = usePlan.getState().plan;
    if (!plan) throw new Error("no plan");
    const lent = BALANCE_V2.economy.dataCenter.pflopsPerDay * 0.5;
    const p = previewPlan(BALANCE_V2, null, plan, lent);
    if (!p.ok) throw new Error(p.reason);
    expect(cost.textContent).toContain(formatMoney(p.preview.cost));
    expect(screen.getByText(/Your data centers make this run/)).toBeTruthy();
  });
});

describe("money screen, growing", () => {
  it("shows income by region and the new cost lines, each opening its window", async () => {
    const view = money(launchedWorld());
    expect(view.me.company?.ledger.current.revenueByRegion.pacific).toBeGreaterThan(0);
    const coming = screen.getByRole("region", { name: "Coming in" });
    await userEvent.click(within(coming).getByRole("button", { name: "Open Pacific" }));
    expect(await screen.findByRole("dialog", { name: "Pacific" })).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Close all windows" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    const out = screen.getByRole("region", { name: "Going out" });
    await userEvent.click(within(out).getByRole("button", { name: "Open Ads and battles" }));
    expect(await screen.findByRole("dialog", { name: "Ads and battles" })).toBeTruthy();
  });

  it("shows money that is neither income nor a cost under the month", () => {
    const s = structuredClone(launchedWorld());
    const c = s.companies.c1;
    if (!c) throw new Error("no company");
    finance(c, "loans", 150_000);
    money(s);
    const other = screen.getByRole("region", { name: "Loans and investors" });
    expect(within(other).getByText("Loans")).toBeTruthy();
    expect(within(other).getByText(formatMoney(150_000))).toBeTruthy();
  });
});
