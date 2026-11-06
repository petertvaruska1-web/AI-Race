import type { Command } from "@airace/protocol";
import type { WorldState } from "@airace/sim";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { Money } from "../src/screens/Money.tsx";
import { TeamPanel } from "../src/spending/Team.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { ReportWindow } from "../src/windows/ReportWindow.tsx";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, runningWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.getState().closeAll();
});

const commands = () => game.ws.sentOf("command").map((m) => m.cmd as Command);

/** `s` with c1's `department` grown to `dollars` a month (budget and team alike). */
function staffed(
  s: WorldState,
  department: "research" | "data" | "sales" | "support",
  dollars: number,
) {
  const next = structuredClone(s);
  const c = next.companies.c1;
  if (!c) throw new Error("no company");
  c.departments[department] = { budget: dollars, staffed: dollars };
  return next;
}

describe("the Team panel", () => {
  it("shows the four departments by name", () => {
    render(<TeamPanel view={viewOf(runningWorld("balance-v2"))} />);
    const panel = screen.getByRole("region", { name: "Team" });
    for (const name of ["Research", "Data team", "Sales & marketing", "Support & operations"]) {
      expect(within(panel).getByText(name)).toBeTruthy();
    }
  });

  it("sends a budget in $10,000 steps when the slider is let go", async () => {
    render(<TeamPanel view={viewOf(runningWorld("balance-v2"))} />);
    const slider = screen.getByLabelText("Research budget") as HTMLInputElement;
    expect(slider.step).toBe("10000");
    expect(slider.max).toBe("500000");
    fireEvent.change(slider, { target: { value: "120000" } });
    expect(commands()).toEqual([]); // nothing until it is let go
    expect(screen.getByText("$120,000 a month")).toBeTruthy();
    fireEvent.pointerUp(slider);
    await waitFor(() =>
      expect(commands()).toEqual([
        { type: "setDepartment", department: "research", budget: 120_000 },
      ]),
    );
  });

  it("says what each team does at its strength", () => {
    // 200,000 staffed: strength 1 − e^−1 ≈ 0.632.
    let s = runningWorld("balance-v2");
    for (const d of ["research", "data", "sales", "support"] as const) s = staffed(s, d, 200_000);
    render(<TeamPanel view={viewOf(s)} />);
    expect(screen.getByText("Training reads 19% more text")).toBeTruthy(); // 0.3 × 0.632
    expect(screen.getByText("Checks cost 32% less")).toBeTruthy(); // 0.5 × 0.632
    expect(screen.getByText("+0.13% awareness a day")).toBeTruthy(); // 0.002 × 0.632
    expect(
      screen.getByText(
        "Bad answers sting 19% less · 19% fewer unhappy customers leave · serving 13% cheaper",
      ),
    ).toBeTruthy();
    expect(screen.getByLabelText("Research strength").getAttribute("aria-valuenow")).toBe("63");
  });

  it("says an unfunded team does nothing yet", () => {
    render(<TeamPanel view={viewOf(runningWorld("balance-v2"))} />);
    expect(screen.getAllByText("No team yet.")).toHaveLength(4);
  });
});

describe("the Team's money", () => {
  it("shows a Team line on the Money screen once there is spending", () => {
    const s = launchedWorld();
    const { unmount } = render(<Money view={viewOf(s)} />);
    expect(screen.queryByTestId("cost-team")).toBeNull();
    unmount();
    const spent = structuredClone(s);
    const c = spent.companies.c1;
    if (!c) throw new Error("no company");
    c.ledger.current.costs.team = 30_000;
    render(<Money view={viewOf(spent)} />);
    expect(screen.getByTestId("cost-team").textContent).toBe("$30,000");
    expect(screen.getByText("Team")).toBeTruthy();
  });

  it("shows what the team cost in a month's report card", () => {
    const s = structuredClone(launchedWorld());
    const c = s.companies.c1;
    if (!c) throw new Error("no company");
    const month = structuredClone(c.ledger.current);
    month.costs.team = 45_000;
    c.ledger.months = [month];
    c.reports = [
      {
        month: month.month,
        revenue: 0,
        costs: 45_000,
        profit: -45_000,
        lastProfit: null,
        customersStart: 0,
        customersEnd: 0,
        growthByGroup: {},
        mood: null,
        best: null,
        worst: null,
        headline: null,
        grade: "C",
        ranks: null,
      } as never,
    ];
    const view = viewOf(s);
    useWindows.getState().open({ kind: "report", month: month.month });
    const entry = useWindows.getState().stack[0];
    if (!entry) throw new Error("no window");
    render(<ReportWindow entry={entry} view={view} month={month.month} />);
    expect(screen.getByText("Your team cost $45,000 this month.")).toBeTruthy();
  });
});
