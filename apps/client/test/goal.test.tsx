import { GoalSpec } from "@airace/protocol";
import { createWorld, projectView, type WorldState } from "@airace/sim";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Home } from "../src/screens/Home.tsx";
import { useWindows } from "../src/store/windows.ts";
import { GoalProgress, Victory } from "../src/surprises/Goals.tsx";
import { apply, command, LOGO, PALETTE, ticks } from "./views.ts";

/** Answers `fetch` by path; records every request. */
function mockFetch(routes: Record<string, unknown>) {
  const calls: { path: string; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init?: RequestInit) => {
      calls.push({ path, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      const data = routes[path];
      return new Response(JSON.stringify(data ?? { error: "not found" }), {
        status: data === undefined ? 404 : 200,
      });
    }),
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());
beforeEach(() => {
  useWindows.getState().closeAll();
  try {
    localStorage.clear();
  } catch {
    // no storage in this environment
  }
});

const ME = { userId: "u1", displayName: "Ana", wsToken: "t" };

/** A running world with `spec` as its goal; c1 is the player's. */
function goalWorld(spec: unknown): WorldState {
  let s = createWorld({
    worldId: "w1",
    name: "Goals",
    hostUserId: "u1",
    hostName: "Ana",
    seed: 1,
    balance: "balance-test",
    inviteCode: "ABC234",
    goal: GoalSpec.parse(spec),
  });
  s = apply(
    s,
    command("u1", { type: "createCompany", name: "Acme", logo: LOGO, palette: PALETTE }, "c1"),
  );
  return apply(s, command("u1", { type: "startWorld" }));
}

describe("the goal picker", () => {
  it("sends the goal picked, with its target and deadline", async () => {
    const calls = mockFetch({
      "/api/worlds": { worlds: [], worldId: "w9" },
      "/api/setup": { trainingData: true, rivals: true },
    });
    render(<Home me={ME} />);
    await screen.findByText(/No worlds yet/);
    fireEvent.change(screen.getByLabelText("World name"), { target: { value: "Race" } });
    await userEvent.click(screen.getByText("Reach 1,000,000 customers"));
    fireEvent.change(screen.getByLabelText("Goal target"), { target: { value: "5000" } });
    fireEvent.change(screen.getByLabelText("Goal deadline"), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(calls.at(-1)?.path).toBe("/api/worlds"));
    expect(calls.at(-1)?.body).toEqual({
      name: "Race",
      goal: { goal: { kind: "customers", target: 5000 }, deadlineYears: 2 },
    });
  });

  it("hides the rival goals without the rival AIs, and sends no goal by default", async () => {
    const calls = mockFetch({
      "/api/worlds": { worlds: [], worldId: "w9" },
      "/api/setup": { trainingData: true, rivals: false },
    });
    render(<Home me={ME} />);
    await screen.findByText(/No worlds yet/);
    await waitFor(() => expect(calls.some((c) => c.path === "/api/setup")).toBe(true));
    expect(screen.queryByText("Beat the arch-rival on customers")).toBeNull();
    expect(screen.queryByText("Build the best AI overall")).toBeNull();
    expect(screen.getByText("No goal: play as long as you like")).toBeTruthy();
    fireEvent.change(screen.getByLabelText("World name"), { target: { value: "Race" } });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));
    await waitFor(() => expect(calls.at(-1)?.body).toEqual({ name: "Race" }));
  });
});

describe("goal progress and the victory", () => {
  it("shows the value against the target, and the deadline", () => {
    const s = goalWorld({ goal: { kind: "customers", target: 1_000 }, deadlineYears: 1 });
    s.market.customers.c1 = { newEngland: { everyday: { free: 250 } } };
    render(<GoalProgress view={projectView(s, "u1")} />);
    const bar = screen.getByTestId("goal-progress");
    expect(bar.textContent).toContain("Reach 1,000 customers");
    expect(bar.textContent).toContain("250 / 1,000");
    expect(bar.textContent).toContain("365 days left");
  });

  it("the victory screen shows once, with the company's story, and Keep playing closes it", async () => {
    const s = ticks(
      goalWorld({ goal: { kind: "customers", target: 1_000 }, deadlineYears: null }),
      3,
    );
    s.goal.reachedDay = s.day;
    s.goal.reachedBy = "c1";
    s.companies.c1?.milestones.push({ id: "firstPaying", day: s.day, detail: null });
    const view = projectView(s, "u1");
    const { unmount } = render(<Victory view={view} />);
    const dialog = await screen.findByRole("dialog", { name: "You reached your goal" });
    expect(dialog.textContent).toContain("Acme did it!");
    expect(dialog.textContent).toContain("First paying customer");
    // Customers and cash are drawn on charts of their own: on one, a few thousand
    // customers would lie flat under a cash line in the hundreds of thousands.
    expect(screen.getByRole("img", { name: "Customers" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "Cash" })).toBeTruthy();
    await userEvent.click(screen.getByRole("button", { name: "Keep playing" }));
    expect(screen.queryByRole("dialog", { name: "You reached your goal" })).toBeNull();
    unmount();
    render(<Victory view={view} />);
    await act(async () => {});
    expect(screen.queryByRole("dialog", { name: "You reached your goal" })).toBeNull();
  });

  it("no victory for a goal another company reached", () => {
    const s = goalWorld({ goal: { kind: "customers", target: 1_000 }, deadlineYears: null });
    s.goal.reachedDay = s.day;
    s.goal.reachedBy = "c2";
    render(<Victory view={projectView(s, "u1")} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
