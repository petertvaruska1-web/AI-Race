import { render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "../src/components/AppShell.tsx";
import { CountUp } from "../src/components/CountUp.tsx";
import { Stat } from "../src/components/Stat.tsx";
import { formatMoney } from "../src/format.ts";
import { parseRoute } from "../src/route.ts";
import { Home } from "../src/screens/Home.tsx";
import { useSession } from "../src/store/session.ts";
import { liveGame } from "./mockSocket.ts";
import { runningWorld, viewOf } from "./views.ts";

beforeEach(async () => {
  useSession.setState({ socket: (await liveGame()).socket });
});
afterEach(() => vi.unstubAllGlobals());

/** `matchMedia` that reports reduced motion as `reduce` says. */
function motion(reduce: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: reduce && query.includes("prefers-reduced-motion: reduce"),
    media: query,
    onchange: null,
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
    dispatchEvent: () => false,
  }));
}

describe("app shell", () => {
  it("shows the company name in its brand color, and its cash", () => {
    const view = viewOf(runningWorld());
    render(
      <AppShell view={view} tab="company">
        <p>Screen</p>
      </AppShell>,
    );
    expect(screen.getByText("Acme").className).toContain("text-brand");
    expect(screen.getByText(formatMoney(view.me.company?.cash ?? -1))).toBeTruthy();
    expect(screen.getByText("Screen")).toBeTruthy();
  });

  it("lists the six screens and marks the open one", () => {
    render(
      <AppShell view={viewOf(runningWorld())} tab="money">
        <p>Screen</p>
      </AppShell>,
    );
    const nav = screen.getByRole("navigation", { name: "Screens" });
    const links = within(nav).getAllByRole("link");
    expect(links.map((l) => l.textContent)).toEqual([
      "Company",
      "Market",
      "Money",
      "AI Lab",
      "Playground",
      "Benchmarks",
    ]);
    expect(within(nav).getByRole("link", { name: "Money" }).getAttribute("aria-current")).toBe(
      "page",
    );
    expect(within(nav).getByRole("link", { name: "Company" }).getAttribute("href")).toBe(
      "#/w/w1/company",
    );
  });
});

describe("routes", () => {
  it("opens the company screen for the old dashboard address, and by default", () => {
    expect(parseRoute("#/w/w1/dashboard")).toEqual({
      screen: "world",
      worldId: "w1",
      tab: "company",
    });
    expect(parseRoute("#/w/w1")).toEqual({ screen: "world", worldId: "w1", tab: "company" });
    expect(parseRoute("#/w/w1/market")).toEqual({ screen: "world", worldId: "w1", tab: "market" });
    expect(parseRoute("#/w/w1/money")).toEqual({ screen: "world", worldId: "w1", tab: "money" });
  });
});

describe("home", () => {
  it("shows a world made with an older version without an Open button", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              worlds: [
                { id: "w0", name: "Old times", outdated: true },
                { id: "w1", name: "Friday night", phase: "running", day: 12, memberCount: 1 },
              ],
            }),
          ),
      ),
    );
    render(<Home me={{ userId: "u1", displayName: "Ana", wsToken: "t" }} />);
    expect(await screen.findByText("Old times")).toBeTruthy();
    expect(screen.getByText("Made with an older version of the game")).toBeTruthy();
    expect(screen.getAllByRole("button", { name: "Open" })).toHaveLength(1);
  });
});

describe("count-up", () => {
  const dollars = (n: number) => `$${Math.round(n)}`;

  it("jumps straight to a new value when motion is reduced", () => {
    motion(true);
    const { rerender } = render(<CountUp value={0} format={dollars} />);
    rerender(<CountUp value={1234} format={dollars} />);
    expect(screen.getByText("$1234")).toBeTruthy();
  });

  it("rolls toward a new value otherwise", () => {
    motion(false);
    const { rerender } = render(<CountUp value={0} format={dollars} />);
    rerender(<CountUp value={1234} format={dollars} />);
    expect(screen.queryByText("$1234")).toBeNull();
  });
});

describe("stat", () => {
  it("colors a change by whether it is good news", () => {
    const { rerender } = render(
      <Stat label="Costs" value={10} delta={5} downIsGood format={String} />,
    );
    expect(screen.getByText(/▲/).className).toContain("text-danger");
    rerender(<Stat label="Costs" value={10} delta={-5} downIsGood format={String} />);
    expect(screen.getByText(/▼/).className).toContain("text-good");
    rerender(<Stat label="Income" value={10} delta={5} format={String} />);
    expect(screen.getByText(/▲/).className).toContain("text-good");
  });
});
