import type { NewsView } from "@airace/protocol";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "../src/components/AppShell.tsx";
import { CountUp } from "../src/components/CountUp.tsx";
import { FloatDelta } from "../src/components/FloatDelta.tsx";
import { Stat } from "../src/components/Stat.tsx";
import { formatCount, formatMoney } from "../src/format.ts";
import { parseRoute } from "../src/route.ts";
import { Home } from "../src/screens/Home.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, runningWorld, viewOf } from "./views.ts";

beforeEach(async () => {
  useSession.setState({ socket: (await liveGame()).socket });
  useWindows.getState().closeAll();
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

describe("day bar", () => {
  it("fills over each game day at the world's speed, and rests while paused", () => {
    const view = viewOf(runningWorld());
    const shell = (v: typeof view) => (
      <AppShell view={v} tab="company">
        <p>Screen</p>
      </AppShell>
    );
    const { rerender } = render(shell(view));
    expect(screen.getByTestId("day-bar").style.animationDuration).toBe("200ms");
    rerender(shell({ ...view, world: { ...view.world, speed: 4 } }));
    expect(screen.getByTestId("day-bar").style.animationDuration).toBe("50ms");
    rerender(shell({ ...view, world: { ...view.world, paused: true } }));
    expect(screen.getByTestId("day-bar").getAttribute("data-paused")).toBe("true");
  });
});

describe("toasts", () => {
  const news = (id: string, companyId: string, headline: string): NewsView => ({
    kind: "news",
    id,
    day: 9,
    companyId,
    sentiment: 0.6,
    topic: "customers",
    headline,
  });

  it("pop up for new news about the player's own company only", async () => {
    const before = viewOf(launchedWorld());
    const shell = (v: typeof before) => (
      <AppShell view={v} tab="company">
        <p>Screen</p>
      </AppShell>
    );
    const { rerender } = render(shell(before));
    expect(screen.queryByRole("status")).toBeNull();
    rerender(
      shell({
        ...before,
        feed: [
          ...before.feed,
          news("n-mine", "c1", "Acme passes 1,000 customers"),
          news("n-theirs", "c2", "Bolt passes 1,000 customers"),
        ],
      }),
    );
    const toast = screen.getByRole("status");
    expect(toast.textContent).toContain("Acme passes 1,000 customers");
    expect(screen.queryByText("Bolt passes 1,000 customers")).toBeNull();
    await userEvent.click(within(toast).getByRole("button", { name: /Acme passes/ }));
    expect(await screen.findByRole("dialog", { name: "Customers" })).toBeTruthy();
  });
});

describe("top bar", () => {
  it("opens the money window from the cash", async () => {
    render(
      <AppShell view={viewOf(launchedWorld())} tab="company">
        <p>Screen</p>
      </AppShell>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Open money" }));
    expect(await screen.findByRole("dialog", { name: "Money" })).toBeTruthy();
  });

  it("marks the open screen with a highlight that moves between screens", () => {
    render(
      <AppShell view={viewOf(runningWorld())} tab="market">
        <p>Screen</p>
      </AppShell>,
    );
    const nav = screen.getByRole("navigation", { name: "Screens" });
    const market = within(nav).getByRole("link", { name: "Market" });
    expect(market.querySelector("[data-pill]")).not.toBeNull();
    expect(nav.querySelectorAll("[data-pill]")).toHaveLength(1);
  });
});

describe("floating changes", () => {
  it("float a number's change up from it, unless motion is reduced", () => {
    motion(false);
    const first = render(<FloatDelta value={10} format={formatCount} />);
    first.rerender(<FloatDelta value={15} format={formatCount} />);
    expect(screen.getByText("+5")).toBeTruthy();
    first.unmount();
    motion(true);
    const second = render(<FloatDelta value={10} format={formatCount} />);
    second.rerender(<FloatDelta value={15} format={formatCount} />);
    expect(screen.queryByText("+5")).toBeNull();
  });
});
