import { type NewsView, REGION_IDS, type WorldView } from "@airace/protocol";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { AppShell } from "../src/components/AppShell.tsx";
import { Feed } from "../src/components/Feed.tsx";
import { STATE_TILES } from "../src/regions.ts";
import { Market } from "../src/screens/Market.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { rivalWorld, ticks, viewOf } from "./views.ts";

beforeEach(async () => {
  const game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.getState().closeAll();
});

const news = (id: string, companyId: string, topic: NewsView["topic"], headline: string) =>
  ({ kind: "news", id, day: 30, companyId, sentiment: 0, topic, headline }) as NewsView;

describe("the map's leaders", () => {
  it("tints each region in the color of the company with the most customers there", async () => {
    const view = viewOf(ticks(rivalWorld(), 30));
    render(<Market view={view} />);
    await userEvent.click(screen.getByRole("radio", { name: "Leaders" }));
    for (const region of REGION_IDS) {
      const leader = [...view.companies].sort(
        (a, b) => b.customersByRegion[region] - a.customersByRegion[region],
      )[0];
      const state = STATE_TILES.find((t) => t.region === region)?.state;
      const tile = document.querySelector(`[data-state="${state}"]`);
      expect(tile?.getAttribute("fill"), region).toBe(leader?.palette.primary);
    }
  });
});

describe("the feed with rivals", () => {
  it("can show only stories about you, and labels trash-talk as game commentary", async () => {
    const view = viewOf(rivalWorld());
    const arch = view.companies.find((c) => c.rival?.archetype);
    if (!arch) throw new Error("no arch-rival");
    const items = [
      news("a", "c1", "customers", "Acme passes 1,000 customers"),
      news("b", arch.id, "trashTalk", `${arch.name}'s founder: "Cute launch, Acme."`),
    ];
    render(<Feed items={items} companyId="c1" companies={view.companies} />);
    expect(screen.getByText("Game commentary")).toBeTruthy();
    await userEvent.click(screen.getByRole("radio", { name: "About you" }));
    expect(screen.queryByText(/Cute launch/)).toBeNull();
    expect(screen.getByText("Acme passes 1,000 customers")).toBeTruthy();
  });
});

describe("rank changes on the report card", () => {
  it("shows each table's place from last month to this", async () => {
    const base = viewOf(ticks(rivalWorld(), 40));
    const company = base.me.company;
    const card = company?.reports.at(-1);
    if (!company || !card) throw new Error("no report card");
    const view: WorldView = {
      ...base,
      me: {
        ...base.me,
        company: {
          ...company,
          reports: [
            ...company.reports.slice(0, -1),
            {
              ...card,
              ranks: [
                { table: "customers", from: 3, to: 1 },
                { table: "revenue", from: null, to: 2 },
                { table: "overall", from: 1, to: 4 },
              ],
            },
          ],
        },
      },
    };
    render(<WindowHost view={view} />);
    useWindows.getState().open({ kind: "report", month: card.month });
    const w = await screen.findByRole("dialog");
    expect(within(w).getByText("Customers: 3rd → 1st")).toBeTruthy();
    expect(within(w).getByText("Revenue: 2nd")).toBeTruthy();
    expect(within(w).getByText("Overall: 1st → 4th")).toBeTruthy();
  });
});

describe("rival toasts", () => {
  it("pop up for the arch-rival's launches, trash-talk naming you and your battle results only", () => {
    const before = viewOf(rivalWorld());
    const arch = before.companies.find((c) => c.rival?.archetype);
    const other = before.companies.find((c) => c.rival && !c.rival.archetype);
    if (!arch || !other) throw new Error("no rivals");
    const shell = (v: WorldView) => (
      <AppShell view={v} tab="company">
        <p>Screen</p>
      </AppShell>
    );
    const { rerender } = render(shell(before));
    rerender(
      shell({
        ...before,
        feed: [
          ...before.feed,
          news("x1", arch.id, "launch", `${arch.name} launches Titan 2`),
          news("x2", other.id, "launch", `${other.name} launches Lite 2`),
          news("x3", arch.id, "trashTalk", `${arch.name}'s founder: "Nice try, Acme."`),
        ],
      }),
    );
    const text = screen
      .getAllByRole("status")
      .map((t) => t.textContent)
      .join(" ");
    expect(text).toContain(`${arch.name} launches Titan 2`);
    expect(text).toContain("Nice try, Acme.");
    expect(text).not.toContain(`${other.name} launches Lite 2`);

    const company = before.me.company;
    if (!company) throw new Error("no company");
    const won: WorldView = {
      ...before,
      me: {
        ...before.me,
        company: {
          ...company,
          battles: [
            {
              id: "b9",
              day: 20,
              skill: "coding",
              rivalId: arch.id,
              mine: { productId: "p1", versionId: "v1", productName: "Spark" },
              theirs: { productId: "x", versionId: "y", productName: "Titan" },
              status: "done",
              rounds: [0, 1, 2].map((k) => ({
                prompt: "Q",
                mine: { reply: "a", score: k < 2 ? 1 : 0 },
                theirs: { reply: "b", score: k < 2 ? 0 : 1 },
              })),
              outcome: "win",
              pointsChange: 16,
            },
          ],
        },
      },
    };
    rerender(shell(won));
    const after = screen
      .getAllByRole("status")
      .map((t) => t.textContent)
      .join(" ");
    expect(after).toContain(`You won the coding battle against ${arch.name}, 2–1`);
  });
});
