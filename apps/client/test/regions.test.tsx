import { GROUP_IDS, REGION_IDS, REGION_NAMES } from "@airace/protocol";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RegionMap } from "../src/components/RegionMap.tsx";
import { formatCount, formatMoney } from "../src/format.ts";
import { STATE_TILES } from "../src/regions.ts";
import { Company } from "../src/screens/Company.tsx";
import { Market } from "../src/screens/Market.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, viewOf } from "./views.ts";

beforeEach(async () => {
  useSession.setState({ socket: (await liveGame()).socket });
  useWindows.getState().closeAll();
});

const flat = (v: number) =>
  Object.fromEntries(REGION_IDS.map((r) => [r, v])) as Record<(typeof REGION_IDS)[number], number>;

describe("region map", () => {
  it("lays out the 50 states and DC as tiles, each in one of the nine regions", () => {
    expect(STATE_TILES).toHaveLength(51);
    expect(new Set(STATE_TILES.map((t) => t.state)).size).toBe(51);
    expect(new Set(STATE_TILES.map((t) => `${t.col},${t.row}`)).size).toBe(51);
    for (const r of REGION_IDS) expect(STATE_TILES.some((t) => t.region === r)).toBe(true);
    expect(STATE_TILES.find((t) => t.state === "CA")?.region).toBe("pacific");
    expect(STATE_TILES.find((t) => t.state === "TX")?.region).toBe("southCentral");
  });

  it("tints each region by its value and opens a region from its button", async () => {
    const onOpen = vi.fn();
    const { container } = render(
      <RegionMap
        label="Your customers by region"
        values={{ ...flat(0), pacific: 1 }}
        color="#4f8cff"
        onOpen={onOpen}
      />,
    );
    const tile = (state: string) => container.querySelector(`[data-state="${state}"]`);
    expect(Number(tile("CA")?.getAttribute("fill-opacity"))).toBeGreaterThan(
      Number(tile("NY")?.getAttribute("fill-opacity")),
    );
    const buttons = screen.getAllByRole("button", { name: /^Open / });
    expect(buttons).toHaveLength(9);
    await userEvent.click(screen.getByRole("button", { name: "Open Pacific" }));
    expect(onOpen).toHaveBeenCalledWith("pacific", expect.anything());
  });
});

describe("region window", () => {
  it("shows the region's people, your customers there by group and this month's revenue", async () => {
    const view = viewOf(launchedWorld());
    const company = view.me.company;
    render(
      <>
        <Market view={view} />
        <WindowHost view={view} />
      </>,
    );
    const map = screen.getByRole("group", { name: "Your customers by region" });
    await userEvent.click(within(map).getByRole("button", { name: "Open Pacific" }));
    const w = await screen.findByRole("dialog", { name: REGION_NAMES.pacific });
    const byGroup = company?.customers.byRegion.pacific;
    const total = GROUP_IDS.reduce((n, g) => n + (byGroup?.[g] ?? 0), 0);
    expect(within(w).getByTestId("region-customers").textContent).toBe(formatCount(total));
    expect(within(w).getByTestId("region-revenue").textContent).toBe(
      formatMoney(company?.ledger.current.revenueByRegion.pacific ?? -1),
    );
    expect(within(w).getByText("Answers come from the cloud")).toBeTruthy();
    expect(within(w).getByRole("img", { name: "Customers in the Pacific by day" })).toBeTruthy();
  });

  it("opens from the small map on the Company screen too", async () => {
    const view = viewOf(launchedWorld());
    render(
      <>
        <Company view={view} />
        <WindowHost view={view} />
      </>,
    );
    const map = screen.getByRole("group", { name: "Customers by region" });
    await userEvent.click(within(map).getByRole("button", { name: "Open New England" }));
    expect(await screen.findByRole("dialog", { name: "New England" })).toBeTruthy();
  });
});
