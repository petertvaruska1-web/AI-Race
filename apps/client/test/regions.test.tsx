import { GROUP_IDS, REGION_IDS, REGION_NAMES, type RegionId } from "@airace/protocol";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RegionMap } from "../src/components/RegionMap.tsx";
import { RegionReadout } from "../src/components/RegionReadout.tsx";
import { formatCount, formatMoney } from "../src/format.ts";
import { leaderTops, type RegionTop, regionCustomers, STATE_TILES } from "../src/regions.ts";
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

const PLAIN = Object.fromEntries(
  REGION_IDS.map((r) => [r, { color: null, height: 0, leaderId: null }]),
) as Record<RegionId, RegionTop>;

describe("region map", () => {
  it("lays out the 50 states and DC as tiles, each in one of the nine regions", () => {
    expect(STATE_TILES).toHaveLength(51);
    expect(new Set(STATE_TILES.map((t) => t.state)).size).toBe(51);
    expect(new Set(STATE_TILES.map((t) => `${t.col},${t.row}`)).size).toBe(51);
    for (const r of REGION_IDS) expect(STATE_TILES.some((t) => t.region === r)).toBe(true);
    expect(STATE_TILES.find((t) => t.state === "CA")?.region).toBe("pacific");
    expect(STATE_TILES.find((t) => t.state === "TX")?.region).toBe("southCentral");
  });

  it("draws each region in its top's color and height, plain glass where nobody leads", () => {
    const tops = { ...PLAIN, pacific: { color: "#ff9f1c", height: 1, leaderId: "c1" } };
    const { container } = render(
      <RegionMap label="Who leads each region" tops={tops} markColor="#ff9f1c" />,
    );
    const tile = (state: string) => container.querySelector(`[data-state="${state}"]`);
    expect(tile("CA")?.getAttribute("fill")).toBe("#ff9f1c");
    expect(tile("CA")?.getAttribute("data-plain")).toBeNull();
    expect(tile("NY")?.getAttribute("data-plain")).toBe("true");
    expect(Number(tile("CA")?.getAttribute("fill-opacity"))).toBeGreaterThan(
      Number(tile("NY")?.getAttribute("fill-opacity")),
    );
    expect(screen.getAllByRole("button", { name: /^Open / })).toHaveLength(9);
  });

  it("opens a region from its key", async () => {
    const onOpen = vi.fn();
    render(
      <RegionMap label="Who leads each region" tops={PLAIN} markColor="#4f8cff" onOpen={onOpen} />,
    );
    await userEvent.click(screen.getByRole("button", { name: "Open Pacific" }));
    expect(onOpen).toHaveBeenCalledWith("pacific", expect.anything());
  });

  it("lights a region and shows its readout while its key has focus", async () => {
    render(
      <RegionMap
        label="Who leads each region"
        tops={PLAIN}
        markColor="#4f8cff"
        readout={(r) => <p>{`readout ${r}`}</p>}
      />,
    );
    screen.getByRole("button", { name: "Open Pacific" }).focus();
    expect(await screen.findByText("readout pacific")).toBeTruthy();
  });

  it("says who leads a region and what you have there", () => {
    const view = viewOf(launchedWorld());
    render(<RegionReadout view={view} region="pacific" />);
    const lead = view.companies.find((c) => c.id === leaderTops(view).pacific.leaderId);
    expect(screen.getByText(lead ? `Led by ${lead.name}` : "Nobody leads it yet")).toBeTruthy();
    expect(screen.getByText(formatCount(regionCustomers(view, "pacific")))).toBeTruthy();
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
    const map = screen.getByRole("group", { name: "Who leads each region" });
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
    const map = screen.getByRole("group", { name: "Who leads each region" });
    await userEvent.click(within(map).getByRole("button", { name: "Open New England" }));
    expect(await screen.findByRole("dialog", { name: "New England" })).toBeTruthy();
  });
});
