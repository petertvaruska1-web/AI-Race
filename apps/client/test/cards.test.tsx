import type { Catalog } from "@airace/protocol";
import { paramCount } from "@airace/sim";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import sampleCatalog from "../../../packages/protocol/fixtures/catalog.sample.json";
import { CardsPanel, realMinutes } from "../src/cards/Cards.tsx";
import { TopBar } from "../src/components/TopBar.tsx";
import { Lab } from "../src/screens/lab/LabLayout.tsx";
import { type CardView, useCards } from "../src/store/cards.ts";
import { useCatalog } from "../src/store/catalog.ts";
import { usePlan } from "../src/store/plan.ts";
import { useSession } from "../src/store/session.ts";
import { liveGame } from "./mockSocket.ts";
import { runningWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useCatalog.setState({ catalog: sampleCatalog as Catalog, error: null, loading: false });
  usePlan.setState({ plan: null, worldId: null });
});
afterEach(() => {
  useCards.setState({ worldId: null, cards: null, limit: null });
  vi.unstubAllGlobals();
});

const card = (o: Partial<CardView>): CardView => ({
  userId: "u1",
  playerName: "Ana",
  gpuName: "NVIDIA GeForce RTX 3050",
  vramMb: 8192,
  speed: 1,
  online: true,
  house: false,
  ...o,
});

function cardsAre(cards: CardView[], limit: { maxBrain: string | null; tokensPerSecond: number }) {
  useCards.setState({ worldId: "w", cards, limit });
}

describe("your computer", () => {
  it("with no card online, the chip asks to connect and training can't start", () => {
    cardsAre([card({ online: false, gpuName: null, vramMb: 0, speed: 0 })], {
      maxBrain: null,
      tokensPerSecond: 0,
    });
    const view = viewOf(runningWorld("balance-v2"));
    render(<TopBar view={view} />);
    expect(screen.getByRole("button", { name: "Connect your computer" })).toBeTruthy();
    render(<Lab view={view} />);
    const start = screen.getByRole("button", { name: "Start training" }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    expect(screen.getByText("Connect your computer to do this.")).toBeTruthy();
  });

  it("names the card in the chip when it is online", () => {
    cardsAre([card({})], { maxBrain: "medium", tokensPerSecond: 450_000 });
    render(<TopBar view={viewOf(runningWorld("balance-v2"))} />);
    expect(screen.getByRole("button", { name: "Your computer: RTX 3050 · online" })).toBeTruthy();
  });

  it("disables brains too big for the card and estimates real minutes", () => {
    cardsAre([card({})], { maxBrain: "small", tokensPerSecond: 450_000 });
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    fireEvent.click(screen.getByRole("button", { name: "3 Design" }));
    const medium = screen.getByRole("button", { name: /^Medium/ }) as HTMLButtonElement;
    expect(medium.disabled).toBe(true);
    expect(medium.textContent).toContain("Your graphics card can train up to the Small brain.");
    expect((screen.getByRole("button", { name: /^Small/ }) as HTMLButtonElement).disabled).toBe(
      false,
    );
    expect(screen.getByTestId("real-minutes").textContent).toMatch(
      /^About \d+ minutes? on your card\.$/,
    );
  });

  it("estimates real minutes from what a token was measured to cost", () => {
    // On the reference card (an RTX 3050 measures 450,000 on the speed test) the first AI, 12.6
    // million tokens into the Tiny brain, trains in about a minute, and the 7.5-million-parameter
    // brain reads 25.2 million tokens in about four (the old estimate said sixteen).
    expect(realMinutes(12_600_000, paramCount(4, 128), 450_000)).toBe(1);
    expect(realMinutes(25_200_000, paramCount(8, 256), 450_000)).toBe(4);
    // The longest run of all, a Huge brain reading the Marathon length: about half an hour.
    expect(realMinutes(50_331_648, paramCount(12, 512), 450_000)).toBe(32);
    // A card twice as fast needs half the time; an unknown speed gives no estimate.
    expect(realMinutes(25_200_000, paramCount(8, 256), 900_000)).toBe(2);
    expect(realMinutes(25_200_000, paramCount(8, 256), 0)).toBeNull();
  });
});

describe("the cards window", () => {
  it("connects with a link carrying this page's address and the player's key", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ key: "u1.secretsig" }), { status: 200 })),
    );
    cardsAre([card({ online: false, gpuName: null })], { maxBrain: null, tokensPerSecond: 0 });
    render(<CardsPanel view={viewOf(runningWorld("balance-v2"))} />);
    const connect = await screen.findByRole("link", { name: "Connect your computer" });
    expect(connect.getAttribute("href")).toBe(
      `airace-helper://connect?url=${encodeURIComponent(location.origin)}&key=${encodeURIComponent("u1.secretsig")}`,
    );
    expect(screen.getByRole("link", { name: "Get the helper" }).getAttribute("href")).toBe(
      "/helper/setup.cmd",
    );
  });

  it("shows another player's card by name and speed, never a key; Fair mode shows the pool", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ key: "u1.secretsig" }), { status: 200 })),
    );
    cardsAre(
      [
        card({}),
        card({ userId: "u2", playerName: "Ben", gpuName: "NVIDIA GeForce RTX 4090", speed: 3.2 }),
      ],
      { maxBrain: "medium", tokensPerSecond: 90_000 },
    );
    const s = runningWorld("balance-v2");
    s.fairMode = true;
    render(<CardsPanel view={viewOf(s)} />);
    const ben = within(screen.getByText("Ben").closest("li") as HTMLElement);
    expect(ben.getByText(/RTX 4090/)).toBeTruthy();
    expect(ben.getByText(/3\.2×/)).toBeTruthy();
    await waitFor(() => expect(screen.getByRole("link", { name: "Connect your computer" })));
    expect(screen.getByText("Ben").closest("li")?.innerHTML).not.toContain("secretsig");
    expect(screen.getByText("Shared cards: 2 online")).toBeTruthy();
  });
});

describe("following the cards", () => {
  it("asks for the cards and the limit, then follows changes", async () => {
    const stop = useCards.getState().follow(game.socket, "w1");
    const queries = game.ws.sentOf("query");
    expect(queries.map((q) => q.q)).toEqual([
      { kind: "cards", worldId: "w1" },
      { kind: "trainingLimit", worldId: "w1" },
    ]);
    game.ws.receive({
      v: 1,
      type: "queryResult",
      id: queries[0]?.id,
      ok: true,
      data: { cards: [] },
    });
    game.ws.receive({
      v: 1,
      type: "queryResult",
      id: queries[1]?.id,
      ok: true,
      data: { maxBrain: null, tokensPerSecond: 0 },
    });
    await waitFor(() =>
      expect(useCards.getState().limit).toEqual({ maxBrain: null, tokensPerSecond: 0 }),
    );
    game.ws.receive({ v: 1, type: "stream", sub: "cards:w1", data: { cards: [card({})] } });
    await waitFor(() => expect(useCards.getState().cards).toHaveLength(1));
    expect(game.ws.sentOf("query").at(-1)?.q).toEqual({ kind: "trainingLimit", worldId: "w1" });
    stop();
    expect(game.ws.sentOf("unsubscribe").at(-1)?.sub).toBe("cards:w1");
  });
});
