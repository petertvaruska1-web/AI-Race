import { ARCHETYPE_WORDS } from "@airace/protocol";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../src/net/api.ts";
import { Playground } from "../src/screens/Playground.tsx";
import { Rivals } from "../src/screens/Rivals.tsx";
import { termsLabel } from "../src/spending/Licensing.tsx";
import { usePlayground } from "../src/store/playground.ts";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
import { readableOn, useLook } from "../src/theme/glass.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, rivalWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;
beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.getState().closeAll();
  usePlayground.setState({ versionId: null });
  vi.spyOn(api, "setup").mockResolvedValue({ trainingData: true, rivals: true });
});

describe("the Rivals screen", () => {
  it("shows the arch-rival first with its type, then the three rivals", () => {
    const view = viewOf(rivalWorld());
    render(<Rivals view={view} />);
    const cards = screen.getAllByRole("article");
    expect(cards).toHaveLength(4);
    const arch = view.companies.find((c) => c.rival?.archetype);
    if (!arch?.rival?.archetype) throw new Error("no arch-rival");
    expect(within(cards[0] as HTMLElement).getByText(arch.name)).toBeTruthy();
    expect(
      within(cards[0] as HTMLElement).getByText(
        `Arch-rival · ${ARCHETYPE_WORDS[arch.rival.archetype]}`,
      ),
    ).toBeTruthy();
  });

  it("writes each rival's name readable on the player's light glass", () => {
    useLook.setState({ palette: { primary: "#1e3a8a", secondary: "#ffb020" } });
    try {
      const view = viewOf(rivalWorld());
      const rivals = view.companies.filter((c) => c.rival);
      expect(rivals.some((c) => readableOn(c.palette.primary, "light") !== c.palette.primary)).toBe(
        true,
      );
      render(<Rivals view={view} />);
      for (const c of rivals) {
        const card = screen.getByRole("article", { name: c.name });
        const name = within(card).getByRole("heading", { name: c.name });
        expect(name.getAttribute("style")).toContain(readableOn(c.palette.primary, "light"));
      }
    } finally {
      useLook.setState({ palette: null });
    }
  });

  it("stands each rival on a plate, the arch-rival's first and widest with its badge", () => {
    const view = viewOf(rivalWorld());
    render(<Rivals view={view} />);
    const plates = screen.getAllByRole("article");
    expect(plates[0]?.className).toContain("rival-plate--arch");
    expect(within(plates[0] as HTMLElement).getByRole("img", { name: /emblem/ })).toBeTruthy();
    for (const plate of plates) expect(plate.className).toContain("rival-plate");
    for (const plate of plates.slice(1)) expect(plate.className).not.toContain("rival-plate--arch");
  });

  it("keeps a rival's license offer to its terms on the plate", () => {
    const view = viewOf(rivalWorld(10));
    const offer = view.licenseOffers[0];
    if (!offer) throw new Error("no offer");
    const company = view.companies.find((c) => c.id === offer.licensorId);
    render(<Rivals view={view} />);
    const plate = screen.getByRole("article", { name: company?.name ?? "?" });
    const key = within(plate).getByRole("button", { name: `License ${offer.aiName}` });
    const line = (key.closest("li") as HTMLElement).textContent ?? "";
    expect(line).toContain(termsLabel(offer));
    expect(line).not.toContain(company?.name ?? "?");
    expect(line).not.toContain("students");
  });

  it("says when a rival has no AI on sale yet, with no empty display", () => {
    const view = viewOf(rivalWorld());
    const rival = view.companies.find((c) => c.rival);
    if (!rival) throw new Error("no rival");
    rival.products = [];
    render(<Rivals view={view} />);
    const plate = screen.getByRole("article", { name: rival.name });
    expect(within(plate).getByText("No AI on sale yet.")).toBeTruthy();
    expect(plate.querySelector(".rival-plate__products")).toBeNull();
  });

  it("shows each rival's public facts from the view", () => {
    const view = viewOf(rivalWorld());
    render(<Rivals view={view} />);
    for (const c of view.companies.filter((x) => x.rival)) {
      const card = screen.getByRole("article", { name: c.name });
      expect(
        within(card).getByText(`${c.followers.toLocaleString("en-US")} followers`),
      ).toBeTruthy();
      for (const p of c.products)
        expect(within(card).getAllByText(p.name).length).toBeGreaterThan(0);
    }
  });

  it("Try it opens the Playground with that rival's AI and chats with it", async () => {
    const view = viewOf(rivalWorld());
    const { unmount } = render(<Rivals view={view} />);
    const arch = view.companies.find((c) => c.rival?.archetype);
    const live = arch?.products.find((p) => p.status === "live" && p.version);
    if (!arch || !live?.version) throw new Error("no rival AI on sale");
    const card = screen.getByRole("article", { name: arch.name });
    await userEvent.click(
      within(card).getAllByRole("button", { name: `Try ${live.name}` })[0] as HTMLElement,
    );
    expect(usePlayground.getState().versionId).toBe(live.version.id);
    expect(location.hash).toBe(`#/w/${view.world.id}/play`);
    unmount();
    render(<Playground view={view} />);
    expect(
      screen.getByText(new RegExp(`Say something to ${arch.name}'s ${live.name}`)),
    ).toBeTruthy();
    await userEvent.type(screen.getByLabelText("Your message"), "Who are you?{Enter}");
    await waitFor(() =>
      expect(game.ws.sentOf("chat").at(-1)).toMatchObject({ versionId: live.version?.id }),
    );
  });

  it("explains a world without rivals, and how they come", async () => {
    vi.spyOn(api, "setup").mockResolvedValue({ trainingData: true, rivals: false });
    render(<Rivals view={viewOf(launchedWorld())} />);
    expect(screen.getByText("This world has no rivals yet.")).toBeTruthy();
    expect(await screen.findByText(/Their AIs aren't built on this computer yet/)).toBeTruthy();
  });

  it("opens a rival's window with its products' scores against yours", async () => {
    const view = viewOf(rivalWorld());
    render(
      <>
        <Rivals view={view} />
        <WindowHost view={view} />
      </>,
    );
    const arch = view.companies.find((c) => c.rival?.archetype);
    if (!arch) throw new Error("no arch-rival");
    await userEvent.click(screen.getByRole("button", { name: `Open ${arch.name}` }));
    const w = await screen.findByRole("dialog", { name: arch.name });
    expect(within(w).getByText("Skills against yours")).toBeTruthy();
  });
});
