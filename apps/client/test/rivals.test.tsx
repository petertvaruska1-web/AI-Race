import { ARCHETYPE_WORDS } from "@airace/protocol";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../src/net/api.ts";
import { Playground } from "../src/screens/Playground.tsx";
import { Rivals } from "../src/screens/Rivals.tsx";
import { usePlayground } from "../src/store/playground.ts";
import { useSession } from "../src/store/session.ts";
import { useWindows } from "../src/store/windows.ts";
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
