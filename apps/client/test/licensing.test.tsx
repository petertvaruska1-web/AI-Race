import type { Command, PostView } from "@airace/protocol";
import type { WorldState } from "@airace/sim";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { PostCard } from "../src/components/PostCard.tsx";
import { Toasts } from "../src/components/Toasts.tsx";
import {
  LicensesPanel,
  OfferWindow,
  ProposalWindow,
  TakeWindow,
} from "../src/spending/Licensing.tsx";
import { useSession } from "../src/store/session.ts";
import { useWindows, type WindowSpec } from "../src/store/windows.ts";
import { liveGame } from "./mockSocket.ts";
import { launchedWorld, rivalWorld, viewOf } from "./views.ts";

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useWindows.getState().closeAll();
});

const commands = () => game.ws.sentOf("command").map((m) => m.cmd as Command);

function entryFor(spec: WindowSpec) {
  useWindows.getState().open(spec);
  const entry = useWindows.getState().stack.at(-1);
  if (!entry) throw new Error("no window");
  return entry;
}

/** A rival (the first with an AI on sale) has asked to license Spark for its focus group. */
function asked(s0: WorldState): { s: WorldState; proposalId: string; rivalName: string } {
  const s = structuredClone(s0);
  const rival = Object.values(s.companies).find(
    (c) => c.rival && Object.values(c.products).some((p) => p.status === "live"),
  );
  const product = rival && Object.values(rival.products).find((p) => p.status === "live");
  if (!rival || !product) throw new Error("no rival on sale");
  s.licenses.seq += 1;
  const proposalId = `proposal-${s.licenses.seq}`;
  s.licenses.proposals.push({
    id: proposalId,
    licensorId: "c1",
    licenseeId: rival.id,
    versionId: "v1",
    aiName: "Spark",
    productId: product.id,
    priceCents: 500,
    months: 6,
    groups: ["students"],
    day: s.day,
  });
  return { s, proposalId, rivalName: rival.name };
}

describe("the Offer window", () => {
  it("sends the exact terms", async () => {
    const view = viewOf(launchedWorld());
    render(
      <OfferWindow
        entry={entryFor({ kind: "offerLicense", versionId: "v1" })}
        view={view}
        versionId="v1"
      />,
    );
    fireEvent.change(screen.getByLabelText("Price per customer a month ($)"), {
      target: { value: "2.5" },
    });
    fireEvent.change(screen.getByLabelText("How long"), { target: { value: "12" } });
    fireEvent.click(screen.getByLabelText("Students"));
    fireEvent.click(screen.getByLabelText("Everyday users"));
    fireEvent.click(screen.getByLabelText("Only one company may license it"));
    fireEvent.click(screen.getByRole("button", { name: "Offer it" }));
    await waitFor(() =>
      expect(commands()).toEqual([
        {
          type: "offerLicense",
          versionId: "v1",
          priceCents: 250,
          months: 12,
          groups: ["students", "everyday"],
          exclusive: true,
        },
      ]),
    );
  });
});

describe("the Take window", () => {
  it("lists your AIs on sale and the offer's groups, and sends the ones you keep", async () => {
    const view = viewOf(rivalWorld(10));
    const offer = view.licenseOffers[0];
    if (!offer) throw new Error("no offer");
    render(
      <TakeWindow
        entry={entryFor({ kind: "takeLicense", offerId: offer.id })}
        view={view}
        offerId={offer.id}
      />,
    );
    const product = screen.getByLabelText("For your AI") as HTMLSelectElement;
    expect([...product.options].map((o) => o.textContent)).toEqual(["Spark"]);
    const groups = within(screen.getByRole("group", { name: "For these customers" }));
    expect(groups.getAllByRole("checkbox")).toHaveLength(offer.groups.length);
    fireEvent.click(groups.getByLabelText("Creators"));
    fireEvent.click(screen.getByRole("button", { name: "License it" }));
    await waitFor(() =>
      expect(commands()).toEqual([
        {
          type: "takeLicense",
          offerId: offer.id,
          productId: "p1",
          groups: offer.groups.filter((g) => g !== "creators"),
        },
      ]),
    );
  });
});

describe("a license request", () => {
  it("pops up as a toast that opens the Proposal window, with Accept and Decline", async () => {
    const before = rivalWorld(10);
    const { rerender } = render(<Toasts view={viewOf(before)} />);
    const { s, proposalId, rivalName } = asked(before);
    rerender(<Toasts view={viewOf(s)} />);
    const toast = await screen.findByText(`${rivalName} asks to license Spark`);
    fireEvent.click(toast);
    expect(useWindows.getState().stack.at(-1)?.spec).toEqual({ kind: "proposal", proposalId });
    const entry = useWindows.getState().stack.at(-1);
    if (!entry) throw new Error("no window");
    render(<ProposalWindow entry={entry} view={viewOf(s)} proposalId={proposalId} />);
    expect(screen.getByRole("button", { name: "Decline" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Accept" }));
    await waitFor(() =>
      expect(commands()).toEqual([{ type: "answerProposal", proposalId, accept: true }]),
    );
  });
});

describe("the Licenses panel", () => {
  it("shows a running license and lets the licensee end it early", async () => {
    const s = rivalWorld(10);
    const view0 = viewOf(s);
    const offer = view0.licenseOffers[0];
    if (!offer) throw new Error("no offer");
    const next = structuredClone(s);
    next.licenses.deals.push({
      id: "deal-9",
      licensorId: offer.licensorId,
      licenseeId: "c1",
      versionId: offer.versionId,
      aiName: offer.aiName,
      productId: "p1",
      priceCents: offer.priceCents,
      groups: ["students"],
      startDay: s.day,
      endDay: s.day + 180,
      exclusive: false,
      owed: 1234,
    });
    render(<LicensesPanel view={viewOf(next)} />);
    const panel = screen.getByRole("region", { name: "Licenses" });
    expect(within(panel).getByText(`${offer.aiName} powers Spark for students`)).toBeTruthy();
    expect(within(panel).getByText(/\$1,234 owed so far this month/)).toBeTruthy();
    fireEvent.click(within(panel).getByRole("button", { name: "End early" }));
    fireEvent.click(within(panel).getByRole("button", { name: "Yes, end it" }));
    await waitFor(() => expect(commands()).toEqual([{ type: "endLicense", dealId: "deal-9" }]));
  });
});

describe("posts", () => {
  it("say which licensed AI wrote the reply", () => {
    const post: PostView = {
      id: "f1",
      kind: "post",
      day: 3,
      companyId: "c1",
      sentiment: 1,
      author: { name: "Maya", handle: "@maya", group: "students", spotlightId: null },
      productName: "Owl",
      question: "2 + 2?",
      reply: "4",
      tone: "praise",
      poweredBy: "Pengu",
    };
    render(<PostCard post={post} />);
    expect(screen.getByText("powered by Pengu")).toBeTruthy();
  });
});
