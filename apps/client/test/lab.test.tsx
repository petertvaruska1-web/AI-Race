import { type Catalog, prepKey } from "@airace/protocol";
import { BALANCE_V2, formatMoney, previewPlan, previewTraining } from "@airace/sim";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import sampleCatalog from "../../../packages/protocol/fixtures/catalog.sample.json";
import { Toasts } from "../src/components/Toasts.tsx";
import { WorldChat } from "../src/components/WorldChat.tsx";
import { words } from "../src/format.ts";
import { Lab } from "../src/screens/lab/LabLayout.tsx";
import { useCatalog } from "../src/store/catalog.ts";
import { usePlan } from "../src/store/plan.ts";
import { useSession } from "../src/store/session.ts";
import { liveGame } from "./mockSocket.ts";
import { runningWorld, viewOf } from "./views.ts";

const CATALOG: Catalog = sampleCatalog as Catalog;

let game: Awaited<ReturnType<typeof liveGame>>;

beforeEach(async () => {
  game = await liveGame();
  useSession.setState({ socket: game.socket });
  useCatalog.setState({ catalog: CATALOG, error: null, loading: false });
  usePlan.setState({ plan: null, worldId: null });
});

const step = (name: string) => fireEvent.click(screen.getByRole("button", { name }));

describe("AI Lab planning", () => {
  it("prices the plan as the server will, and changes with the brain size", async () => {
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    step("3 Design");
    const cost = await screen.findByTestId("plan-cost");
    expect(cost.textContent).toContain("$41,596");
    fireEvent.click(screen.getByRole("radio", { name: /^Small/ }));
    const plan = usePlan.getState().plan;
    if (!plan) throw new Error("no plan");
    const small = previewPlan(BALANCE_V2, null, plan);
    if (!small.ok) throw new Error(small.reason);
    expect(screen.getByTestId("plan-cost").textContent).toContain(formatMoney(small.preview.cost));
  });

  it("prices with the company's Data team, and reads more with its Research, as the server does", async () => {
    const s = runningWorld("balance-v2");
    const company = s.companies.c1;
    if (!company) throw new Error("no company");
    company.departments.data = { budget: 200_000, staffed: 200_000 };
    company.departments.research = { budget: 200_000, staffed: 200_000 };
    render(<Lab view={viewOf(s)} />);
    step("2 Prepare");
    fireEvent.click(screen.getByRole("switch", { name: /^Fact-checking/ }));
    const plan = usePlan.getState().plan;
    if (!plan?.factCheck) throw new Error("no fact-checked plan");
    const server = previewTraining(s, "c1", plan);
    if (!server.ok) throw new Error(server.reason);
    const plain = previewPlan(BALANCE_V2, null, plan);
    if (!plain.ok) throw new Error(plain.reason);
    expect(server.preview.cost).toBeLessThan(plain.preview.cost);
    expect(screen.getByTestId("plan-cost").textContent).toContain(formatMoney(server.preview.cost));
  });

  it("says when there is not enough cash", () => {
    const s = runningWorld("balance-v2");
    const company = s.companies.c1;
    if (!company) throw new Error("no company");
    company.cash = 1000;
    render(<Lab view={viewOf(s)} />);
    const start = screen.getByRole("button", { name: "Start training" }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    expect(screen.getByRole("alert").textContent).toMatch(/Not enough cash/);
  });

  it("shows each dataset with its samples as text", () => {
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    for (const d of CATALOG.datasets) {
      expect(screen.getByText(d.title)).toBeTruthy();
      fireEvent.click(
        within(screen.getByTestId(`dataset-${d.id}`)).getByRole("button", {
          name: /see examples/i,
        }),
      );
      // Samples keep their line breaks (pre-wrap), so compare the raw text.
      const shown = screen.getByRole("region", { name: "Examples" });
      for (const sample of d.samples) expect(shown.textContent).toContain(sample);
    }
  });

  it("mixes the reading on a desk of faders, one per dataset", () => {
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    for (const d of CATALOG.datasets) {
      const fader = within(screen.getByTestId(`dataset-${d.id}`)).getByRole("slider", {
        name: `${d.title} share`,
      });
      expect(fader.getAttribute("aria-orientation")).toBe("vertical");
    }
    const first = CATALOG.datasets[0];
    if (!first) throw new Error("no dataset");
    const before = usePlan.getState().plan?.mix[first.id] ?? 0;
    fireEvent.keyDown(screen.getByRole("slider", { name: `${first.title} share` }), {
      key: "ArrowUp",
    });
    expect(usePlan.getState().plan?.mix[first.id]).toBe(before + 1);
    expect(screen.getByTestId("dataset-notebook")).toBeTruthy();
  });

  it("turns the step dial to the step on screen, its five keys named as before", () => {
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    const dial = screen.getByRole("navigation", { name: "Lab steps" });
    expect(
      within(dial)
        .getAllByRole("button")
        .map((b) => b.textContent),
    ).toEqual(["1 Data", "2 Prepare", "3 Design", "4 Train", "5 Versions"]);
    step("5 Versions");
    expect(screen.getByRole("button", { name: "5 Versions" }).getAttribute("aria-current")).toBe(
      "step",
    );
    fireEvent.keyDown(screen.getByRole("button", { name: "5 Versions" }), { key: "ArrowLeft" });
    expect(screen.getByRole("button", { name: "4 Train" }).getAttribute("aria-current")).toBe(
      "step",
    );
  });

  it("shows what thorough cleaning removes and what is left", () => {
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    step("2 Prepare");
    fireEvent.click(screen.getByRole("radio", { name: "Thorough" }));
    const removed = CATALOG.datasets[0]?.removedExamples.cleaning.thorough[0] ?? "";
    expect(screen.getAllByText(removed).length).toBeGreaterThan(0);
    const left = CATALOG.datasets.reduce(
      (n, d) => n + (d.pools[prepKey("thorough", false, false)]?.tokens ?? 0),
      0,
    );
    expect(screen.getByTestId("pool").textContent).toContain(words(left));
  });

  it("starts training with exactly the plan on screen", async () => {
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    step("3 Design");
    fireEvent.click(screen.getByRole("radio", { name: /^Long/ }));
    fireEvent.change(screen.getByPlaceholderText("What is the sun?"), {
      target: { value: "Who are you?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    await waitFor(() => expect(game.ws.sentOf("command")).toHaveLength(1));
    const plan = usePlan.getState().plan;
    expect(plan).toMatchObject({
      length: "long",
      probePrompt: "Who are you?",
      parentVersionId: null,
    });
    expect(game.ws.sentOf("command")[0]?.cmd).toEqual({ type: "startTraining", plan });
  });

  it("sets the cleaning on lit keys and each option on a switch with its lamp", () => {
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    step("2 Prepare");
    const cleaning = screen.getByRole("group", { name: "Cleaning" });
    expect(
      within(cleaning)
        .getAllByRole("radio")
        .map((r) => r.closest("label")?.textContent),
    ).toEqual(["Light", "Standard", "Thorough"]);
    for (const name of [
      "Remove duplicates",
      /^Fact-checking/,
      /^Fact-check plus/,
      "Balance the topics",
    ])
      expect(screen.getByRole("switch", { name })).toBeTruthy();
    fireEvent.click(screen.getByRole("switch", { name: "Balance the topics" }));
    expect(usePlan.getState().plan?.variety).toBe("balanced");
  });

  it("designs the brain on lit keys, each size with its price and game days", () => {
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    step("3 Design");
    const sizes = screen.getByRole("group", { name: "Brain size" });
    const plan = usePlan.getState().plan;
    if (!plan) throw new Error("no plan");
    const small = previewPlan(BALANCE_V2, null, { ...plan, brainSize: "small" });
    if (!small.ok) throw new Error(small.reason);
    const key = within(sizes).getByRole("radio", { name: /^Small/ });
    expect(key.closest("label")?.textContent).toContain(formatMoney(small.preview.cost));
    fireEvent.click(key);
    expect(usePlan.getState().plan?.brainSize).toBe("small");
    for (const name of ["Attention span", "Training length"])
      expect(screen.getByRole("group", { name })).toBeTruthy();
    fireEvent.click(
      within(screen.getByRole("group", { name: "Attention span" })).getByRole("radio", {
        name: /^512 words/,
      }),
    );
    expect(usePlan.getState().plan?.attentionSpan).toBe(512);
    expect(screen.getByRole("slider", { name: "Learning style" })).toBeTruthy();
  });

  it("keeps plain words on every control", () => {
    const { container } = render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    for (const s of ["1 Data", "2 Prepare", "3 Design"]) {
      step(s);
      const text = container.textContent?.toLowerCase() ?? "";
      for (const jargon of [
        "parameter",
        "learning rate",
        "token",
        "epoch",
        "loss",
        "dropout",
        "batch",
      ]) {
        expect(text, `${s} mentions ${jargon}`).not.toContain(jargon);
      }
    }
  });
});

describe("the Lab's summary plate and the corners", () => {
  it("keeps its plate on the bench, so the world chat stays at the foot of the station column (notifications print at the top)", async () => {
    const root = document.documentElement;
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    step("1 Data");
    await screen.findByRole("button", { name: "Start training" });
    expect(root.style.getPropertyValue("--bottom-bar")).toBe("");
    const { container: toasts } = render(<Toasts view={viewOf(runningWorld("balance-v2"))} />);
    const slips = toasts.firstElementChild as HTMLElement;
    expect(slips.style.bottom).toBe("");
    expect(slips.className).toContain("top-[100px]");
    const { container: chat } = render(<WorldChat view={viewOf(runningWorld("balance-v2"))} />);
    const intercom = chat.firstElementChild as HTMLElement;
    expect(intercom.style.bottom).toBe("");
    expect(intercom.className).toContain("bottom-4");
  });
});
