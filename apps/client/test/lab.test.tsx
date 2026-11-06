import { type Catalog, prepKey } from "@airace/protocol";
import { BALANCE_V2, formatMoney, previewPlan, previewTraining } from "@airace/sim";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
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
    fireEvent.click(screen.getByRole("button", { name: /^Small/ }));
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
    fireEvent.click(screen.getByRole("checkbox", { name: /^Fact-checking/ }));
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
    const { container } = render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    for (const d of CATALOG.datasets) {
      expect(screen.getByText(d.title)).toBeTruthy();
      // Samples keep their line breaks (pre-wrap), so compare the raw text.
      for (const sample of d.samples) expect(container.textContent).toContain(sample);
    }
  });

  it("shows what thorough cleaning removes and what is left", () => {
    render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    step("2 Prepare");
    fireEvent.click(screen.getByRole("button", { name: "Thorough" }));
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
    fireEvent.click(screen.getByRole("button", { name: /^Long/ }));
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

describe("the Lab's bottom bar and the corners", () => {
  it("reserves its height while shown, so notifications and the world chat sit above it", async () => {
    const root = document.documentElement;
    const { unmount } = render(<Lab view={viewOf(runningWorld("balance-v2"))} />);
    step("1 Data");
    await screen.findByRole("button", { name: "Start training" });
    expect(root.style.getPropertyValue("--bottom-bar")).toMatch(/^\d+px$/);
    unmount();
    expect(root.style.getPropertyValue("--bottom-bar")).toBe("");
    const { container: toasts } = render(<Toasts view={viewOf(runningWorld("balance-v2"))} />);
    expect((toasts.firstElementChild as HTMLElement).style.bottom).toContain("--bottom-bar");
    const { container: chat } = render(<WorldChat view={viewOf(runningWorld("balance-v2"))} />);
    expect((chat.firstElementChild as HTMLElement).style.bottom).toContain("--bottom-bar");
  });
});
