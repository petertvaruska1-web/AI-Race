import { GROUP_IDS } from "@airace/protocol";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { formatCount } from "../src/format.ts";
import { GROUPS } from "../src/groups.ts";
import { aboutPeople, groupCustomers } from "../src/people.ts";
import { crowdStep, Market } from "../src/screens/Market.tsx";
import { useWindows } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { launchedWorld, viewOf } from "./views.ts";

describe("market screen", () => {
  it("shows six customer groups, each with its spotlight customer as text", () => {
    const s = launchedWorld();
    const first = s.spotlights[0];
    if (!first) throw new Error("no spotlight");
    first.bio = "<b>bold</b> <|end|>";
    const view = viewOf(s);
    render(<Market view={view} />);
    for (const g of GROUP_IDS) {
      const card = screen.getByRole("region", { name: GROUPS[g].many });
      const person = view.spotlights.find((p) => p.group === g);
      expect(within(card).getByText(person?.name ?? "?")).toBeTruthy();
    }
    expect(screen.getByText("<b>bold</b> <|end|>")).toBeTruthy();
    expect(document.querySelector("section b")).toBeNull();
  });

  it("shows a group's satisfaction as a face and a percentage", () => {
    const view = viewOf(launchedWorld());
    const product = view.me.company?.products[0]?.id ?? "";
    const sat = view.me.company?.satisfaction[product];
    if (!sat) throw new Error("no satisfaction");
    sat.students = 0.8;
    render(<Market view={view} />);
    const card = screen.getByRole("region", { name: "Students" });
    expect(within(card).getByText("80%")).toBeTruthy();
    expect(within(card).getByRole("img", { name: "Happy" })).toBeTruthy();
  });

  it("shows how many people are willing to use an AI", () => {
    const view = viewOf(launchedWorld());
    render(<Market view={view} />);
    expect(screen.getByText(aboutPeople(view.world.market.adoptable))).toBeTruthy();
    expect(aboutPeople(33_512_000)).toBe("about 33.5 million");
  });
});

describe("market crowd", () => {
  beforeEach(() => useWindows.getState().closeAll());

  it("draws each group's customers as dots, one per step of people", () => {
    const view = viewOf(launchedWorld());
    render(<Market view={view} />);
    const step = crowdStep(Math.max(...GROUP_IDS.map((g) => groupCustomers(view, g))));
    for (const g of GROUP_IDS) {
      const card = screen.getByRole("region", { name: GROUPS[g].many });
      const n = groupCustomers(view, g);
      expect(card.querySelectorAll("[data-dot]")).toHaveLength(
        n > 0 ? Math.max(1, Math.round(n / step)) : 0,
      );
    }
    expect(
      screen.getByText(`Each dot is ${step === 1 ? "1 person" : `${formatCount(step)} people`}`),
    ).toBeTruthy();
  });

  it("picks a step of people that keeps every crowd small", () => {
    expect(crowdStep(0)).toBe(1);
    expect(crowdStep(80)).toBe(1);
    expect(crowdStep(81)).toBe(10);
    expect(crowdStep(800)).toBe(10);
    expect(crowdStep(801)).toBe(100);
    expect(crowdStep(2_500_000)).toBe(100_000);
  });

  it("opens a group's window from its card", async () => {
    const view = viewOf(launchedWorld());
    render(
      <>
        <Market view={view} />
        <WindowHost view={view} />
      </>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Open Students" }));
    expect(await screen.findByRole("dialog", { name: "Students" })).toBeTruthy();
  });
});
