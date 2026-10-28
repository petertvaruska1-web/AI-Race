import { GROUP_IDS } from "@airace/protocol";
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GROUPS } from "../src/groups.ts";
import { aboutPeople, Market } from "../src/screens/Market.tsx";
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
