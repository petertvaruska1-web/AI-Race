import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Benchmarks, CATEGORIES } from "../src/screens/Benchmarks.tsx";
import { trainedWorld, viewOf } from "./views.ts";

const SCORES = {
  language: 41.5,
  reasoning: 12,
  pattern: 30,
  knowledge: 8,
  coding: 3,
  creativity: null,
  consistency: 55,
  instruction: 20,
};

describe("benchmarks", () => {
  it("shows all eight categories, with unmeasured ones said plainly", () => {
    render(<Benchmarks view={viewOf(trainedWorld(SCORES))} />);
    expect(CATEGORIES).toHaveLength(8);
    for (const c of CATEGORIES) {
      const card = within(screen.getByTestId(`score-${c.id}`));
      expect(card.getByText(c.name)).toBeTruthy();
    }
    expect(within(screen.getByTestId("score-creativity")).getByText("Not measured")).toBeTruthy();
    expect(within(screen.getByTestId("score-language")).getByText("42")).toBeTruthy();
    expect(screen.getByTestId("overall").textContent).toBe("30");
    expect(screen.getByText("curious")).toBeTruthy();
    expect(screen.getByText(/0 means no better than guessing/)).toBeTruthy();
  });

  it("says when a version is still being scored", () => {
    render(<Benchmarks view={viewOf(trainedWorld())} />);
    expect(screen.getByText(/Scoring v1/)).toBeTruthy();
  });
});
