import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { formatCount, formatMoney } from "../src/format.ts";
import { Company } from "../src/screens/Company.tsx";
import { useSession } from "../src/store/session.ts";
import { liveGame } from "./mockSocket.ts";
import { apply, feedback, launchedWorld, runningWorld, trainedWorld, viewOf } from "./views.ts";

beforeEach(async () => {
  useSession.setState({ socket: (await liveGame()).socket });
});

describe("company screen", () => {
  it("shows a launched AI's customers and the company's cash", () => {
    const view = viewOf(launchedWorld());
    const company = view.me.company;
    expect(company?.customers.total).toBeGreaterThan(0);
    render(<Company view={view} />);
    expect(screen.getByText(formatCount(company?.customers.total ?? -1))).toBeTruthy();
    expect(screen.getByText(formatMoney(company?.cash ?? -1))).toBeTruthy();
  });

  it("quotes a customer's post exactly as the AI wrote it, as text", () => {
    const s = apply(
      launchedWorld(),
      feedback("w9", [
        ["students", 0.9, "<b>x</b> <|end|>"],
        ["developers", 0.1, "No idea."],
      ]),
    );
    const { container } = render(<Company view={viewOf(s)} />);
    expect(screen.getByText("<b>x</b> <|end|>")).toBeTruthy();
    expect(container.querySelector("blockquote b")).toBeNull();
  });

  it("folds a long reply behind Show more, keeping all of it", () => {
    const long = "x".repeat(2000);
    const s = apply(
      launchedWorld(),
      feedback("w9", [
        ["students", 0.9, long],
        ["developers", 0.1, "No idea."],
      ]),
    );
    const { container } = render(<Company view={viewOf(s)} />);
    expect(screen.getByRole("button", { name: "Show more" })).toBeTruthy();
    expect(container.textContent).toContain(long);
  });

  it("shows the spark and asks for a first AI before there is one", () => {
    render(<Company view={viewOf(runningWorld())} />);
    expect(screen.getByRole("img", { name: /spark/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Train your first AI" })).toBeTruthy();
  });

  it("holds the launch until the newest AI has been scored", () => {
    render(<Company view={viewOf(trainedWorld())} />);
    const launch = screen.getByRole("button", { name: "Launch" }) as HTMLButtonElement;
    expect(launch.disabled).toBe(true);
    expect(launch.title).toMatch(/Wait until your AI has been scored/);
    expect(screen.getByText(/Wait until your AI has been scored/)).toBeTruthy();
  });
});
