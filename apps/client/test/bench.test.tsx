import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Display } from "../src/bench/Display.tsx";
import { Lamp } from "../src/bench/Lamp.tsx";
import { Module } from "../src/bench/Module.tsx";
import { Readout } from "../src/bench/Readout.tsx";
import { Panel } from "../src/components/Panel.tsx";
import { Stat } from "../src/components/Stat.tsx";

describe("bench surfaces", () => {
  it("engraves a module's label as its heading", () => {
    render(<Module label="Customers">x</Module>);
    expect(screen.getByRole("heading", { name: "Customers" }).className).toContain("engrave");
  });

  it("keeps Panel's title, hint and actions", () => {
    render(
      <Panel title="Money" hint="This month" actions={<button type="button">Open</button>}>
        y
      </Panel>,
    );
    expect(screen.getByRole("heading", { level: 2, name: "Money" })).toBeTruthy();
    expect(screen.getByText("This month")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Open" })).toBeTruthy();
  });

  it("lets a scrolling display be reached and named", () => {
    render(
      <Display scroll label="What people are saying">
        z
      </Display>,
    );
    const region = screen.getByRole("region", { name: "What people are saying" });
    expect(region.tabIndex).toBe(0);
  });

  it("names a lamp by its state", () => {
    render(<Lamp state="ready" label="Ready to found" />);
    expect(screen.getByRole("img", { name: "Ready to found" }).dataset.state).toBe("ready");
  });

  it("shows a readout's value in the dot face", () => {
    render(<Readout label="Cash" value="$1,200" testId="cash" />);
    expect(screen.getByTestId("cash").className).toContain("dot");
    expect(screen.getByText("Cash").className).toContain("engrave");
  });

  it("shows a stat's value in a display, keeping its test id", () => {
    render(<Stat label="Cash" value={1200} format={(n) => `$${n}`} testId="cash" />);
    expect(screen.getByTestId("cash").closest(".display")).not.toBeNull();
  });
});
