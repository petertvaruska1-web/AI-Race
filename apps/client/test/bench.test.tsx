import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Display } from "../src/bench/Display.tsx";
import { KeyRow } from "../src/bench/KeyRow.tsx";
import { Lamp } from "../src/bench/Lamp.tsx";
import { Module } from "../src/bench/Module.tsx";
import { PushButton } from "../src/bench/PushButton.tsx";
import { Readout } from "../src/bench/Readout.tsx";
import { RotarySelector } from "../src/bench/RotarySelector.tsx";
import { ToggleSwitch } from "../src/bench/ToggleSwitch.tsx";
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

  it("keeps display numbers clear of the crowd's bobbing dots", () => {
    render(<Readout label="Cash" value="$1,200" testId="cash" />);
    expect(screen.getByTestId("cash").classList.contains("dot")).toBe(false);
    render(<Stat label="Customers" value={3} format={String} testId="count" />);
    expect(screen.getByTestId("count").classList.contains("dot")).toBe(false);
  });

  it("shows a stat's value in a display, keeping its test id", () => {
    render(<Stat label="Cash" value={1200} format={(n) => `$${n}`} testId="cash" />);
    expect(screen.getByTestId("cash").closest(".display")).not.toBeNull();
  });
});

describe("bench controls", () => {
  it("presses a primary button and keeps a disabled reason", async () => {
    const onClick = vi.fn();
    render(
      <>
        <PushButton onClick={onClick}>Train</PushButton>
        <PushButton disabled title="Train it first">
          Launch
        </PushButton>
      </>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Train" }));
    expect(onClick).toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Launch" }).getAttribute("title")).toBe(
      "Train it first",
    );
  });

  it("flips a switch", async () => {
    const onChange = vi.fn();
    render(<ToggleSwitch label="Fair mode" checked={false} onChange={onChange} />);
    await userEvent.click(screen.getByRole("switch", { name: "Fair mode" }));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("picks a key in a key row", async () => {
    const onChange = vi.fn();
    render(
      <KeyRow
        legend="Difficulty"
        options={[
          { label: "Easy", value: "easy" },
          { label: "Hard", value: "hard" },
        ]}
        value="easy"
        onChange={onChange}
      />,
    );
    await userEvent.click(screen.getByRole("radio", { name: "Hard" }));
    expect(onChange).toHaveBeenCalledWith("hard");
  });

  it("turns a rotary selector with the arrow keys", async () => {
    const onChange = vi.fn();
    render(
      <RotarySelector
        legend="Step"
        options={[
          { label: "Data", value: "data" },
          { label: "Prepare", value: "prepare" },
        ]}
        value="data"
        onChange={onChange}
      />,
    );
    screen.getByRole("radio", { name: "Data" }).focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenCalledWith("prepare");
    expect(screen.getByRole("radiogroup", { name: "Step" })).toBeTruthy();
  });
});
