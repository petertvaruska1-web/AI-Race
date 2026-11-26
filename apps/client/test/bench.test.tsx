import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Cartridge } from "../src/bench/Cartridge.tsx";
import { Display } from "../src/bench/Display.tsx";
import { HueDial } from "../src/bench/HueDial.tsx";
import { Keypad } from "../src/bench/Keypad.tsx";
import { KeyRow } from "../src/bench/KeyRow.tsx";
import { Knob } from "../src/bench/Knob.tsx";
import { Lamp } from "../src/bench/Lamp.tsx";
import { Lever } from "../src/bench/Lever.tsx";
import { Module } from "../src/bench/Module.tsx";
import { PushButton } from "../src/bench/PushButton.tsx";
import { Readout } from "../src/bench/Readout.tsx";
import { RotarySelector } from "../src/bench/RotarySelector.tsx";
import { Slider } from "../src/bench/Slider.tsx";
import { ToggleSwitch } from "../src/bench/ToggleSwitch.tsx";
import { Trace } from "../src/bench/Trace.tsx";
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

  it("draws profit by month as a trace", () => {
    render(<Trace values={[-2, 4, 9]} label="Profit by month" zeroLine />);
    expect(
      screen.getByRole("img", { name: "Profit by month" }).querySelector("polyline"),
    ).not.toBeNull();
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

describe("controls for setting things", () => {
  it("turns a knob with the keys and names its value", async () => {
    const onChange = vi.fn();
    render(<Knob label="Corners" value={6} min={3} max={12} onChange={onChange} />);
    const knob = screen.getByRole("slider", { name: "Corners" });
    expect(knob.getAttribute("aria-valuenow")).toBe("6");
    expect(knob.getAttribute("aria-valuetext")).toBe("6");
    knob.focus();
    await userEvent.keyboard("{ArrowUp}");
    expect(onChange).toHaveBeenLastCalledWith(7);
    await userEvent.keyboard("{Home}");
    expect(onChange).toHaveBeenLastCalledWith(3);
    await userEvent.keyboard("{End}");
    expect(onChange).toHaveBeenLastCalledWith(12);
  });

  it("turns a knob with the wheel only once the player has picked it", () => {
    const onChange = vi.fn();
    render(<Knob label="Glow" value={40} min={0} max={100} step={5} onChange={onChange} />);
    const knob = screen.getByRole("slider", { name: "Glow" });
    const wheel = (deltaY: number) => {
      const e = new WheelEvent("wheel", { deltaY, bubbles: true, cancelable: true });
      knob.dispatchEvent(e);
      return e;
    };
    // Scrolling the page past a knob changes nothing and still scrolls the page.
    expect(wheel(-100).defaultPrevented).toBe(false);
    expect(onChange).not.toHaveBeenCalled();
    knob.focus();
    expect(wheel(-100).defaultPrevented).toBe(true);
    expect(onChange).toHaveBeenLastCalledWith(45);
    // A trackpad's small steps add up before the knob moves a notch.
    onChange.mockClear();
    wheel(20);
    expect(onChange).not.toHaveBeenCalled();
    wheel(40);
    expect(onChange).toHaveBeenLastCalledWith(35);
  });

  it("turns a hue dial in steps big enough to go round in a few dozen notches", () => {
    const onChange = vi.fn();
    render(<HueDial label="Main color" hue={100} color="#ff0000" onChange={onChange} />);
    const dial = screen.getByRole("slider", { name: "Main color" });
    dial.focus();
    dial.dispatchEvent(new WheelEvent("wheel", { deltaY: -100, bubbles: true, cancelable: true }));
    expect(onChange).toHaveBeenLastCalledWith(109);
  });

  it("slides a slider", () => {
    const onChange = vi.fn();
    render(<Slider label="Ring weight" value={3} min={1} max={6} onChange={onChange} />);
    fireEvent.change(screen.getByRole("slider", { name: "Ring weight" }), {
      target: { value: "5" },
    });
    expect(onChange).toHaveBeenCalledWith(5);
  });

  it("turns a hue dial around the color wheel", async () => {
    const onChange = vi.fn();
    render(<HueDial label="Main color" hue={359} color="#ff0000" onChange={onChange} />);
    const dial = screen.getByRole("slider", { name: "Main color" });
    dial.focus();
    await userEvent.keyboard("{ArrowUp}");
    expect(onChange).toHaveBeenLastCalledWith(0);
  });

  it("presses a quick-color cartridge", async () => {
    const onClick = vi.fn();
    render(
      <Cartridge
        label="Ember"
        primary="#ff9f1c"
        secondary="#ff5964"
        pressed={false}
        onClick={onClick}
      />,
    );
    const cart = screen.getByRole("button", { name: "Ember" });
    expect(cart.getAttribute("aria-pressed")).toBe("false");
    await userEvent.click(cart);
    expect(onClick).toHaveBeenCalled();
  });

  it("pulls the lever once from the keyboard, never while disabled", async () => {
    const onPull = vi.fn();
    const { rerender } = render(<Lever label="Found company" disabled onPull={onPull} />);
    await userEvent.click(screen.getByRole("button", { name: "Found company" }));
    expect(onPull).not.toHaveBeenCalled();
    rerender(<Lever label="Found company" onPull={onPull} />);
    screen.getByRole("button", { name: "Found company" }).focus();
    await userEvent.keyboard("{Enter}");
    expect(onPull).toHaveBeenCalledTimes(1);
    rerender(<Lever label="Found company" down onPull={onPull} />);
    screen.getByRole("button", { name: "Found company" }).focus();
    await userEvent.keyboard(" ");
    expect(onPull).toHaveBeenCalledTimes(1);
  });

  it("types a code from the keys and cleans a pasted one", async () => {
    const onChange = vi.fn();
    const pad = (value: string) => (
      <Keypad
        label="Invite code"
        alphabet="ABC23"
        value={value}
        maxLength={6}
        onChange={onChange}
      />
    );
    const { rerender } = render(pad(""));
    await userEvent.click(screen.getByRole("button", { name: "A" }));
    expect(onChange).toHaveBeenLastCalledWith("A");
    rerender(pad("A"));
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(onChange).toHaveBeenLastCalledWith("");
    fireEvent.change(screen.getByRole("textbox", { name: "Invite code" }), {
      target: { value: " ab c2 " },
    });
    expect(onChange).toHaveBeenLastCalledWith("ABC2");
  });
});
