import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DayChart } from "../src/components/charts/DayChart.tsx";
import { MonthBars } from "../src/components/charts/MonthBars.tsx";
import { formatCount, formatMoney } from "../src/format.ts";

const days = Array.from({ length: 100 }, (_, i) => i + 1);
const series = [
  { id: "a", label: "Students", color: "#4f8cff", values: days.map((d) => d * 2) },
  { id: "b", label: "Creators", color: "#ffb020", values: days.map((d) => d) },
];

describe("day chart", () => {
  it("shows every day, or only the last 30 or 90", async () => {
    render(<DayChart label="Customers by day" days={days} series={series} format={formatCount} />);
    const chart = screen.getByRole("img", { name: "Customers by day" });
    expect(chart.getAttribute("data-points")).toBe("100");
    await userEvent.click(screen.getByRole("button", { name: "30 days" }));
    expect(chart.getAttribute("data-points")).toBe("30");
    await userEvent.click(screen.getByRole("button", { name: "90 days" }));
    expect(chart.getAttribute("data-points")).toBe("90");
  });

  it("reads out every series for the day under the pointer, with the total when stacked", () => {
    render(
      <DayChart
        label="Customers by day"
        days={days}
        series={series}
        format={formatCount}
        stacked
      />,
    );
    const chart = screen.getByRole("img", { name: "Customers by day" });
    const surface = chart.querySelector("[data-hover-surface]");
    if (!surface) throw new Error("no hover surface");
    fireEvent.pointerMove(surface, { clientX: 10_000, clientY: 10 });
    const readout = screen.getByTestId("readout");
    expect(within(readout).getByText("Students")).toBeTruthy();
    expect(within(readout).getByText(formatCount(200))).toBeTruthy();
    expect(within(readout).getByText(formatCount(100))).toBeTruthy();
    expect(within(readout).getByText(formatCount(300))).toBeTruthy();
    fireEvent.pointerLeave(surface);
    expect(screen.queryByTestId("readout")).toBeNull();
  });

  it("says when there is no history yet", () => {
    render(<DayChart label="Cash by day" days={[]} series={[]} format={formatMoney} />);
    expect(screen.getByText(/starts tomorrow/)).toBeTruthy();
  });
});

describe("month bars", () => {
  it("makes each month a button that says its value and opens it", async () => {
    const onOpen = vi.fn();
    render(
      <MonthBars
        label="Profit by month"
        bars={[
          { month: "2027-01", value: -1200 },
          { month: "2027-02", value: 3400 },
        ]}
        format={formatMoney}
        onOpen={onOpen}
      />,
    );
    const feb = screen.getByRole("button", { name: `February 2027: ${formatMoney(3400)}` });
    await userEvent.click(feb);
    expect(onOpen).toHaveBeenCalledWith("2027-02", feb);
    expect(
      screen.getByRole("button", { name: `January 2027: ${formatMoney(-1200)}` }),
    ).toBeTruthy();
  });
});
