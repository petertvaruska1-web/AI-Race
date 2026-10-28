import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { AppShell } from "../src/components/AppShell.tsx";
import { formatCount, formatMoney, monthName } from "../src/format.ts";
import { useSession } from "../src/store/session.ts";
import { useWindows, type WindowSpec } from "../src/store/windows.ts";
import { WindowHost } from "../src/windows/WindowHost.tsx";
import { liveGame } from "./mockSocket.ts";
import { apply, feedback, launchedWorld, SCORES, ticks, trainedWorld, viewOf } from "./views.ts";

const total = (r: Record<string, number>) => Object.values(r).reduce((a, b) => a + b, 0);

beforeEach(async () => {
  useSession.setState({ socket: (await liveGame()).socket });
  useWindows.getState().closeAll();
});

function Opener({ spec }: { spec: WindowSpec }) {
  const open = useWindows((s) => s.open);
  return (
    <button type="button" onClick={(e) => open(spec, e.currentTarget)}>
      Open it
    </button>
  );
}

function openFrom(spec: WindowSpec, view = viewOf(launchedWorld())) {
  render(
    <>
      <Opener spec={spec} />
      <WindowHost view={view} />
    </>,
  );
  return userEvent.click(screen.getByRole("button", { name: "Open it" }));
}

describe("windows", () => {
  it("open from what was clicked; Escape closes and gives focus back", async () => {
    await openFrom({ kind: "customers" });
    expect(screen.getByRole("dialog", { name: "Customers" })).toBeTruthy();
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Open it" }));
  });

  it("stack a window opened inside another; Back returns to the first", async () => {
    await openFrom({ kind: "customers" });
    const first = screen.getByRole("dialog", { name: "Customers" });
    await userEvent.click(within(first).getByRole("button", { name: "Open Students" }));
    const second = await screen.findByRole("dialog", { name: "Students" });
    await userEvent.click(within(second).getByRole("button", { name: "Back" }));
    expect(await screen.findByRole("dialog", { name: "Customers" })).toBeTruthy();
  });

  it("all close from the backdrop or the close button", async () => {
    await openFrom({ kind: "customers" });
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Open Students" }),
    );
    await screen.findByRole("dialog", { name: "Students" });
    await userEvent.click(screen.getByRole("button", { name: "Close all windows" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(useWindows.getState().stack).toEqual([]);

    await userEvent.click(screen.getByRole("button", { name: "Open it" }));
    await userEvent.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Close" }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("all close when the player changes screens", async () => {
    const view = viewOf(launchedWorld());
    const { rerender } = render(
      <AppShell view={view} tab="company">
        <Opener spec={{ kind: "customers" }} />
      </AppShell>,
    );
    await userEvent.click(screen.getByRole("button", { name: "Open it" }));
    expect(screen.getByRole("dialog", { name: "Customers" })).toBeTruthy();
    rerender(
      <AppShell view={view} tab="money">
        <p>Money</p>
      </AppShell>,
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
});

describe("customers window", () => {
  it("shows customers day by day, stacked by group", async () => {
    const view = viewOf(launchedWorld());
    const company = view.me.company;
    await openFrom({ kind: "customers" }, view);
    const w = screen.getByRole("dialog", { name: "Customers" });
    expect(within(w).getAllByText(formatCount(company?.customers.total ?? -1)).length).toBe(1);
    const chart = within(w).getByRole("img", { name: "Customers by day" });
    expect(chart.getAttribute("data-points")).toBe(
      String(Object.keys(company?.history ?? {}).length),
    );
  });
});

describe("group window", () => {
  it("shows the group's customers, its spotlight person and only its own posts", async () => {
    const s = apply(
      launchedWorld(),
      feedback("w9", [
        ["students", 0.9, "Paris, of course."],
        ["developers", 0.1, "No idea."],
      ]),
    );
    const view = viewOf(s);
    await openFrom({ kind: "group", group: "students" }, view);
    const w = screen.getByRole("dialog", { name: "Students" });
    const students = Object.values(view.me.company?.customers.byGroup.students ?? {}).reduce(
      (a, b) => a + b,
      0,
    );
    expect(within(w).getByTestId("group-customers").textContent).toBe(formatCount(students));
    const person = view.spotlights.find((p) => p.group === "students");
    expect(within(within(w).getByTestId("spotlight")).getByText(person?.name ?? "?")).toBeTruthy();
    expect(within(w).getByText("Paris, of course.")).toBeTruthy();
    expect(within(w).queryByText("No idea.")).toBeNull();
    expect(within(w).getByRole("img", { name: "Students by day" })).toBeTruthy();
  });
});

describe("product window", () => {
  it("shows the product's plans and the AI behind it, which opens in its own window", async () => {
    const view = viewOf(launchedWorld());
    const version = view.me.company?.versions[0];
    await openFrom({ kind: "product", productId: "p1" }, view);
    const w = screen.getByRole("dialog", { name: "Spark" });
    expect(within(w).getByText("On sale")).toBeTruthy();
    expect(within(w).getByText("Free")).toBeTruthy();
    expect(
      within(w).getByText(`Personal · ${formatMoney(view.me.company?.plans.personal.price ?? -1)}`),
    ).toBeTruthy();
    await userEvent.click(within(w).getByRole("button", { name: `About ${version?.label}` }));
    expect(await screen.findByRole("dialog", { name: `Your AI ${version?.label}` })).toBeTruthy();
  });
});

describe("AI window", () => {
  it("shows every score as a bar and how much it has read", async () => {
    const view = viewOf(trainedWorld(SCORES));
    const version = view.me.company?.versions[0];
    await openFrom({ kind: "ai", versionId: version?.id ?? "?" }, view);
    const w = screen.getByRole("dialog", { name: `Your AI ${version?.label}` });
    const language = within(w).getByRole("meter", { name: "Language" });
    expect(language.getAttribute("aria-valuenow")).toBe("60");
    expect(within(w).getAllByRole("meter")).toHaveLength(Object.keys(SCORES).length);
    expect(within(w).getByText(/million words|thousand words/)).toBeTruthy();
  });
});

describe("post window", () => {
  it("shows the whole reply, unfolded, and opens the poster's group", async () => {
    const long = "y".repeat(900);
    const s = apply(
      launchedWorld(),
      feedback("w9", [
        ["students", 0.9, long],
        ["developers", 0.1, "No idea."],
      ]),
    );
    const view = viewOf(s);
    const post = view.feed.find((f) => f.kind === "post" && f.reply === long);
    await openFrom({ kind: "post", postId: post?.id ?? "?" }, view);
    const w = screen.getByRole("dialog");
    expect(within(w).getByText(long)).toBeTruthy();
    expect(within(w).queryByRole("button", { name: "Show more" })).toBeNull();
    await userEvent.click(within(w).getByRole("button", { name: "Open Students" }));
    expect(await screen.findByRole("dialog", { name: "Students" })).toBeTruthy();
  });
});

describe("money windows", () => {
  const view = () => viewOf(ticks(launchedWorld(), 40));

  it("money shows cash day by day and profit by month; a month opens its own window", async () => {
    const v = view();
    const company = v.me.company;
    if (!company) throw new Error("no company");
    await openFrom({ kind: "money" }, v);
    const w = screen.getByRole("dialog", { name: "Money" });
    expect(within(w).getByRole("img", { name: "Cash by day" }).getAttribute("data-points")).toBe(
      String(Object.keys(company.history).length),
    );
    const closed = company.ledger.months.at(-1);
    if (!closed) throw new Error("no closed month");
    const profit = total(closed.income) - total(closed.costs);
    await userEvent.click(
      within(w).getByRole("button", { name: `${monthName(closed.month)}: ${formatMoney(profit)}` }),
    );
    const m = await screen.findByRole("dialog", { name: monthName(closed.month) });
    expect(within(m).getByTestId("month-profit").textContent).toBe(formatMoney(profit));
    expect(within(m).getByTestId("month-customers").textContent).toBe(
      formatCount(closed.customersEnd),
    );
    expect(within(m).getByTestId("in-personal").textContent).toBe(
      formatMoney(closed.income.personal),
    );
    expect(within(m).getByTestId("out-serving").textContent).toBe(
      formatMoney(closed.costs.serving),
    );
  });

  it("a money category shows each month, and each month opens", async () => {
    const v = view();
    const company = v.me.company;
    if (!company) throw new Error("no company");
    await openFrom({ kind: "cost", category: "serving" }, v);
    const w = screen.getByRole("dialog", { name: "Serving customers" });
    const months = [...company.ledger.months, company.ledger.current];
    const bars = within(
      within(w).getByRole("group", { name: "Serving customers by month" }),
    ).getAllByRole("button");
    expect(bars).toHaveLength(months.length);
    await userEvent.click(bars[0] as HTMLElement);
    expect(
      await screen.findByRole("dialog", { name: monthName(months[0]?.month ?? "") }),
    ).toBeTruthy();
  });
});
