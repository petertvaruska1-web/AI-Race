import { PALETTE_PRESETS } from "@airace/sim";
import { expect, test } from "./fixtures.ts";

// Stage 1 end to end: one player, a real server, a real CPU worker training a real (tiny) model.

const EMBER = PALETTE_PRESETS.find((p) => p.id === "ember")?.palette.primary ?? "";
// The test balance's model (2 layers of 64, 65,536 words) costs $58 of the $250,000 founding cash.
const CASH_AFTER_ONE_RUN = "$249,942";

test("a player founds a company, trains an AI, survives a reload and a restart, then talks to it", async ({
  page,
  stack,
}) => {
  const lossPoints = async () =>
    Number(await page.getByTestId("loss-chart").getAttribute("data-points"));

  // 1–2. Sign up, start a world, found a company in a preset palette.
  await page.goto(stack.url);
  await page.getByRole("textbox").fill("Ana");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByLabel("World name").fill("Solo test");
  await page.getByRole("button", { name: "Start" }).click();
  await page.getByPlaceholder("Acme Intelligence").fill("Ember Labs");
  await page.getByRole("button", { name: /Ember/ }).click();
  await page.getByRole("button", { name: "Found company" }).click();
  await expect(page.getByRole("button", { name: "Open for business" })).toBeVisible();
  const brand = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue("--brand"),
  );
  expect(brand.trim()).toBe(EMBER);

  // 3. Open for business and start training with the defaults; pause at once so the run stays mid-way.
  await page.getByRole("button", { name: "Open for business" }).click();
  await page.getByRole("link", { name: "AI Lab" }).click();
  await page.getByRole("button", { name: "Start training" }).click();
  await page.getByRole("button", { name: "Pause" }).click();
  await expect(page.getByText("Training", { exact: true })).toBeVisible(); // a worker took it
  await stack.workerLog(/finished/); // the worker finished; the game reveals it as days pass

  // 4. Resume until the loss chart has points, then pause again.
  await page.getByRole("button", { name: "Resume" }).click();
  await expect.poll(lossPoints).toBeGreaterThan(0);
  await page.getByRole("button", { name: "Pause" }).click();
  const shown = await lossPoints();

  // 5. Reload mid-run: the same backlog comes back.
  await page.reload();
  await page.getByRole("button", { name: "4 Train" }).click();
  await expect.poll(lossPoints).toBe(shown);

  // 6. Restart the server mid-run: the page reconnects and the run completes.
  await stack.restartServer();
  await expect(page.getByRole("button", { name: "Resume" })).toBeVisible({ timeout: 60_000 });
  await page.getByRole("button", { name: "Resume" }).click();
  await expect(page.getByText(/Done: meet v1/)).toBeVisible();
  await expect.poll(lossPoints).toBeGreaterThanOrEqual(shown);
  await expect(page.getByTestId("sample-feed").getByRole("listitem").first()).toBeVisible();

  // 7. One version, one charge.
  await page.getByRole("button", { name: "5 Versions" }).click();
  await expect(page.getByTestId("version-v1")).toHaveCount(1);
  await expect(page.getByText(CASH_AFTER_ONE_RUN)).toBeVisible();

  // 8. Talk to it: the reply streams in.
  await page.getByRole("link", { name: "Playground" }).click();
  await page.getByLabel("Your message").fill("Hello");
  await page.getByRole("button", { name: "Send" }).click();
  const reply = page.getByTestId("chat-log").getByRole("listitem").nth(1);
  await expect(reply).not.toContainText("Thinking…", { timeout: 120_000 });
  expect(((await reply.locator("p").textContent()) ?? "").length).toBeGreaterThan(0);

  // 9. Benchmarks: eight categories once it is scored.
  await page.getByRole("link", { name: "Benchmarks" }).click();
  await expect(page.getByTestId("overall")).toBeVisible({ timeout: 300_000 });
  await expect(page.locator('[data-testid^="score-"]')).toHaveCount(8);

  // 10. Launch it as "Spark" on the free and personal plans, with a 3-day countdown.
  await page.getByRole("link", { name: "Company" }).click();
  await page.getByRole("button", { name: "Launch", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByLabel("Product name").fill("Spark");
  await dialog.getByRole("slider").fill("3");
  await dialog.getByRole("button", { name: "Launch", exact: true }).click();
  await expect(dialog).toBeHidden();

  // 11. Once it is live, customers arrive, the news covers the launch, and customers post
  // about it, quoting what the AI really said.
  const customers = async () =>
    Number(((await page.getByTestId("customers").textContent()) ?? "").replace(/\D/g, ""));
  await expect.poll(customers, { timeout: 60_000 }).toBeGreaterThan(0);
  // The feed has it (a toast may show the same headline for a few seconds).
  const feed = page.getByRole("list", { name: "News and posts" });
  await expect(feed.getByText(/Ember Labs launches Spark/)).toBeVisible();
  const quote = feed.locator("blockquote").first();
  await expect(quote).toBeVisible({ timeout: 300_000 });
  expect(((await quote.textContent()) ?? "").trim().length).toBeGreaterThan(0);

  // 12. The Money screen shows what the personal plan has earned.
  await page.getByRole("link", { name: "Money" }).click();
  await expect
    .poll(async () => (await page.getByTestId("income-personal").textContent())?.trim(), {
      timeout: 60_000,
    })
    .not.toBe("$0");

  // 13. A reload keeps the customers and the feed.
  await page.getByRole("link", { name: "Company" }).click();
  await expect.poll(customers).toBeGreaterThan(0);
  const before = await customers();
  await page.reload();
  await expect.poll(customers).toBeGreaterThanOrEqual(before);
  await expect(feed.getByText(/Ember Labs launches Spark/)).toBeVisible();
  await expect(
    page.getByRole("list", { name: "News and posts" }).locator("blockquote").first(),
  ).toBeVisible();

  // 14. Phase 2: customers spread over the regions on the map; each region opens its window.
  await page.getByRole("link", { name: "Market" }).click();
  const map = page.getByRole("group", { name: "Your customers by region" });
  const regionCustomers = async (name: string) => {
    await map.getByRole("button", { name: `Open ${name}` }).click();
    const n = Number(
      ((await page.getByTestId("region-customers").textContent()) ?? "").replace(/\D/g, ""),
    );
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog")).toHaveCount(0);
    return n;
  };
  expect(await regionCustomers("Pacific")).toBeGreaterThan(0);
  expect(await regionCustomers("Mid-Atlantic")).toBeGreaterThan(0);

  // 15. An ad campaign in the Pacific, listed on the Market and paid for on the Money screen.
  await page.getByRole("button", { name: "New ad campaign" }).click();
  const ad = page.getByRole("dialog", { name: "New ad campaign" });
  await ad.getByLabel("Where").selectOption("pacific");
  await ad.getByRole("button", { name: "Start the campaign" }).click();
  await expect(ad).toHaveCount(0);
  const campaigns = page.getByRole("region", { name: "Ad campaigns" });
  await expect(campaigns.getByText("Pacific · everyone")).toBeVisible();
  await page.getByRole("link", { name: "Money" }).click();
  await expect
    .poll(async () => (await page.getByTestId("cost-marketing").textContent())?.trim(), {
      timeout: 30_000,
    })
    .not.toBe("$0");

  // 16. Months close in seconds on the test balance: a report card opens with its grade.
  const cards = page.getByRole("region", { name: "Report cards" });
  await expect(cards.getByRole("button").first()).toBeVisible({ timeout: 60_000 });
  await cards.getByRole("button").first().click();
  await expect(page.getByRole("dialog", { name: /report card$/ })).toBeVisible();
  await expect(page.getByTestId("grade")).toHaveText(/^[A-F]$/);
  await page.keyboard.press("Escape");

  // 17. The computers say how many questions get answered.
  const computers = page.getByRole("region", { name: "Computers" });
  await expect(computers.getByText("of questions answered")).toBeVisible();

  // 18. Phase 3: four rivals, the arch-rival first.
  await page.getByRole("link", { name: "Rivals" }).click();
  const rivalCards = page.getByRole("article");
  await expect(rivalCards).toHaveCount(4);
  await expect(rivalCards.first()).toContainText("Arch-rival ·");

  // 19. A rival's launch is in the news.
  await page.getByRole("link", { name: "Company" }).click();
  await expect(feed.getByText(/launches/)).not.toHaveCount(1, { timeout: 60_000 });

  // 20. Try it: a rival's AI answers in the Playground, streamed.
  await page.getByRole("link", { name: "Rivals" }).click();
  await page.getByRole("button", { name: /^Try / }).first().click();
  await expect(page.getByText(/Say something to .+'s /)).toBeVisible();
  await page.getByLabel("Your message").fill("Who are you?");
  await page.getByRole("button", { name: "Send" }).click();
  const rivalReply = page.getByTestId("chat-log").getByRole("listitem").nth(1);
  await expect(rivalReply).not.toContainText("Thinking…", { timeout: 120_000 });

  // 21. A knowledge battle: both columns fill, then the result and the ranking points.
  await page.getByRole("link", { name: "Arena" }).click();
  await page.getByRole("button", { name: "Start a battle" }).click();
  const start = page.getByRole("dialog", { name: "Start a battle" });
  await start.getByRole("button", { name: "Start the battle" }).click();
  await expect(start).toHaveCount(0);
  await page.getByRole("button", { name: /^Open the knowledge battle/ }).click();
  const battle = page.getByRole("dialog", { name: /battle/ });
  await expect(battle.getByText(/^(Win|Loss|Draw) \d–\d$/)).toBeVisible({ timeout: 180_000 });
  await expect(battle.getByText(/ranking points$/)).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("table", { name: "Customers" }).getByRole("row")).toHaveCount(6);

  // 22. The next report card places the company against its rivals.
  await page.getByRole("link", { name: "Money" }).click();
  await page.getByRole("region", { name: "Report cards" }).getByRole("button").first().click();
  const place = /^Customers: (\d+(st|nd|rd|th) → )?\d+(st|nd|rd|th)$/;
  await expect(page.getByRole("dialog").getByText(place)).toBeVisible({ timeout: 60_000 });
});
