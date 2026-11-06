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

  // Phase 4: the goal's victory screen can open at any moment once 1,000 people use the AI;
  // "Keep playing" takes the player back to where they were.
  let victory = false;
  await page.addLocatorHandler(
    page.getByRole("dialog", { name: "You reached your goal" }),
    async () => {
      victory = true;
      await page.getByRole("button", { name: "Keep playing" }).click();
    },
  );

  // 1–2. Sign up, start a world with the goal "Reach 1,000 customers", found a company in a
  // preset palette.
  await page.goto(stack.url);
  await page.getByRole("textbox").fill("Ana");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.getByLabel("World name").fill("Solo test");
  await page.getByText("Reach 1,000,000 customers").click();
  await page.getByLabel("Goal target").fill("1000");
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
  await page.keyboard.press("Escape");

  // 23. Playtest pack: the fund takes money out of cash and shows it. Paused, since the fund's
  // value moves every game day.
  await page.getByRole("button", { name: "Pause" }).click();
  await page.getByLabel("Amount").fill("1000");
  await page.getByRole("button", { name: "Invest", exact: true }).click();
  await expect(page.getByTestId("fund")).toHaveText("$1,000");
  await page.getByRole("button", { name: "Resume" }).click();

  // 24. A logo motion is bought from the Company screen's Customize window.
  await page.getByRole("link", { name: "Company" }).click();
  await page.getByRole("button", { name: /Customize your look/ }).click();
  const looks = page.getByRole("dialog", { name: /Customize/ });
  await looks
    .getByRole("region", { name: "Logo motion" })
    .getByRole("button", { name: /^Buy for/ })
    .first()
    .click();
  await expect(
    looks.getByRole("region", { name: "Logo motion" }).getByText("Wearing it"),
  ).toBeVisible();
  await page.keyboard.press("Escape");

  // 25. A practice Pattern Bridge: the real AI picks every tile; the run ends with a score.
  await page.getByRole("link", { name: "Playground" }).click();
  // The pill's label covers its hidden radio, so the label is what a player clicks.
  await page.getByRole("group", { name: "Playground mode" }).getByText("Games").click();
  await page.getByRole("button", { name: "Play Pattern Bridge" }).click();
  await page.getByRole("button", { name: "Practice (free)" }).click();
  await expect(page.getByText("What comes next?")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByTestId("result")).toBeVisible({ timeout: 180_000 });
  await expect(page.getByTestId("score")).toHaveText(/^\d+ of 12$/);

  // 26. Phase 4: a notebook line, saved privately; the Data step counts its words.
  await page.getByRole("link", { name: "AI Lab" }).click();
  await page.getByRole("button", { name: "1 Data" }).click();
  const notebook = page.getByTestId("dataset-notebook");
  await notebook.getByRole("button", { name: "Write" }).click();
  await page.getByLabel("Notebook text").fill("Ember Labs makes friendly AIs.");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByText("Saved. Only your company's AI reads this.")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(notebook.getByTestId("notebook-words")).toHaveText("5 / 5,000 words");

  // 27. The first scored AI has a discovery (the test balance's thresholds); publishing it
  // makes a story in the feed.
  await page.getByRole("button", { name: "5 Versions" }).click();
  await page
    .getByTestId("version-v1")
    .getByRole("button", { name: /^(Strength|Weakness|Better|Worse|Changed)/ })
    .first()
    .click();
  const found = page.getByRole("dialog");
  await found.getByRole("button", { name: "Publish as news" }).click();
  await expect(found.getByText("Published")).toBeVisible();
  await page.keyboard.press("Escape");
  await page.getByRole("link", { name: "Company" }).click();
  await expect(feed.getByText(/Ember Labs(’s|'s| admits)/).first()).toBeVisible();

  // 28. An interview: a prompt card, the AI's real answer, ticked and published; the press
  // writes it up.
  await page.getByRole("link", { name: "Playground" }).click();
  await page.getByRole("group", { name: "Playground mode" }).getByText("Interview").click();
  await page.getByRole("button", { name: /^Tell us a joke\./ }).click();
  await page.getByRole("checkbox", { name: "Publish this" }).check({ timeout: 120_000 });
  await page.getByRole("button", { name: "Publish interview" }).click();
  const interview = page.getByTestId("interview-result");
  await expect(interview.getByText(/interview/)).toBeVisible({ timeout: 120_000 });

  // 29. The trophy shelf holds what was earned, and the goal was reached along the way.
  await page.getByRole("link", { name: "Company" }).click();
  await expect(page.getByTestId("trophy-shelf")).toContainText("First paying customer");
  await expect.poll(() => victory, { timeout: 60_000 }).toBe(true);
  await expect(page.getByTestId("goal-progress")).toContainText("Reached on");

  // 30. Coaching: three replies, each liked and given a better answer (six lessons), then a
  // run that trains on them.
  await page.getByRole("link", { name: "Playground" }).click();
  await page.getByRole("group", { name: "Playground mode" }).getByText("Chat").click();
  // Your own AI (the Playground may still be on the rival's from step 20): only yours is coached.
  await page.getByLabel("Talk to").selectOption({ index: 0 });
  await page.getByRole("button", { name: "New conversation" }).click();
  const log = page.getByTestId("chat-log");
  for (let k = 0; k < 3; k++) {
    await page.getByLabel("Your message").fill(`Question ${k + 1}?`);
    await page.getByRole("button", { name: "Send" }).click();
    const replyK = log.getByRole("listitem").nth(2 * k + 1);
    await expect(replyK.getByRole("button", { name: "👍 Good answer" })).toBeVisible({
      timeout: 120_000,
    });
    await replyK.getByRole("button", { name: "👍 Good answer" }).click();
    await expect(replyK.getByText("Saved as a lesson.")).toBeVisible();
    await replyK.getByRole("button", { name: "✏️ Write a better answer" }).click();
    await replyK.getByLabel("Your answer").fill(`The answer to question ${k + 1}.`);
    await replyK.getByRole("button", { name: "Save" }).click();
    await expect(replyK.getByLabel("Your answer")).toHaveCount(0);
  }
  await page.getByRole("link", { name: "AI Lab" }).click();
  await page.getByRole("button", { name: "1 Data" }).click();
  await expect(page.getByText("6 lessons")).toBeVisible();
  await page.getByLabel("Train on your lessons").check();
  await page.getByRole("button", { name: "Start training" }).click();
  await expect(page.getByText("Training", { exact: true })).toBeVisible({ timeout: 60_000 });
});
