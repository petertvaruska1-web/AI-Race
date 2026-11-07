import type { Page } from "@playwright/test";
import { expect, type FriendsStack, test } from "./friends.fixtures.ts";

// Stage 2 end to end: two players in two browsers, each with their own helper (CPU workers on
// this computer), in a world with Fair mode off and in one with it on.

const NOTEBOOK = "Ana's secret recipe: kindness and tea.";

async function signUp(page: Page, url: string, name: string) {
  await page.goto(url);
  await page.getByRole("textbox").fill(name);
  await page.getByRole("button", { name: "Continue" }).click();
}

async function createWorld(page: Page, name: string, fair: boolean) {
  await page.getByLabel("World name").fill(name);
  if (fair) await page.getByLabel("Fair mode").check();
  await page.getByRole("button", { name: "Start", exact: true }).click();
}

async function found(page: Page, name: string) {
  await page.getByPlaceholder("Acme Intelligence").fill(name);
  await page.getByRole("button", { name: "Found company" }).click();
}

async function joinByLink(page: Page, url: string, code: string, name: string, notice: boolean) {
  await page.goto(`${url}/join/${code}`);
  await page.getByRole("textbox").fill(name);
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("button", { name: "Join", exact: true })).toBeVisible();
  await expect(page.getByText(/Play with people you trust/)).toHaveCount(notice ? 1 : 0);
  await page.getByRole("button", { name: "Join", exact: true }).click();
}

async function helperKey(page: Page, url: string): Promise<string> {
  const res = await page.request.get(`${url}/api/helper/key`);
  return ((await res.json()) as { key: string }).key;
}

async function startTraining(page: Page) {
  await page.getByRole("link", { name: "AI Lab" }).click();
  await page.getByRole("button", { name: "Start training" }).click();
}

async function trained(page: Page) {
  await expect(page.getByText(/Done: meet v1/)).toBeVisible({ timeout: 300_000 });
}

async function tryShared(owner: Page, friend: Page) {
  await owner.getByRole("button", { name: "5 Versions" }).click();
  await owner.getByRole("button", { name: "Share with friends" }).click();
  await expect(owner.getByRole("button", { name: "Stop sharing" })).toBeVisible();
  await friend.getByRole("link", { name: "AI Directory" }).click();
  await friend.getByRole("button", { name: "Try it" }).first().click();
  await expect(friend.getByText("Its maker's computer answers your messages.")).toBeVisible();
  await friend.getByLabel("Your message").fill("Hello, friend's AI");
  await friend.getByRole("button", { name: "Send" }).click();
  const reply = friend.getByTestId("chat-log").getByRole("listitem").nth(1);
  await expect(reply).not.toContainText("Thinking…", { timeout: 180_000 });
}

/** Everything a page receives (socket frames and HTTP bodies), for the opacity check. */
function recordTraffic(page: Page): string[] {
  const seen: string[] = [];
  page.on("websocket", (ws) => ws.on("framereceived", (f) => seen.push(String(f.payload))));
  page.on("response", async (r) => {
    try {
      seen.push(await r.text());
    } catch {
      // a body that can't be read (a redirect, a closed page)
    }
  });
  return seen;
}

async function scoredAndLaunched(page: Page, name: string) {
  await page.getByRole("link", { name: "Benchmarks" }).click();
  await expect(page.getByTestId("overall")).toBeVisible({ timeout: 400_000 });
  await page.getByRole("link", { name: "Company" }).click();
  await page.getByRole("button", { name: "Launch", exact: true }).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByLabel("Product name").fill(name);
  await dialog.getByRole("slider").fill("3");
  await dialog.getByRole("button", { name: "Launch", exact: true }).click();
  await expect(dialog).toBeHidden();
}

test.describe.configure({ mode: "serial" });

test("Fair mode off: each AI trains on its own player's helper; sharing, a battle and chat work; nothing private leaks", async ({
  browser,
  friends,
}) => {
  test.setTimeout(30 * 60_000);
  const { url } = friends as FriendsStack;
  const a = await (await browser.newContext()).newPage();
  const b = await (await browser.newContext()).newPage();
  const bSaw = recordTraffic(b);

  // 1. Ana makes the world; Ben joins by its link. Both found companies; Ana opens it.
  await signUp(a, url, "Ana");
  await createWorld(a, "Friends", false);
  await found(a, "Acme");
  const code = ((await a.getByTestId("invite-code").textContent()) ?? "").trim();
  expect(code).toMatch(/^[A-Z0-9]+$/);
  await joinByLink(b, url, code, "Ben", false);
  await found(b, "Bolt");
  await a.getByRole("button", { name: "Open for business" }).click();
  await expect(b.getByRole("link", { name: "AI Lab" })).toBeVisible();

  // Ana's private notebook, which Ben must never receive.
  await a.getByRole("link", { name: "AI Lab" }).click();
  await a.getByRole("button", { name: "1 Data" }).click();
  await a.getByTestId("dataset-notebook").getByRole("button", { name: "Write" }).click();
  await a.getByLabel("Notebook text").fill(NOTEBOOK);
  await a.getByRole("button", { name: "Save" }).click();
  await expect(a.getByText("Saved. Only your company's AI reads this.")).toBeVisible();
  await a.keyboard.press("Escape");

  // 2. Ben connects his own helper; then both train at once, each on their own card.
  const anaKey = await helperKey(a, url);
  const bens = await friends.startHelper(await helperKey(b, url), "ben-pc");
  await expect(b.getByRole("button", { name: /^Your computer: .* online$/ })).toBeVisible();
  const houseTrains = friends.house.count(/\(train\) started/);
  await startTraining(a);
  await startTraining(b);
  await friends.house.log(/\(train\) started/, houseTrains + 1);
  await bens.log(/\(train\) started/, 1);
  await trained(a);
  await trained(b);

  // 3. Ana shares v1; Ben tries it and gets a reply from Ana's card.
  await tryShared(a, b);

  // 4. A battle between their AIs, once both are scored and on sale.
  await scoredAndLaunched(a, "Spark");
  await scoredAndLaunched(b, "Nova");
  await b.getByRole("link", { name: "Arena" }).click();
  await b.getByRole("button", { name: "Start a battle" }).click();
  const start = b.getByRole("dialog", { name: "Start a battle" });
  await expect(
    start.getByLabel("Against").locator("option", { hasText: "Acme's Spark" }),
  ).toHaveCount(1, { timeout: 120_000 });
  await start.getByLabel("Against").selectOption({ label: "Acme's Spark" });
  await start.getByRole("button", { name: "Start the battle" }).click();
  await expect(start).toHaveCount(0);
  await b.getByRole("button", { name: /^Open the knowledge battle/ }).click();
  await expect(b.getByRole("dialog").getByText(/^(Win|Loss|Draw) \d–\d$/)).toBeVisible({
    timeout: 300_000,
  });
  await b.keyboard.press("Escape");
  await a.getByRole("link", { name: "Arena" }).click();
  await expect(a.getByText(/Bolt challenged you/)).toBeVisible({ timeout: 60_000 });

  // 4b. Licensing between friends: Ana offers Spark; Ben licenses it for Nova's students; at
  // the month's close Ana's money shows License income.
  await a.getByRole("link", { name: "Company" }).click();
  await a.getByRole("button", { name: /^About your AI / }).click();
  await a.getByRole("button", { name: "Offer for license" }).click();
  const offer = a.getByRole("dialog", { name: /for license$/ });
  await offer.getByLabel("Price per customer a month ($)").fill("2");
  await offer.getByLabel("Students").check();
  await offer.getByRole("button", { name: "Offer it" }).click();
  await expect(offer).toHaveCount(0);
  await a.keyboard.press("Escape");
  await b.getByRole("link", { name: "AI Directory" }).click();
  await b
    .getByRole("region", { name: "License offers" })
    .getByRole("button", { name: "License Spark" })
    .click({ timeout: 60_000 });
  const take = b.getByRole("dialog", { name: "License Spark" });
  await take.getByRole("button", { name: "License it" }).click();
  await expect(take).toHaveCount(0);
  await a.getByRole("link", { name: "Money" }).click();
  await expect(a.getByTestId("income-licenseIncome")).toBeVisible({ timeout: 180_000 });
  // Back where the later steps expect Ana (the Money screen has a second "Open money").
  await a.getByRole("link", { name: "Arena" }).click();

  // 5. The world chat, both ways.
  await a.getByRole("button", { name: /World chat/ }).click();
  await a.getByLabel("Message").fill("Good game, Ben!");
  await a.getByLabel("Message").press("Enter");
  await b.getByRole("button", { name: /World chat/ }).click();
  await expect(b.getByText("Good game, Ben!")).toBeVisible();
  await b.getByLabel("Message").fill("Thanks, Ana!");
  await b.getByLabel("Message").press("Enter");
  await expect(a.getByText("Thanks, Ana!")).toBeVisible();

  // 6. Opacity: nothing Ben's browser received holds Ana's notebook, her helper key or a
  // signed link to anyone's AI.
  const everything = bSaw.join("\n");
  expect(everything).not.toContain(NOTEBOOK);
  expect(everything).not.toContain(anaKey.split(".")[1] ?? anaKey);
  expect(everything).not.toMatch(/\/blobs\/[^"\s]*\?/);

  // 7. Ben's helper stops: with Fair mode off he can't train until it is back.
  await bens.stop();
  await b.getByRole("link", { name: "AI Lab" }).click();
  await expect(b.getByText("Connect your computer to do this.").first()).toBeVisible({
    timeout: 60_000,
  });
  await expect(b.getByRole("button", { name: "Start training" })).toBeDisabled();

  // 8. A server restart: both pages come back; the day, the versions and the cash stay.
  const cash = async () =>
    /Cash\$([\d,]+)/.exec(
      (await a.getByRole("button", { name: "Open money" }).textContent()) ?? "",
    )?.[1];
  await a.getByRole("button", { name: "Pause" }).click();
  // Paused: the cash stops moving (a count-up may still be finishing).
  let settled = await cash();
  await expect
    .poll(async () => {
      const now = await cash();
      const same = now === settled;
      settled = now;
      return same;
    })
    .toBe(true);
  const dayBefore = await a
    .getByText(/^Day \d+$/)
    .first()
    .textContent();
  await friends.restartServer();
  await expect(a.getByRole("button", { name: "Resume" })).toBeVisible({ timeout: 60_000 });
  await expect.poll(cash).toBe(settled);
  await expect(a.getByText(/^Day \d+$/).first()).toHaveText(dayBefore ?? "");
  await a.getByRole("link", { name: "AI Lab" }).click();
  await a.getByRole("button", { name: "5 Versions" }).click();
  await expect(a.getByTestId("version-v1")).toHaveCount(1);
  await expect(b.getByRole("link", { name: "AI Lab" })).toBeVisible();
});

test("Fair mode on: the cards are shared, runs land on either card, and a friend's AI answers", async ({
  browser,
  friends,
}) => {
  test.setTimeout(20 * 60_000);
  const { url } = friends as FriendsStack;
  const c = await (await browser.newContext()).newPage();
  const d = await (await browser.newContext()).newPage();

  await signUp(c, url, "Cara");
  await createWorld(c, "Fair friends", true);
  await found(c, "Cove");
  const code = ((await c.getByTestId("invite-code").textContent()) ?? "").trim();
  await joinByLink(d, url, code, "Dan", true);
  await found(d, "Dune");
  await c.getByRole("button", { name: "Open for business" }).click();

  const dans = await friends.startHelper(await helperKey(d, url), "dan-pc");
  const before = friends.house.count(/\(train\) started/) + dans.count(/\(train\) started/);
  await startTraining(c);
  await startTraining(d);
  await trained(c);
  await trained(d);
  // Both runs went to the shared cards, whichever was free.
  expect(friends.house.count(/\(train\) started/) + dans.count(/\(train\) started/)).toBe(
    before + 2,
  );

  await tryShared(c, d);
});
