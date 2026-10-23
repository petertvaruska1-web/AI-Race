import { afterEach, describe, expect, it } from "vitest";
import { verifyWsToken } from "../src/auth.ts";
import { schema } from "../src/db/index.ts";
import { startTestServer, type TestServer } from "./harness.ts";

let server: TestServer | null = null;
afterEach(async () => {
  await server?.close();
  server = null;
});

async function guest(s: TestServer, displayName: unknown): Promise<Response> {
  return fetch(`${s.url}/api/guest`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ displayName }),
  });
}

function sessionCookie(res: Response): string {
  const header = res.headers.get("set-cookie") ?? "";
  const match = /airace_session=([^;]+)/.exec(header);
  if (!match?.[1]) throw new Error(`no session cookie in ${header}`);
  return match[1];
}

async function me(s: TestServer, token?: string): Promise<Response> {
  return fetch(`${s.url}/api/me`, token ? { headers: { cookie: `airace_session=${token}` } } : {});
}

describe("guest accounts", () => {
  it("creates a guest with a session cookie that /api/me recognizes", async () => {
    server = await startTestServer();
    const res = await guest(server, "  Ana ");
    expect(res.status).toBe(200);
    const body = (await res.json()) as { userId: string; displayName: string };
    expect(body.displayName).toBe("Ana");
    const header = res.headers.get("set-cookie") ?? "";
    expect(header).toMatch(/HttpOnly/i);
    expect(header).toMatch(/SameSite=Lax/i);
    expect(header).not.toMatch(/Secure/i);

    const who = await me(server, sessionCookie(res));
    expect(who.status).toBe(200);
    const meBody = (await who.json()) as { userId: string; displayName: string; wsToken: string };
    expect(meBody).toMatchObject({ userId: body.userId, displayName: "Ana" });
    expect(verifyWsToken(server.ctx, meBody.wsToken)).toBe(body.userId);
  });

  it("stores only a hash of the session token", async () => {
    server = await startTestServer();
    const token = sessionCookie(await guest(server, "Ana"));
    const rows = [
      ...(await server.ctx.db.select().from(schema.sessions)),
      ...(await server.ctx.db.select().from(schema.authIdentities)),
      ...(await server.ctx.db.select().from(schema.users)),
    ];
    expect(rows.length).toBe(3);
    expect(JSON.stringify(rows)).not.toContain(token);
  });

  it("refuses display names that are too long or blank", async () => {
    server = await startTestServer();
    expect((await guest(server, "x".repeat(25))).status).toBe(400);
    expect((await guest(server, "   ")).status).toBe(400);
    expect((await guest(server, 42)).status).toBe(400);
    expect((await guest(server, "x".repeat(24))).status).toBe(200);
  });

  it("answers 401 without a valid session", async () => {
    server = await startTestServer();
    expect((await me(server)).status).toBe(401);
    expect((await me(server, "not-a-real-token")).status).toBe(401);
  });

  it("marks the cookie Secure when players reach the server over https", async () => {
    server = await startTestServer({ publicUrl: "https://race.example" });
    expect((await guest(server, "Ana")).headers.get("set-cookie")).toMatch(/Secure/i);
  });
});

describe("ws tokens", () => {
  it("rejects tampered and expired tokens", async () => {
    server = await startTestServer();
    const token = sessionCookie(await guest(server, "Ana"));
    const { wsToken, userId } = (await (await me(server, token)).json()) as {
      wsToken: string;
      userId: string;
    };
    expect(verifyWsToken(server.ctx, wsToken)).toBe(userId);

    const decoded = Buffer.from(wsToken, "base64url").toString();
    const [, exp, mac] = decoded.split(".");
    const forged = Buffer.from(`00000000-0000-4000-8000-000000000000.${exp}.${mac}`).toString(
      "base64url",
    );
    expect(verifyWsToken(server.ctx, forged)).toBeNull();
    expect(verifyWsToken(server.ctx, `${wsToken}x`)).toBeNull();
    expect(verifyWsToken(server.ctx, "")).toBeNull();

    await server.scheduler.advance(11 * 60 * 1000);
    expect(verifyWsToken(server.ctx, wsToken)).toBeNull();
  });
});
