import { createHash } from "node:crypto";
import type { Query } from "@airace/protocol";
import { eq } from "drizzle-orm";
import type { AppContext } from "../app.ts";
import { HttpError } from "../auth.ts";
import { schema } from "../db/index.ts";
import type { NotebookSaveMessage } from "../realtime/gateway.ts";
import { notebookProblem, wordCount } from "./filter.ts";

type Result = { ok: true; data: { words: number } } | { ok: false; reason: string };

/**
 * The player's notebook (game-layer design §10): private text that joins their training as a
 * ninth share. The server keeps the text, stored by its sha256 so a run trains on exactly the
 * text saved when it started; the world gets only the fingerprint and the word count (R2).
 * Nothing here logs the text.
 */
export class NotebookService {
  constructor(private readonly ctx: AppContext) {
    ctx.gateway.setNotebookHandler((userId, msg) => this.save(userId, msg));
    ctx.gateway.registerQuery("notebook", (userId, q) => this.read(userId, q));
  }

  private async companyOf(userId: string, worldId: string): Promise<string | null> {
    try {
      const actor = await this.ctx.worlds.get(worldId);
      return actor.state().members[userId]?.companyId ?? null;
    } catch {
      return null;
    }
  }

  private async save(userId: string, msg: NotebookSaveMessage): Promise<Result> {
    const companyId = await this.companyOf(userId, msg.worldId);
    if (!companyId) return { ok: false, reason: "Found your company first." };
    const problem = notebookProblem(msg.text);
    if (problem) return { ok: false, reason: problem };
    const hash = createHash("sha256").update(msg.text).digest("hex");
    const words = wordCount(msg.text);
    const { db } = this.ctx;
    await db.insert(schema.notebookTexts).values({ hash, text: msg.text }).onConflictDoNothing();
    await db
      .insert(schema.notebooks)
      .values({ companyId, worldId: msg.worldId, hash })
      .onConflictDoUpdate({ target: schema.notebooks.companyId, set: { hash } });
    const actor = await this.ctx.worlds.get(msg.worldId);
    const r = await actor.submit({
      kind: "system",
      event: { type: "notebookSaved", companyId, hash, words },
    });
    if (!r.ok) return { ok: false, reason: r.reason };
    return { ok: true, data: { words } };
  }

  private async read(userId: string, q: Query): Promise<unknown> {
    if (q.kind !== "notebook") throw new HttpError(400, "Unknown request.");
    const companyId = await this.companyOf(userId, q.worldId);
    if (!companyId) throw new HttpError(403, "That notebook isn't yours.");
    const [row] = await this.ctx.db
      .select({ text: schema.notebookTexts.text })
      .from(schema.notebooks)
      .innerJoin(schema.notebookTexts, eq(schema.notebookTexts.hash, schema.notebooks.hash))
      .where(eq(schema.notebooks.companyId, companyId));
    return { text: row?.text ?? "" };
  }

  /** The text a run trains on, by the fingerprint it recorded (null when it is gone). */
  async textOf(hash: string): Promise<string | null> {
    const [row] = await this.ctx.db
      .select({ text: schema.notebookTexts.text })
      .from(schema.notebookTexts)
      .where(eq(schema.notebookTexts.hash, hash));
    return row?.text ?? null;
  }
}
