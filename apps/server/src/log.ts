import { type Logger, pino } from "pino";

export type { Logger } from "pino";

/** Context fields a log line carries when known. */
export type LogBindings = {
  worldId?: string;
  userId?: string;
  cid?: string;
  jobId?: string;
  runId?: string;
};

export function createLogger(level = "info"): Logger {
  return pino({ level, base: { svc: "airace-server" } });
}

export function withContext(log: Logger, bindings: LogBindings): Logger {
  return log.child(bindings);
}
