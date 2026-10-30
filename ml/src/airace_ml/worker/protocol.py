"""Messages between a worker and the game server (``packages/protocol/src/worker.ts``).

The TypeScript zod schemas are the authority; ``packages/protocol/fixtures/worker/`` holds one
example of every message, and both languages' tests read the same files. Envelope keys are
camelCase on the wire; Python-made payloads (training config and results) are snake_case.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal

JobType = Literal["train", "eval", "generate"]
JOB_TYPES: tuple[str, ...] = ("train", "eval", "generate", "feedback", "battle")
VERSION = 1


class ProtocolError(ValueError):
    """A message that is not one this protocol defines."""


# -- server to worker -----------------------------------------------------------------------


@dataclass(frozen=True)
class WWelcome:
    pass


@dataclass(frozen=True)
class Lease:
    job_id: str
    job_type: str
    spec: dict[str, Any]
    lease_ms: int


@dataclass(frozen=True)
class Cancel:
    job_id: str


ServerMsg = WWelcome | Lease | Cancel


# -- worker to server -----------------------------------------------------------------------


@dataclass(frozen=True)
class Capabilities:
    device: Literal["cuda", "cpu"]
    vram_mb: int
    max_params: int
    job_types: tuple[str, ...]
    slots: dict[str, int]
    data_identity: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "vramMb": self.vram_mb,
            "maxParams": self.max_params,
            "jobTypes": list(self.job_types),
            "slots": {t: int(self.slots.get(t, 0)) for t in JOB_TYPES},
            "dataIdentity": self.data_identity,
        }


@dataclass(frozen=True)
class WHello:
    token: str
    worker_id: str
    capabilities: Capabilities
    running: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Heartbeat:
    pass


@dataclass(frozen=True)
class Progress:
    job_id: str
    event: Any


@dataclass(frozen=True)
class Chunk:
    job_id: str
    text: str


@dataclass(frozen=True)
class Done:
    job_id: str
    result: Any


@dataclass(frozen=True)
class Failed:
    job_id: str
    error: str
    retryable: bool


WorkerMsg = WHello | Heartbeat | Progress | Chunk | Done | Failed


# -- encoding ---------------------------------------------------------------------------------


def dumps(msg: ServerMsg | WorkerMsg) -> str:
    """A message as the JSON the other side expects (``v: 1``, camelCase envelope)."""
    match msg:
        case WHello():
            body = {
                "type": "whello",
                "token": msg.token,
                "workerId": msg.worker_id,
                "capabilities": msg.capabilities.to_wire(),
                "running": list(msg.running),
            }
        case Heartbeat():
            body = {"type": "heartbeat"}
        case Progress():
            body = {"type": "progress", "jobId": msg.job_id, "event": msg.event}
        case Chunk():
            body = {"type": "chunk", "jobId": msg.job_id, "text": msg.text}
        case Done():
            body = {"type": "done", "jobId": msg.job_id, "result": msg.result}
        case Failed():
            body = {
                "type": "failed",
                "jobId": msg.job_id,
                "error": msg.error,
                "retryable": msg.retryable,
            }
        case WWelcome():
            body = {"type": "wwelcome"}
        case Lease():
            body = {
                "type": "lease",
                "jobId": msg.job_id,
                "jobType": msg.job_type,
                "spec": msg.spec,
                "leaseMs": msg.lease_ms,
            }
        case Cancel():
            body = {"type": "cancel", "jobId": msg.job_id}
        case _:
            raise ProtocolError(f"not a protocol message: {msg!r}")
    return json.dumps({"v": VERSION, **body}, ensure_ascii=False)


def _load(raw: str | bytes) -> dict[str, Any]:
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProtocolError(f"not JSON: {e}") from None
    if not isinstance(d, dict):
        raise ProtocolError("a message must be a JSON object")
    if d.get("v") != VERSION:
        raise ProtocolError(f"unsupported protocol version {d.get('v')!r}")
    return d


def _str(d: dict[str, Any], key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v:
        raise ProtocolError(f"{d.get('type')}: {key} must be a non-empty string")
    return v


def parse_server_message(raw: str | bytes) -> ServerMsg:
    d = _load(raw)
    match d.get("type"):
        case "wwelcome":
            return WWelcome()
        case "lease":
            spec, lease_ms = d.get("spec"), d.get("leaseMs")
            if not isinstance(spec, dict):
                raise ProtocolError("lease: spec must be an object")
            if not isinstance(lease_ms, int) or isinstance(lease_ms, bool) or lease_ms <= 0:
                raise ProtocolError("lease: leaseMs must be a positive integer")
            return Lease(_str(d, "jobId"), _str(d, "jobType"), spec, lease_ms)
        case "cancel":
            return Cancel(_str(d, "jobId"))
        case other:
            raise ProtocolError(f"unknown server message type {other!r}")


def parse_worker_message(raw: str | bytes) -> WorkerMsg:
    """The inverse of :func:`dumps` for worker messages (tests and fake servers use it)."""
    d = _load(raw)
    match d.get("type"):
        case "whello":
            c = d.get("capabilities")
            if not isinstance(c, dict):
                raise ProtocolError("whello: capabilities must be an object")
            caps = Capabilities(
                device=c["device"],
                vram_mb=int(c["vramMb"]),
                max_params=int(c["maxParams"]),
                job_types=tuple(c["jobTypes"]),
                slots={k: int(v) for k, v in c["slots"].items()},
                data_identity=c["dataIdentity"],
            )
            return WHello(_str(d, "token"), _str(d, "workerId"), caps, tuple(d.get("running", [])))
        case "heartbeat":
            return Heartbeat()
        case "progress":
            return Progress(_str(d, "jobId"), d.get("event"))
        case "chunk":
            text = d.get("text")
            if not isinstance(text, str):
                raise ProtocolError("chunk: text must be a string")
            return Chunk(_str(d, "jobId"), text)
        case "done":
            return Done(_str(d, "jobId"), d.get("result"))
        case "failed":
            return Failed(_str(d, "jobId"), str(d.get("error", "")), bool(d.get("retryable")))
        case other:
            raise ProtocolError(f"unknown worker message type {other!r}")
