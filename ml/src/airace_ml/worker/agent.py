"""The dial-out worker agent (plan Task 17, spec §5.4).

A worker connects to the server's ``/worker`` endpoint (so it works from behind a home router),
says ``whello`` with what it can run and which data it holds, sends a heartbeat every few
seconds, and runs each leased job in a thread. Progress and chat text stream back as they come;
a job's result is kept until the server has it, across reconnects: ``whello.running`` names every
job whose result the server has not received yet, so the server keeps those leases.
"""

import asyncio
import logging
import re
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

from airace_ml.data.identity import data_identity
from airace_ml.worker.protocol import (
    JOB_TYPES,
    Cancel,
    Capabilities,
    Chunk,
    Done,
    Failed,
    Heartbeat,
    Lease,
    Progress,
    ProtocolError,
    WHello,
    WorkerMsg,
    WWelcome,
    dumps,
    parse_server_message,
)

log = logging.getLogger("airace.worker")

HEARTBEAT_S = 5.0
BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 30.0
MAX_PARAMS = 85_000_000
CLOSE_UNAUTHORIZED = 4401
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class JobError(Exception):
    """A job that cannot finish. ``retryable`` tells the server whether to try it again."""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class JobContext:
    """What a handler gets besides its spec: a place to work and ways to report."""

    job_id: str
    job_type: str
    work_dir: Path
    _emit: Callable[[WorkerMsg], None]
    _cancelled: threading.Event = field(default_factory=threading.Event)

    def progress(self, event: Any) -> None:
        if not self.cancelled:
            self._emit(Progress(self.job_id, event))

    def chunk(self, text: str) -> None:
        if not self.cancelled and text:
            self._emit(Chunk(self.job_id, text))

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()


Handler = Callable[[dict[str, Any], JobContext], Any]


def _default_worker_id(device: str) -> str:
    host = re.sub(r"[^A-Za-z0-9._-]", "-", socket.gethostname())[:48] or "worker"
    return f"{host}-{device}"


def _vram_mb(device: str) -> int:
    if device != "cuda":
        return 0
    import torch

    return int(torch.cuda.get_device_properties(0).total_memory // 2**20)


class WorkerAgent:
    def __init__(
        self,
        server_url: str,
        token: str,
        data_root: Path,
        *,
        device: str = "cpu",
        handlers: dict[str, Handler],
        slots: dict[str, int] | None = None,
        worker_id: str | None = None,
        work_root: Path | None = None,
        heartbeat_s: float = HEARTBEAT_S,
        backoff_s: float = BACKOFF_START_S,
    ) -> None:
        if not handlers:
            raise ValueError("a worker needs at least one job handler")
        self.server_url = server_url
        self.token = token
        self.device = device
        self.handlers = dict(handlers)
        self.slots = {t: 1 for t in self.handlers} | (slots or {})
        self.worker_id = worker_id or _default_worker_id(device)
        self.work_root = work_root or Path(data_root) / "worker" / "work"
        self.heartbeat_s = heartbeat_s
        self.backoff_s = backoff_s
        vram = _vram_mb(device)
        self.capabilities = Capabilities(
            device="cuda" if device == "cuda" else "cpu",
            vram_mb=vram,
            # A GPU holds about 10k parameters' training state per MB; the CPU is bounded by time.
            max_params=MAX_PARAMS if device != "cuda" else min(MAX_PARAMS, vram * 10_000),
            # The game's job types in their usual order, then any others (tests register some).
            job_types=tuple(
                [t for t in JOB_TYPES if t in self.handlers]
                + sorted(t for t in self.handlers if t not in JOB_TYPES)
            ),
            slots=self.slots,
            data_identity=data_identity(Path(data_root)),
        )
        self._jobs: dict[str, JobContext] = {}
        self._unreported: set[str] = set()
        self._ws: ClientConnection | None = None
        self._stopping = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[WorkerMsg] | None = None
        self._connected: asyncio.Event | None = None
        self._tasks: set[asyncio.Task] = set()

    # -- lifecycle -------------------------------------------------------------------------

    async def run_forever(self) -> None:
        """Connects, serves leases and reconnects (1, 2, 4 ... 30 s apart) until :meth:`stop`."""
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._connected = asyncio.Event()
        sender = asyncio.create_task(self._sender())
        delay = self.backoff_s
        try:
            while not self._stopping:
                code = await self._session()
                if code == CLOSE_UNAUTHORIZED:
                    log.error("the server refused this worker's token; stopping")
                    return
                if code == 0:
                    delay = self.backoff_s  # a session that got going resets the backoff
                if self._stopping:
                    return
                log.info("reconnecting in %.1f s", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, BACKOFF_MAX_S)
        finally:
            sender.cancel()

    async def stop(self) -> None:
        self._stopping = True
        ws = self._ws
        if ws is not None:
            await ws.close()

    async def _session(self) -> int | None:
        """One connection. Returns 0 after a welcomed session, the close code if refused, else None."""
        assert self._connected is not None
        try:
            async with connect(self.server_url, max_size=MAX_MESSAGE_BYTES, open_timeout=10) as ws:
                hello = WHello(
                    self.token, self.worker_id, self.capabilities, tuple(sorted(self._unreported))
                )
                await ws.send(dumps(hello))
                try:
                    first = parse_server_message(await ws.recv())
                except ConnectionClosed:
                    return ws.close_code
                if not isinstance(first, WWelcome):
                    raise ProtocolError(f"expected wwelcome, got {first!r}")
                log.info("connected to %s as %s", self.server_url, self.worker_id)
                self._ws = ws
                self._connected.set()
                beat = asyncio.create_task(self._heartbeat(ws))
                try:
                    async for raw in ws:
                        try:
                            self._on_message(parse_server_message(raw))
                        except ProtocolError as e:
                            log.warning("ignored a message from the server: %s", e)
                except ConnectionClosed:
                    pass
                finally:
                    beat.cancel()
                    self._ws = None
                    self._connected.clear()
                return CLOSE_UNAUTHORIZED if ws.close_code == CLOSE_UNAUTHORIZED else 0
        except (OSError, InvalidHandshake, InvalidURI, ProtocolError, TimeoutError) as e:
            log.warning("could not reach the server: %s", e)
            return None

    async def _heartbeat(self, ws: ClientConnection) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_s)
            try:
                await ws.send(dumps(Heartbeat()))
            except ConnectionClosed:
                return

    async def _sender(self) -> None:
        """Sends queued messages in order. Results wait for a connection; progress is dropped offline."""
        assert self._queue is not None and self._connected is not None
        while True:
            msg = await self._queue.get()
            final = isinstance(msg, (Done, Failed))
            while True:
                ws = self._ws
                if ws is None:
                    if not final:
                        break
                    await self._connected.wait()
                    continue
                try:
                    await ws.send(dumps(msg))
                except ConnectionClosed:
                    if not final:
                        break
                    await asyncio.sleep(0.05)
                    continue
                if final:
                    self._unreported.discard(msg.job_id)
                break

    # -- jobs ----------------------------------------------------------------------------------

    def _emit(self, msg: WorkerMsg) -> None:
        """Queues a message from any thread."""
        assert self._loop is not None and self._queue is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)

    def _on_message(self, msg: Lease | Cancel | WWelcome) -> None:
        if isinstance(msg, Lease):
            self._start(msg)
        elif isinstance(msg, Cancel):
            job = self._jobs.get(msg.job_id)
            if job is not None:
                job._cancelled.set()
            self._unreported.discard(msg.job_id)

    def _start(self, lease: Lease) -> None:
        if lease.job_id in self._jobs or lease.job_id in self._unreported:
            return  # already running, or finished and waiting to be reported
        handler = self.handlers.get(lease.job_type)
        if handler is None:
            self._emit(Failed(lease.job_id, f"this worker can't run {lease.job_type} jobs", False))
            return
        ctx = JobContext(lease.job_id, lease.job_type, self.work_root / lease.job_id, self._emit)
        self._jobs[lease.job_id] = ctx
        self._unreported.add(lease.job_id)
        task = asyncio.create_task(asyncio.to_thread(self._run, handler, lease, ctx))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _run(self, handler: Handler, lease: Lease, ctx: JobContext) -> None:
        """Runs in a worker thread."""
        log.info("job %s (%s) started", lease.job_id, lease.job_type)
        try:
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
            msg: WorkerMsg = Done(lease.job_id, handler(lease.spec, ctx))
        except JobError as e:
            msg = Failed(lease.job_id, str(e), e.retryable)
        except ValueError as e:  # a spec or config that can't run: trying again won't help
            msg = Failed(lease.job_id, str(e), False)
        except OSError as e:  # disk or network trouble: worth another try
            msg = Failed(lease.job_id, f"{type(e).__name__}: {e}", True)
        except Exception as e:
            log.exception("job %s crashed", lease.job_id)
            msg = Failed(lease.job_id, f"{type(e).__name__}: {e}", False)
        if isinstance(msg, Failed):
            log.warning("job %s failed: %s", lease.job_id, msg.error)
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._finish, ctx, msg)

    def _finish(self, ctx: JobContext, msg: WorkerMsg) -> None:
        self._jobs.pop(ctx.job_id, None)
        if ctx.cancelled:
            self._unreported.discard(ctx.job_id)
            return
        assert self._queue is not None
        self._queue.put_nowait(msg)
        log.info("job %s finished", ctx.job_id)
