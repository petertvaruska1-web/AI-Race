import asyncio
import io
import tarfile
import threading
import uuid
from pathlib import Path

import pytest
from websockets.asyncio.server import serve

from airace_ml.data.identity import data_identity
from airace_ml.worker.agent import JobContext, WorkerAgent
from airace_ml.worker.blobs import UnsafeTarError, download_tar, upload_dir_tar
from airace_ml.worker.protocol import (
    Done,
    Failed,
    Heartbeat,
    Lease,
    Progress,
    WHello,
    WWelcome,
    dumps,
    parse_worker_message,
)

TIMEOUT = 5


def echo(spec: dict, ctx: JobContext) -> dict:
    for i in range(3):
        ctx.progress({"step": i})
    return {"echo": spec}


class FakeServer:
    """An in-process `/worker` endpoint that records what the agent sends."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sockets: list = []

    async def handler(self, ws) -> None:
        self.sockets.append(ws)
        async for raw in ws:
            await self.inbox.put(parse_worker_message(raw))

    async def next(self, *types, skip_heartbeats: bool = True):
        while True:
            msg = await asyncio.wait_for(self.inbox.get(), TIMEOUT)
            if skip_heartbeats and isinstance(msg, Heartbeat) and Heartbeat not in types:
                continue
            assert isinstance(msg, types), msg
            return msg

    @property
    def ws(self):
        return self.sockets[-1]


async def _with_agent(data_root: Path, handlers: dict, scenario, tmp_path: Path) -> None:
    server = FakeServer()
    async with serve(server.handler, "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        agent = WorkerAgent(
            f"ws://127.0.0.1:{port}",
            "test-token",
            data_root,
            handlers=handlers,
            worker_id="test-worker",
            work_root=tmp_path / "work",
            heartbeat_s=0.05,
            backoff_s=0.05,
        )
        running = asyncio.create_task(agent.run_forever())
        try:
            await scenario(server, agent)
        finally:
            await agent.stop()
            await asyncio.wait_for(running, TIMEOUT)


def test_says_hello_with_its_data_then_heartbeats_and_runs_a_job(tiny_data_root, tmp_path):
    job = str(uuid.uuid4())

    async def scenario(server, agent):
        hello = await server.next(WHello)
        assert hello.token == "test-token"
        assert hello.worker_id == "test-worker"
        assert hello.capabilities.data_identity == data_identity(tiny_data_root)
        assert hello.capabilities.job_types == ("echo",)
        assert hello.running == ()
        await server.ws.send(dumps(WWelcome()))
        await server.next(Heartbeat)
        await server.ws.send(dumps(Lease(job, "echo", {"x": 1}, 30_000)))
        steps = [(await server.next(Progress)).event["step"] for _ in range(3)]
        assert steps == [0, 1, 2]
        done = await server.next(Done)
        assert done == Done(job, {"echo": {"x": 1}})

    asyncio.run(_with_agent(tiny_data_root, {"echo": echo}, scenario, tmp_path))


def test_reconnects_and_names_the_job_it_is_still_running(tiny_data_root, tmp_path):
    job = str(uuid.uuid4())
    release = threading.Event()

    def slow(spec, ctx):
        assert release.wait(TIMEOUT)
        return {"ok": True}

    async def scenario(server, agent):
        await server.next(WHello)
        await server.ws.send(dumps(WWelcome()))
        await server.ws.send(dumps(Lease(job, "slow", {}, 30_000)))
        await asyncio.sleep(0.1)
        await server.ws.close()
        again = await server.next(WHello)
        assert again.running == (job,)
        await server.ws.send(dumps(WWelcome()))
        release.set()
        assert await server.next(Done) == Done(job, {"ok": True})

    asyncio.run(_with_agent(tiny_data_root, {"slow": slow}, scenario, tmp_path))


def test_reports_failures_and_jobs_it_cannot_run(tiny_data_root, tmp_path):
    bad, unknown, broken = (str(uuid.uuid4()) for _ in range(3))

    def strict(spec, ctx):
        if spec.get("io"):
            raise OSError("disk full")
        raise ValueError("a brain can't shrink")

    async def scenario(server, agent):
        await server.next(WHello)
        await server.ws.send(dumps(WWelcome()))
        await server.ws.send(dumps(Lease(bad, "strict", {}, 30_000)))
        assert await server.next(Failed) == Failed(bad, "a brain can't shrink", False)
        await server.ws.send(dumps(Lease(broken, "strict", {"io": True}, 30_000)))
        failed = await server.next(Failed)
        assert failed.job_id == broken and failed.retryable
        await server.ws.send(dumps(Lease(unknown, "train", {}, 30_000)))
        failed = await server.next(Failed)
        assert failed.job_id == unknown and not failed.retryable

    asyncio.run(_with_agent(tiny_data_root, {"strict": strict}, scenario, tmp_path))


def test_stops_when_the_server_refuses_its_token(tiny_data_root, tmp_path):
    async def main():
        async def refuse(ws):
            await ws.recv()
            await ws.close(4401, "unknown worker")

        async with serve(refuse, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            agent = WorkerAgent(
                f"ws://127.0.0.1:{port}",
                "wrong",
                tiny_data_root,
                handlers={"echo": echo},
                backoff_s=0.05,
            )
            await asyncio.wait_for(agent.run_forever(), TIMEOUT)

    asyncio.run(main())


# -- blobs ---------------------------------------------------------------------------------------


def test_uploads_and_downloads_a_model_folder(blob_server, tmp_path):
    url, store, _ = blob_server
    src = tmp_path / "model"
    (src / "sub").mkdir(parents=True)
    (src / "weights.bin").write_bytes(b"\x00\x01" * 1000)
    (src / "sub" / "meta.json").write_text('{"a": 1}')
    (src / "resume").mkdir()
    (src / "resume" / "state.pt").write_bytes(b"resume")
    upload_dir_tar(src, f"{url}/blobs/v1.tar?sig=x", exclude=("resume",))
    assert "/blobs/v1.tar" in store
    out = download_tar(f"{url}/blobs/v1.tar?sig=x", tmp_path / "copy")
    assert (out / "weights.bin").read_bytes() == b"\x00\x01" * 1000
    assert (out / "sub" / "meta.json").read_text() == '{"a": 1}'
    assert not (out / "resume").exists()


def test_refuses_a_tar_that_escapes_its_folder(blob_server, tmp_path):
    url, store, _ = blob_server
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("../evil")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"evil"))
    store["/blobs/evil.tar"] = buf.getvalue()
    with pytest.raises(UnsafeTarError):
        download_tar(f"{url}/blobs/evil.tar", tmp_path / "dest")
    assert not (tmp_path / "evil").exists()
