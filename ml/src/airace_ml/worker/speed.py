"""What a helper tells the game about its computer (Stage 2): the graphics card's name, a short
speed test, and the game link it was started from.

The speed is training tokens a second on a fixed reference model (2 layers of 64, context 64,
batches of 8,192 tokens of random ids), so every card is timed on the same work. It is measured
once per device and remembered, so a restart doesn't repeat the test.
"""

import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from airace_ml.model.shape import ModelShape
from airace_ml.model.transformer import Transformer
from airace_ml.tokenizer import VOCAB_SIZE

REFERENCE_SHAPE = ModelShape(n_layer=2, d_model=64, ctx_len=64)
REFERENCE_BATCH_TOKENS = 8192
SPEED_FILE = "speed.json"


def gpu_name(device: torch.device) -> str:
    """The graphics card's name, or "CPU" when the helper runs on the processor."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU"


def _step(model: Transformer, opt: torch.optim.Optimizer, device: torch.device) -> int:
    batch = REFERENCE_BATCH_TOKENS // REFERENCE_SHAPE.ctx_len
    ids = torch.randint(0, VOCAB_SIZE, (batch, REFERENCE_SHAPE.ctx_len + 1), device=device)
    logits = model(ids[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), ids[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return batch * REFERENCE_SHAPE.ctx_len


def measure_speed(
    device: torch.device, seconds: float = 5.0, cache_dir: Path | None = None
) -> float:
    """Training tokens a second on the reference model, measured for about ``seconds`` (one
    warm-up step is not counted), and remembered in ``cache_dir/speed.json`` by card name."""
    name = gpu_name(device)
    cache = Path(cache_dir) / SPEED_FILE if cache_dir is not None else None
    known: dict[str, float] = {}
    if cache is not None and cache.is_file():
        try:
            known = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            known = {}
        if isinstance(known.get(name), (int, float)) and known[name] > 0:
            return float(known[name])
    torch.manual_seed(0)
    model = Transformer(REFERENCE_SHAPE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _step(model, opt, device)  # warm-up: kernels, allocations
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tokens = 0
    start = time.perf_counter()
    while True:
        tokens += _step(model, opt, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            break
    speed = round(tokens / elapsed, 1)
    if cache is not None:
        known[name] = speed
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=2), encoding="utf-8")
    return speed


def parse_connect_link(link: str) -> tuple[str, str | None]:
    """The worker address and helper key in a game link.

    ``airace-helper://connect?url=<the game's address>&key=<key>`` gives
    ``("wss://<host>/worker", key)`` (``ws://`` for an http game). A plain ``ws://``/``wss://``
    worker address passes through with no key. Anything else is refused."""
    if link.startswith(("ws://", "wss://")):
        return link, None
    parsed = urlparse(link)
    if parsed.scheme != "airace-helper" or parsed.netloc != "connect":
        raise ValueError(f"that isn't a game link: {link!r}")
    query = parse_qs(parsed.query)
    url = (query.get("url") or [""])[0]
    key = (query.get("key") or [None])[0]
    game = urlparse(url)
    if game.scheme not in ("http", "https") or not game.netloc:
        raise ValueError(f"that game link has no game address: {link!r}")
    scheme = "wss" if game.scheme == "https" else "ws"
    return f"{scheme}://{game.netloc}/worker", key
