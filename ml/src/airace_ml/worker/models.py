"""Trained models a worker keeps at hand for evals and chat (plan Task 18).

A model comes from the server once, as a checkpoint tar, and is kept on disk under the
worker's models folder. Recently used ones stay loaded: the newest ``capacity`` on the compute
device, the next ``capacity`` in CPU memory, older ones only on disk.
"""

import shutil
import threading
from collections import OrderedDict
from pathlib import Path

import torch

from airace_ml.infer.lm import TorchLM
from airace_ml.model.checkpoint import META_NAME, load_checkpoint
from airace_ml.tokenizer import Tok
from airace_ml.worker.blobs import download_tar


class ModelCache:
    def __init__(self, models_dir: Path, tok: Tok, device: torch.device, capacity: int = 4) -> None:
        self.models_dir = Path(models_dir)
        self.tok = tok
        self.device = device
        self.capacity = capacity
        self.downloads = 0
        self._on_device: OrderedDict[str, tuple[TorchLM, threading.Lock]] = OrderedDict()
        self._on_cpu: OrderedDict[str, tuple[TorchLM, threading.Lock]] = OrderedDict()
        self._lock = threading.Lock()
        self._fetching: dict[str, threading.Lock] = {}

    def get(self, version_id: str, url: str) -> tuple[TorchLM, threading.Lock]:
        """The model and a lock to hold while using it (one generation at a time per model)."""
        with self._lock:
            if version_id in self._on_device:
                self._on_device.move_to_end(version_id)
                return self._on_device[version_id]
            if version_id in self._on_cpu:
                lm, lock = self._on_cpu.pop(version_id)
                lm.model.to(self.device)
                lm.device = self.device
                return self._keep((lm, lock), version_id)
            fetch = self._fetching.setdefault(version_id, threading.Lock())
        with fetch:  # one download per version, even when two jobs ask at once
            folder = self.fetch(version_id, url)
        model, _ = load_checkpoint(folder, self.device)
        if model.tok_emb.num_embeddings != self.tok.vocab_size:
            raise ValueError(
                f"model {version_id} has a {model.tok_emb.num_embeddings}-word vocabulary; "
                f"this worker's tokenizer has {self.tok.vocab_size}"
            )
        with self._lock:
            if version_id in self._on_device:  # another job loaded it meanwhile
                return self._on_device[version_id]
            return self._keep((TorchLM(model, self.tok, self.device), threading.Lock()), version_id)

    def fetch(self, version_id: str, url: str) -> Path:
        """The model's folder on disk, downloaded if it isn't there yet."""
        folder = self.models_dir / version_id
        if (folder / META_NAME).is_file():
            return folder
        shutil.rmtree(folder, ignore_errors=True)  # a download that stopped halfway
        partial = self.models_dir / f"{version_id}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        download_tar(url, partial)
        self.downloads += 1
        partial.rename(folder)
        return folder

    def _keep(
        self, entry: tuple[TorchLM, threading.Lock], version_id: str
    ) -> tuple[TorchLM, threading.Lock]:
        """Puts ``entry`` first on the device, moving the oldest to the CPU and dropping beyond that."""
        self._on_device[version_id] = entry
        while len(self._on_device) > self.capacity:
            old_id, (old, old_lock) = self._on_device.popitem(last=False)
            old.model.to("cpu")
            old.device = torch.device("cpu")
            self._on_cpu[old_id] = (old, old_lock)
        while len(self._on_cpu) > self.capacity:
            self._on_cpu.popitem(last=False)
        return entry
