"""What the data under a data root is, as one digest.

The feasibility gate's runs, the reference judge and the novelty index are all built from the
tokenizer and the eight base corpora. Each records this digest when it is built and is used only
while the data under the root still has the same one, so nothing built from an earlier build of
the data is ever mistaken for something built from this one.
"""

import hashlib
from pathlib import Path

from airace_ml.data.corpus import DATASET_IDS, INFO_FILE, OFFSETS_FILE, TAGS_FILE, TOKENS_FILE
from airace_ml.paths import corpus_dir, tokenizer_path

_CHUNK = 1 << 20


def data_identity(root: Path) -> str:
    """A digest of the tokenizer and every base corpus's documents and tags, so runs and
    instruments built on other data are never reused.

    It reads the tokenizer, and each corpus's ``info.json``, offsets and tags whole, and takes a
    corpus's token file (the large one) by its size alone. The digest depends on paths relative
    to ``root`` only, so a copy of the data has the same identity. ``FileNotFoundError`` if a
    file is missing.
    """
    digest = hashlib.sha256()
    paths = [tokenizer_path(root)]
    for ds in DATASET_IDS:
        folder = corpus_dir(root) / ds
        paths += [folder / INFO_FILE, folder / OFFSETS_FILE, folder / TAGS_FILE]
        digest.update(f"{ds}/{TOKENS_FILE}:{(folder / TOKENS_FILE).stat().st_size};".encode())
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                digest.update(chunk)
    return digest.hexdigest()
