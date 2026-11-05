"""Packs the helper's code for friends' PCs (`npm run host`): ml/src, pyproject.toml and uv.lock.

Usage: python pack.py <ml folder> <out zip>. Written next to the target, then swapped in, so a
download never gets half a zip.
"""

import os
import sys
import zipfile


def main() -> None:
    ml, out = sys.argv[1], sys.argv[2]
    partial = out + ".partial"
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("pyproject.toml", "uv.lock"):
            z.write(os.path.join(ml, name), name)
        src = os.path.join(ml, "src")
        for base, dirs, files in os.walk(src):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.endswith(".egg-info"))
            for f in sorted(files):
                if f.endswith(".pyc"):
                    continue
                path = os.path.join(base, f)
                z.write(path, os.path.relpath(path, ml).replace(os.sep, "/"))
    os.replace(partial, out)


if __name__ == "__main__":
    main()
