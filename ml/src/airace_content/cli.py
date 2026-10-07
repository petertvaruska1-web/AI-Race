"""``airace-content``: build the training corpora.

    airace-content build --scale {tiny,full} [--datasets a,b] [--out PATH]
                         [--retrain-tokenizer] [--fixtures DIR]

Sources are streamed from the Hugging Face hub, except that ``--fixtures DIR`` (tiny scale only)
reads them from local ``<source id>.jsonl`` files, so a tiny build can run offline. Exit codes:
0 on success, 2 for a bad argument, 1 when the build cannot run as asked.
"""

import argparse
import sys
from pathlib import Path

from airace_content.build import BuildError, BuildSummary, build_corpus
from airace_content.sources import fixture_fetch, hf_fetch
from airace_ml.data.corpus import DATASET_IDS
from airace_ml.paths import corpus_dir, data_root


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # report instead of exiting, so main() returns the code
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="airace-content", description="Build the AI Race training corpora.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the corpora and (once) the tokenizer")
    build.add_argument("--scale", required=True, choices=("tiny", "full"))
    build.add_argument(
        "--datasets",
        help=f"comma-separated datasets to build (default: all of {','.join(DATASET_IDS)})",
    )
    build.add_argument(
        "--out", type=Path, help="data root to build into (default: AIRACE_DATA or <repo>/data)"
    )
    build.add_argument(
        "--retrain-tokenizer",
        action="store_true",
        help="train the tokenizer again (all datasets only)",
    )
    build.add_argument(
        "--fixtures",
        type=Path,
        metavar="DIR",
        help="read sources from DIR/<source>.jsonl (tiny scale only)",
    )
    return parser


def _datasets(value: str | None) -> list[str]:
    if value is None:
        return list(DATASET_IDS)
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise _UsageError("--datasets needs at least one dataset")
    unknown = [name for name in names if name not in DATASET_IDS]
    if unknown:
        raise _UsageError(
            f"unknown dataset(s) {', '.join(unknown)}; expected some of {', '.join(DATASET_IDS)}"
        )
    return list(dict.fromkeys(names))


def format_summary(summary: BuildSummary, out_root: Path) -> str:
    lines = [f"corpus written to {corpus_dir(out_root)}"]
    for ds, entry in summary.datasets.items():
        noise = ", ".join(
            f"{kind} {count}"
            for kind, count in entry["noise_counts"].items()
            if kind != "none" and count
        )
        target = (
            f" ({entry['tokens'] / entry['target_tokens']:.0%} of target)"
            if "target_tokens" in entry
            else ""
        )
        lines.append(
            f"  {ds:<14}{entry['docs']:>9,} docs {entry['tokens']:>12,} tokens{target}"
            f"  held out {entry['heldout_docs']:,}  noise: {noise or 'none'}"
        )
    lines.append(f"tokenizer: {'trained' if summary.tokenizer_trained else 'frozen, reused'}")
    lines.append(f"done in {summary.seconds:.1f} s")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        datasets = _datasets(args.datasets)
        if args.fixtures is not None:
            if args.scale != "tiny":
                raise _UsageError("--fixtures works only with --scale tiny")
            if not args.fixtures.is_dir():
                raise _UsageError(f"fixture directory not found: {args.fixtures}")
    except _UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:  # --help
        return e.code if isinstance(e.code, int) else 0

    fetch = fixture_fetch(args.fixtures) if args.fixtures is not None else hf_fetch
    out_root = args.out if args.out is not None else data_root()
    try:
        summary = build_corpus(
            args.scale, out_root, fetch, datasets=datasets, retrain_tokenizer=args.retrain_tokenizer
        )
    except (BuildError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(format_summary(summary, out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
