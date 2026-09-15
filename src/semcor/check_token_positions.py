"""Guard against `tokens` spans silently drifting out of alignment with
`text`.

Every fix for #8-#18 edits characters in `text` (removing a spurious
space, restoring a dropped comma, splitting an over-merged token, ...),
which means every `tokens` [start, end] pair at or after the edit has to
shift to match. `semcor-validate` only checks that spans are *in bounds*
-- it can't tell a correctly-shifted span from one that now points at
the wrong word entirely.

This takes a fixed set of sample tokens (by file, sentence, and index
within the sentence -- not by absolute character offset, since that's
exactly what a legitimate fix is allowed to change) and checks that
`text[start:end]` for each one still equals the word recorded when the
sample was taken. A mismatch means something shifted `tokens` without
correspondingly updating `text` (or vice versa) somewhere upstream of
that sample.

This is a tripwire, not a correctness proof: it only catches drift at
the sampled positions, and it *will* need `--generate`-ing again
whenever a fix deliberately changes tokenization at one of those
positions (e.g. splitting `in_which` back into `in` + `which` per #10)
-- that's expected; review the regenerated diff to confirm the changes
are the ones you intended, the same way you'd review any snapshot
update.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

SAMPLES_PATH = Path(__file__).resolve().parents[2] / "token-position-samples.yaml"
DEFAULT_SAMPLES_PER_FILE = 12

# Skip tokens that are pure punctuation (quotes, commas, dashes, ...) --
# they're exactly the kind of token several open issues (#8, #9, #14, ...)
# are going to legitimately change, which would make the fixture noisy to
# maintain. Sampling actual words keeps this focused on catching alignment
# drift, not re-litigating punctuation formatting.
def _has_letter(s: str) -> bool:
    return any(ch.isalpha() for ch in s)


def _iter_file_tokens(path: Path):
    """Yield (sentence_id, index, token_text) for every token in a file,
    in sentence/token order."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)
    for sent_id, sent in data.items():
        if sent_id == "_meta" or not isinstance(sent, dict):
            continue
        text = sent.get("text") or ""
        tokens = sent.get("tokens") or []
        for i, (start, end) in enumerate(tokens):
            yield sent_id, i, text[start:end]


def generate(fixture_path: Path, samples_per_file: int, data_dir: Path) -> int:
    files = find_yaml_files(data_dir)
    all_samples: list[dict] = []

    for path in files:
        rel = path.relative_to(data_dir.parent)
        candidates = [
            (sent_id, i, tok)
            for sent_id, i, tok in _iter_file_tokens(path)
            if _has_letter(tok)
        ]
        if not candidates:
            continue

        n = min(samples_per_file, len(candidates))
        # Evenly spaced indices across the file, not random -- so
        # regenerating without any real change to the file reproduces the
        # same sample set instead of picking new tokens every time.
        picks = sorted({(i * len(candidates)) // n for i in range(n)})

        for idx in picks:
            sent_id, tok_idx, tok = candidates[idx]
            all_samples.append(
                {
                    "file": str(rel),
                    "sentence": sent_id,
                    "index": tok_idx,
                    "text": tok,
                }
            )

    with fixture_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(all_samples, f, allow_unicode=True, sort_keys=False)

    print(f"Wrote {len(all_samples)} sample(s) from {len(files)} file(s) to {fixture_path}.")
    return 0


def verify(fixture_path: Path, repo_root: Path) -> int:
    if not fixture_path.is_file():
        print(f"error: no fixture at {fixture_path}; run with --generate first", file=sys.stderr)
        return 1

    with fixture_path.open("r", encoding="utf-8") as f:
        samples = yaml.load(f, Loader=_YAML_LOADER) or []

    by_file: dict[str, list[dict]] = {}
    for sample in samples:
        by_file.setdefault(sample["file"], []).append(sample)

    failures: list[str] = []
    checked = 0

    for rel_path, file_samples in by_file.items():
        path = repo_root / rel_path
        if not path.is_file():
            failures.append(f"{rel_path}: file no longer exists")
            continue

        with path.open("r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=_YAML_LOADER)

        for sample in file_samples:
            checked += 1
            sent = data.get(sample["sentence"])
            if not isinstance(sent, dict):
                failures.append(
                    f"{rel_path} sentence {sample['sentence']!r}: sentence no longer exists"
                )
                continue

            text = sent.get("text") or ""
            tokens = sent.get("tokens") or []
            index = sample["index"]
            if not (0 <= index < len(tokens)):
                failures.append(
                    f"{rel_path} sentence {sample['sentence']!r} token[{index}]: "
                    f"index out of range (sentence now has {len(tokens)} token(s))"
                )
                continue

            start, end = tokens[index]
            actual = text[start:end]
            if actual != sample["text"]:
                failures.append(
                    f"{rel_path} sentence {sample['sentence']!r} token[{index}]: "
                    f"expected {sample['text']!r}, found {actual!r}"
                )

    if failures:
        print(f"{len(failures)} of {checked} sampled token(s) drifted:\n")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "\nIf this drift is from an intentional fix (e.g. #8-#18), "
            "re-run with --generate and review the diff before committing it."
        )
        return 1

    print(f"All {checked} sampled token(s) still match.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that data/*.yaml's `tokens` spans still point at "
        "the same words as when the sample fixture was generated -- a "
        "tripwire against tokens/text drifting out of alignment."
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate the fixture from the current data/*.yaml instead "
        "of verifying against it. Review the diff before committing.",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=DEFAULT_SAMPLES_PER_FILE,
        help="Sample tokens per file when generating (default: %(default)s)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=SAMPLES_PATH,
        help="Fixture file to read/write (default: %(default)s)",
    )
    args = parser.parse_args()

    repo_root = DATA_DIR.parent
    if args.generate:
        return generate(args.fixture, args.samples_per_file, DATA_DIR)
    return verify(args.fixture, repo_root)


if __name__ == "__main__":
    sys.exit(main())
