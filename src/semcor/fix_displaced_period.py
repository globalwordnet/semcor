"""Move an abbreviation's period off the following number and back onto
the abbreviation itself (fixes #69).

When an abbreviation (`Nov`, `No`, `Figs`, `pp`, ...) is immediately
followed by a number, the period ends up attached to the *number*
instead of the abbreviation: `Nov .8` for Brown's `Nov. 8`, `No .3` for
`No. 3`. Detection: a bare-letters token immediately followed by a
single space then a `.` + digits token. Most matches for this shape are
ordinary decimal numbers with a real word before them (`batted .365`,
`the .028`) and are already correct -- confirmed individually against
`src/semcor/brown-nolines.txt` (both shapes are genuinely present in
Brown's real text, so this can't be a blanket rule). Only the 12
instances where the preceding word is itself a recognized abbreviation
(`Nov`/`Oct`/`Sept`, `No`, `Figs`, `pp`) are real bugs, each confirmed
individually.

The fix is a 2-character swap: the gap (a space, at `pos`) and the
following period (at `pos + 1`) trade places, moving the period onto the
abbreviation and leaving a single space before the number. Token *count*
never changes: the abbreviation token's own end and the number token's
own start each shift by one character (in opposite directions from each
other, but both simply `+= 1` in absolute terms since the swap doesn't
move anything else) -- no other token in the sentence is affected, since
only these two characters change position. `lemmas`/`pos`/every
sense-key layer are completely untouched.

`src/semcor/displaced-period-fixes.yaml` lists all 12 confirmed `{file,
sentence, pos}` fixes (`pos` is the character offset of the space to
swap with the period after it) -- generated once, offline, against
`brown-nolines.txt`, this script has no NLTK dependency and just applies
that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "displaced-period-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")

_WIDTH = 10**9


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[int]]]:
    """Return {filename: {sentence_id: [pos, ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append(entry["pos"])
    return by_file


def fix_sentence(
    text: str, tokens: list[list[int]], positions: list[int]
) -> tuple[str, list[list[int]]] | None:
    """Apply a sentence's displaced-period fixes.

    Returns (new_text, new_tokens), or None if every fix was already
    applied (e.g. a second run).
    """
    applied = False
    tokens = [list(t) for t in tokens]
    for pos in positions:
        if text[pos] == "." and text[pos + 1] == " ":
            continue  # already fixed -- no-op
        if not (text[pos] == " " and text[pos + 1] == "."):
            raise RuntimeError(
                f"manifest entry (pos={pos}) doesn't match current text "
                f"-- refusing to edit"
            )
        text = text[:pos] + "." + " " + text[pos + 2 :]
        for i, (s, e) in enumerate(tokens):
            if e == pos:
                tokens[i][1] = pos + 1
            if s == pos + 1:
                tokens[i][0] = pos + 2
        applied = True

    if not applied:
        return None
    return text, tokens


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def _render_tokens(tokens: list[list[int]]) -> str:
    body = ", ".join(f"[{s}, {e}]" for s, e in tokens)
    return f"    tokens: [{body}]"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested displaced-period corrections.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, tuple[str, list[list[int]]]] = {}
    for sid, positions in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        result = fix_sentence(sent["text"], sent["tokens"], positions)
        if result is not None:
            changed[sid] = result

    if not changed or dry_run:
        return list(changed)

    raw = path.read_text(encoding="utf-8")
    matches = list(_DOC_BOUNDARY.finditer(raw))
    bounds = [m.start() for m in matches] + [len(raw)]
    doc_ids = list(data.keys())
    if len(matches) != len(doc_ids):
        raise RuntimeError(
            f"{path}: found {len(matches)} top-level blocks in raw text but "
            f"{len(doc_ids)} keys when parsed -- refusing to edit"
        )

    pieces = []
    for i, doc_id in enumerate(doc_ids):
        chunk = raw[bounds[i] : bounds[i + 1]]
        if doc_id in changed:
            new_text, new_tokens = changed[doc_id]
            text_repl = _render_text(new_text)
            chunk, n = _TEXT_BLOCK.subn(lambda m: text_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text block for {doc_id!r}")
            tokens_repl = _render_tokens(new_tokens)
            chunk, n = _TOKENS_LINE.subn(lambda m: tokens_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find tokens line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move an abbreviation's period off the following "
        "number in data/*.yaml (fixes #69), per "
        "displaced-period-fixes.yaml."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to update (default: data/)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Manifest file to read (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing anything",
    )
    args = parser.parse_args()

    if args.paths:
        files = []
        for p in args.paths:
            files.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])
    else:
        files = find_yaml_files(DATA_DIR)

    manifest = load_manifest(args.manifest)

    total_sentences = 0
    changed_files = 0
    for path in files:
        changed = fix_file(path, manifest, dry_run=args.dry_run)
        if changed:
            changed_files += 1
            total_sentences += len(changed)

    verb = "Would fix" if args.dry_run else "Fixed"
    print(f"{verb} {total_sentences} sentence(s) across {changed_files} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
