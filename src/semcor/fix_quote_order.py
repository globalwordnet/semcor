"""Swap a mis-ordered closing quote pair back to the correct nesting order
(fixes #72).

When a double-quoted span contains a nested single-quoted word/phrase and
both close at the same point, Brown's real text always closes the *inner*
single quote before the *outer* double quote: `elected'".` -- never
`elected"'.`. This corpus has 42 raw `"` immediately followed by `'`
candidates (all at closing positions -- none at opening positions, which
would be the reverse, correct order); each was verified individually
against `src/semcor/brown-nolines.txt` via context-window matching,
reconstructing the target word with the pair swapped back (`elected` +
`'` + `"`) and requiring a unique, context-confirmed match in the
reference. 27 are confirmed as genuine order bugs this way (5 more,
`foreigners'"`/`dead'"`/`all'"`/`ideas'"`/`again'"`, matched a target that
is unique in the *entire* reference document, which is stronger evidence
than the local context window needs, even though the automated
context-window check itself came back ambiguous for those five). The
remaining 15 are left unfixed (5 are sentence-initial with no preceding
word to reconstruct a target from; the rest involve dialect apostrophes
or other irregular preceding tokens that don't reconstruct cleanly).

The fix is a pure 2-character content swap: the `"` and `'` characters
trade places. Since both belong to fixed-position single-character
tokens, no token span changes at all -- only the two characters at `pos`
and `pos + 1` in `text` change content. `tokens`/`lemmas`/`pos`/every
sense-key layer are completely untouched.

`src/semcor/quote-order-fixes.yaml` lists all 27 confirmed `{file,
sentence, pos}` fixes (`pos` is the character offset of the `"` to swap
with the `'` immediately after it) -- generated once, offline, against
`brown-nolines.txt`; this script has no NLTK dependency and just applies
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

MANIFEST_PATH = Path(__file__).resolve().parent / "quote-order-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)

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


def fix_sentence(text: str, positions: list[int]) -> str | None:
    """Apply a sentence's quote-order-swap fixes.

    Returns the new text, or None if every fix was already applied (e.g.
    a second run).
    """
    applied = False
    for pos in positions:
        if text[pos] == "'" and text[pos + 1] == '"':
            continue  # already fixed -- no-op
        if not (text[pos] == '"' and text[pos + 1] == "'"):
            raise RuntimeError(
                f"manifest entry (pos={pos}) doesn't match current text "
                f"-- refusing to edit"
            )
        text = text[:pos] + "'" + '"' + text[pos + 2 :]
        applied = True

    if not applied:
        return None
    return text


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested quote-order corrections.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, str] = {}
    for sid, positions in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        result = fix_sentence(sent["text"], positions)
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
            text_repl = _render_text(changed[doc_id])
            chunk, n = _TEXT_BLOCK.subn(lambda m: text_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text block for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Swap mis-ordered closing quote pairs ('\"' -> '\\'\"') "
        "in data/*.yaml (fixes #72), per quote-order-fixes.yaml."
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
