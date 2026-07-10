"""Strip the leading space from `text` (fixes #3), shifting `tokens`.

Many sentences' `text` carries a single leading space -- an artifact of
how sentences were split from the original Brown Corpus text -- which
this removes, shifting every `tokens` [start, end] pair down by 1 to
match. No other layer needs adjusting: wn16_key/wn30_key/oewn_key are
indexed by token position, not character offset, so a shift in `text`
doesn't touch them.

Edits are targeted substitutions on each document's raw text block, not
a YAML parse/dump round-trip: `text` scalars use varied YAML quoting and
line-wrapping styles that a generic dumper won't reliably reproduce, so
only the exact characters that need to change -- the leading space, and
each digit in `tokens` -- are touched.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

# A top-level document (or `_meta`) key: unindented, ending the line.
_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
# The leading space in a document's (always single-quoted, when present)
# `text` value -- only single-quoting can carry a leading space in YAML.
_TEXT_LEADING_SPACE = re.compile(r"(?m)^(    text: )'( )")
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")


def _shift_tokens(match: re.Match[str]) -> str:
    body = match.group(1)
    shifted = re.sub(r"\d+", lambda m: str(int(m.group()) - 1), body)
    return f"    tokens: {shifted}"


def fix_file(path: Path) -> int:
    """Strip the leading space from affected documents' `text` in `path`.

    Returns the number of documents changed.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    affected = {
        doc_id
        for doc_id, doc in data.items()
        if doc_id != "_meta"
        and isinstance(doc, dict)
        and isinstance(doc.get("text"), str)
        and doc["text"].startswith(" ")
    }
    if not affected:
        return 0

    raw = path.read_text(encoding="utf-8")
    matches = list(_DOC_BOUNDARY.finditer(raw))
    bounds = [m.start() for m in matches] + [len(raw)]
    # Use parsed key order for identity (some doc IDs are quoted in the raw
    # YAML, e.g. `"9114":`, to force string typing on an all-digit ID, so
    # the regex capture and the parsed key can differ textually even
    # though they refer to the same block); only rely on the regex for
    # counting/locating the blocks themselves.
    doc_ids = list(data.keys())
    if len(matches) != len(doc_ids):
        raise RuntimeError(
            f"{path}: found {len(matches)} top-level blocks in raw text but "
            f"{len(doc_ids)} keys when parsed -- refusing to edit"
        )

    pieces = []
    changed = 0
    for i, doc_id in enumerate(doc_ids):
        chunk = raw[bounds[i] : bounds[i + 1]]
        if doc_id in affected:
            chunk, n = _TEXT_LEADING_SPACE.subn(r"\1'", chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text line for {doc_id!r}")
            chunk, n = _TOKENS_LINE.subn(_shift_tokens, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find tokens line for {doc_id!r}")
            changed += 1
        pieces.append(chunk)

    if changed:
        path.write_text("".join(pieces), encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strip the leading space from data/*.yaml's `text` "
        "layer, shifting `tokens` to match (fixes #3)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to update (default: data/)",
    )
    args = parser.parse_args()

    if args.paths:
        files = []
        for p in args.paths:
            files.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])
    else:
        files = find_yaml_files(DATA_DIR)

    total_docs = 0
    changed_files = 0
    for path in files:
        changed = fix_file(path)
        if changed:
            changed_files += 1
            total_docs += changed

    print(f"Changed {total_docs} document(s) across {changed_files} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
