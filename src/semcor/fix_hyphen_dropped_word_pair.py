"""Restore a hyphen dropped between two ordinary words (fixes #43/#46).

#43 found this corpus sometimes drops a compound modifier's hyphen
entirely, leaving two separately-tokenized words with an ordinary space
between them where Brown's real text has one hyphenated word
(`term end` for `term-end`, `ever growing` for `ever-growing`). #46 split
off the bulk of these (549 confirmed) because almost every one already
carries a real WordNet sense on *both* words, and #24's own precedent
(merging a number and a word into one token, dropping the number's
always-generic sense) doesn't transfer: here neither side's sense is
safely droppable, and no compound-specific sense exists for any of these
in `external/english-wordnet`'s own source data, so merging into a
single token would force a real, unrecoverable editorial choice.

That choice turns out to be unnecessary. Unlike #16/#24's placeholder-
and number-merges, there's no need to merge these two tokens into one at
all: a hyphen is exactly one character, the same width as the space it
replaces, so swapping the character *between* two adjacent tokens
leaves both tokens' own spans completely unchanged -- both words keep
their own separate, untouched sense (`oewn_key`/`wn16_key`/`wn30_key`),
`lemmas`, and `pos`. Only `text` changes, at exactly the one character
position between the two tokens; `tokens` itself needs no update at all
since neither span moves. This also covers the one genuine hyphen-range
case found alongside the compound modifiers (`September October` for
Brown's `September-October`, the same shape as #9's number ranges like
`1960-1962` -- which this corpus already keeps as separate tokens with
the hyphen as its own token, not merged either).

Re-scanning against `src/semcor/brown-nolines.txt` (the same reference
`semcor-compare-brown-nolines` uses) with a context-window word search
-- requiring a unique match for the literal `A-B` reference word, using
context from both sides of the pair and pulling extra words from
neighbouring sentences when the current one runs out (the same
technique the #8 quote-gap follow-up used) -- confirms 522 instances
(508 with a sense on both words, 14 with a sense on only one, 0 with
neither -- the sole unsensed instance #46 already found and fixed
directly is why that bucket is empty here).

`src/semcor/hyphen-dropped-word-pair-fixes.yaml` lists all 522 confirmed
`{file, sentence, index}` fixes (`index` is the first of the two
tokens) -- kept next to this module since nothing else needs it;
generated once, offline, against `brown-nolines.txt`, this script has no
NLTK/Brown runtime dependency and just applies that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "hyphen-dropped-word-pair-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)

_WIDTH = 10**9


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[int]]]:
    """Return {filename: {sentence_id: [index, ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append(entry["index"])
    return by_file


def fix_sentence(sent: dict, indices: list[int]) -> str | None:
    """Apply a sentence's dropped-hyphen fixes.

    Returns the new `text`, or None if every fix was already applied
    (e.g. a second run).
    """
    text = sent["text"]
    tokens = sent["tokens"]

    applied = False
    for index in sorted(indices):
        s0, e0 = tokens[index]
        s1, e1 = tokens[index + 1] if index + 1 < len(tokens) else (None, None)
        if s1 == e0 + 1 and text[e0:s1] == "-":
            continue  # already fixed -- no-op
        if not (s1 == e0 + 1 and text[e0:s1] == " "):
            raise RuntimeError(
                f"manifest entry (index={index}) doesn't match current "
                f"tokens/text -- refusing to edit"
            )
        text = text[:e0] + "-" + text[s1:]
        applied = True

    return text if applied else None


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested dropped-hyphen corrections.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, str] = {}
    for sid, indices in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        new_text = fix_sentence(sent, indices)
        if new_text is not None:
            changed[sid] = new_text

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
        description="Restore a hyphen dropped between two ordinary words "
        "in data/*.yaml (fixes #43/#46), per hyphen-dropped-word-pair-fixes.yaml."
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
