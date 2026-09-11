"""Close every remaining spurious single-character gap found by a
whole-document alignment against Brown's real text (fixes #63).

Every narrower fix in this repo (quote-gaps in #8, `'s` in #57, the
abbreviation period in #61, the comma in #62, ...) works the same way:
find places where this corpus's own reconstructed text has an extra
space or MWE-joining underscore that Brown's real text doesn't, and
delete it. Each of those was found by a *local* signal specific to its
own pattern (a quote token, a `'s` fragment, ...), which necessarily
misses anything that doesn't match that specific shape.

This fix instead asks the general question directly: reusing the same
per-document word-list alignment `semcor-compare-brown-nolines` already
does (`difflib.SequenceMatcher` between this corpus's reconstructed
words and `brown-nolines.txt`'s, the same alignment
`_adopt_our_casing` uses for `{...}` spans), find every `replace` block
where several of *our* words concatenate, verbatim and exactly, to one
reference word. An exact string match after whole-document alignment
isn't a guess -- unlike the quote-direction problem #8 documents as
unsafe to infer structurally, here the reference is asked directly, so
nesting/ambiguity concerns don't apply.

**2353 confirmed instances** across 293 files, covering (among much
else) #61's abbreviation-period remainder (`N._Y.` -> `N.Y.`, `p.m.` `.`
-> `p.m..`), #62's comma-flush remainder (the embedded number-range
hyphens, `607_-_608.` -> `607-608.`), more of #8's own quote-gap pattern
than the original context-window manifest could uniquely confirm, and a
long tail of previously-uncatalogued shapes (ordinal suffixes `72nd`,
race/time notation `2:36h;`, citation abbreviations `U.S.C.`,
parenthetical flush-ness `(1955).`, apostrophe-prefixed names
`B'dikkat`). Every one of the gaps between the merged words turns out to
be exactly one character -- a plain space or an underscore -- never
anything more complex. 15 further candidates are excluded because the
gap spans a sentence boundary in this corpus's own data (e.g.
`appellant".)` ending one sentence right before a lone `.` starts the
next) -- a different, more structural issue than a single stray
character, left for a separate look.

Each fix is a single-character deletion, exactly like
`semcor-fix-genitive-gap`/`semcor-fix-abbreviation-period-gap`: token
*count* never changes, only the deleted position's own token and
everything after it in the sentence shift left by one. `lemmas`/`pos`/
every sense-key layer are untouched -- only `text` and the shifted
`tokens` offsets change. A sentence needing more than one deletion (an
N-way merge contributes N-1 gaps) is handled by applying its positions
highest-first, so an earlier deletion never invalidates a
not-yet-applied position later in the same pass.

`src/semcor/reconstruction-gap-fixes.yaml` lists all 2353 confirmed
`{file, sentence, pos}` fixes (`pos` is the character offset of the
space/underscore to delete) -- generated once, offline, against
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

MANIFEST_PATH = Path(__file__).resolve().parent / "reconstruction-gap-fixes.yaml"

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
    """Apply a sentence's reconstruction-gap fixes.

    Returns (new_text, new_tokens), or None if every fix was already
    applied (e.g. a second run). Positions are applied highest-first so
    an earlier deletion never shifts a not-yet-applied position.

    This manifest grows over time (each new sweep appends more entries
    rather than replacing it), so a position from an earlier sweep can
    already be out of range against a sentence a *later* sweep's fixes
    already shrank -- out of range means already applied, same as a
    non-space/underscore character there.
    """
    applied = False
    for pos in sorted(positions, reverse=True):
        if pos >= len(text) or text[pos] not in (" ", "_"):
            continue  # already fixed (or stale/out of range) -- no-op
        text = text[:pos] + text[pos + 1 :]
        tokens = [
            [s - 1 if s > pos else s, e - 1 if e > pos else e]
            for s, e in tokens
        ]
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
    """Apply this file's manifested reconstruction-gap corrections.

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
        description="Close every remaining spurious single-character gap "
        "in data/*.yaml (fixes #63), per reconstruction-gap-fixes.yaml."
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
