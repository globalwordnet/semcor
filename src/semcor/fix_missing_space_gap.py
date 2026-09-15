"""Insert a missing separator between two words that should be split
apart -- either two already-separate tokens with nothing between them,
or one existing token that swallowed both (fixes #68).

Mirror image of `semcor-fix-reconstruction-gaps` (#63/#64): instead of
finding N of our words that collapse to 1 reference word (an extra
separator to remove), this looks for the reverse -- 1 of our words that
should split into N reference words (a separator missing entirely) --
using the same whole-document `difflib` alignment plus an exact-
concatenation check, just run backwards. 205 raw candidates found.

Two shapes, both a pure single-character insertion (the mirror image of
`semcor-fix-genitive-gap`'s deletion) needing no lemma/pos/sense change
at all -- verified by checking that every split point this fix would
insert at lands exactly on an existing token boundary (shape one) or
strictly inside exactly one existing token's span (shape two); anything
else is left alone rather than guessed at:

- **180 instances** where this corpus's own `tokens` *already* has the
  two (or more) words as separate, correctly tokenized tokens -- `and`/
  `other` in `andother`, `23d`/`ward` in `23dward` -- flush against each
  other with nothing between. Insert a literal space; token *count*
  never changes, only the inserted position's own token and everything
  after it in the sentence shift right by one.
- **21 instances** where the gap is missing *inside* a single existing
  token that's otherwise correct -- an organization/person name
  (`Eligio_(Kika)de_la_Garza`, missing the gap after `(Kika)`), an
  abbreviation (`D.C.` for "direct current", missing the gap after
  `D.`), a historical spelling this corpus's own sense already covers
  correctly (`76-percent` for Brown's `76-per cent`, both meaning the
  same `percent` WordNet sense; confirmed as a real, common spelling --
  102 occurrences of `per cent` in `brown-nolines.txt`, not a one-off --
  unlike the excluded typos below). Insert this corpus's own established
  MWE-joining `_` instead of a literal space (the same convention
  `e._g.` already uses elsewhere in this corpus): the token's own span
  just grows by one, keeping its existing `lemmas`/`pos`/sense exactly
  as they were, rather than raising the kind of editorial question
  #43/#46 hit for token *merges* about which half (if either) keeps the
  sense.

4 raw candidates are deliberately left alone:
- 2 are typos in `brown_nolines.txt` itself (confirmed directly in the
  reference file: `to d o whatever`, `the s ame level`), not bugs in
  this corpus's already-correctly-spelled `do`/`same`.
- 2 involve `**f`, the formula-placeholder escape this repo already
  leaves alone elsewhere (`semcor-fix-reconstruction-gaps`'s own
  docstring) as tangled up with the already-tracked #16/#34 gap rather
  than clean reference-side noise.

`src/semcor/missing-space-fixes.yaml` lists all 201 confirmed `{file,
sentence, pos, char}` fixes (`pos` is the character offset to insert at;
`char` is `' '` or `'_'`, per the two shapes above) -- generated once,
offline, against `brown-nolines.txt`, this script has no NLTK dependency
and just applies that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "missing-space-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")

_WIDTH = 10**9


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, str]]]]:
    """Return {filename: {sentence_id: [(pos, char), ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["pos"], entry["char"]))
    return by_file


def fix_sentence(
    text: str, tokens: list[list[int]], fixes: list[tuple[int, str]]
) -> tuple[str, list[list[int]]] | None:
    """Apply a sentence's missing-space-gap fixes.

    `char` is `' '` for the ordinary case -- inserting between two
    already-separate, already-flush tokens -- or `'_'` when the gap is
    *inside* a single existing token that's otherwise correct (an
    organization/person name, an abbreviation like `e._g.`): the missing
    separator is filled in with this corpus's own established MWE-joining
    convention instead of a literal space, so the token's own span just
    grows by one and its `lemmas`/`pos`/sense stay exactly as they were.

    Returns (new_text, new_tokens), or None if every fix was already
    applied (e.g. a second run). Fixes are applied highest-position-first
    so an earlier insertion never shifts a not-yet-applied position.
    """
    applied = False
    for pos, char in sorted(fixes, key=lambda f: f[0], reverse=True):
        if char == " ":
            # A token boundary must sit exactly at `pos` (that's what
            # makes this a pure gap-insertion, not a token split) -- if
            # the two tokens on either side aren't already flush there,
            # either this fix already ran, or the manifest is stale.
            if not any(e == pos for _, e in tokens) or not any(s == pos for s, _ in tokens):
                continue  # already fixed (or stale) -- no-op
        else:
            # `pos` must sit strictly inside one existing token's span --
            # if no token still straddles it, this fix already ran (that
            # token's own span grew past `pos`), or the manifest is stale.
            if not any(s < pos < e for s, e in tokens):
                continue  # already fixed (or stale) -- no-op
        text = text[:pos] + char + text[pos:]
        tokens = [
            [s + 1 if s >= pos else s, e + 1 if e > pos else e]
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
    """Apply this file's manifested missing-space-gap corrections.

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
        description="Insert a missing space between two already-adjacent "
        "tokens in data/*.yaml (fixes #68), per missing-space-fixes.yaml."
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
