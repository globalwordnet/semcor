"""Align this corpus's capitalization with Brown wherever they differ by
exactly one character's case, in either direction (generalizes #13).

#13 only restored a capital lost in one narrow trigger: a `"` followed
by `.`/`?`/`!`, immediately followed by a lowercase word (152
instances). Running the same whole-document `nltk.corpus.brown`
alignment `semcor-verify-brown` already does (see #12,
`src/semcor/verify_brown.py`) and filtering for single-character,
case-only divergences found the problem is both much bigger and runs
both ways:

- 861 places this corpus has a lowercase letter where Brown has a
  capital -- the same shape of bug as #13, just not gated behind the
  quote+terminal trigger (`Despite` -> `despite` after an ordinary
  period, `Larimer St.` -> `Larimer_st.`, `Jim` -> `jim`), plus large
  blocks of title-case words inside embedded titles and foreign-name
  particles that this corpus had *correctly* normalized away from
  Brown's inconsistent title-casing (`Twilight Of Southern
  Regionalism` -> `Twilight of Southern_Regionalism`, `De Falla` ->
  `de Falla`), and single-letter math/science variables in the
  `learned` genre (`at T` -> `at t`).
- 115 places in the *reverse* direction -- this corpus adding
  capitalization Brown's plaintext doesn't have at all (a book title
  mentioned in running prose that Brown left lowercase but this corpus
  title-cased as `The Revolt Of The Moderates`; rhetorical
  capitalization in dialogue like `Warmongering Capitalists` that
  Brown transcribed as plain lowercase).

Given that mix, there's no clean rule that separates "genuine bug"
from "deliberate improvement over Brown" -- the maintainer was walked
through the full breakdown, including the cases where this corpus
looks more correct than Brown, and decided on unconditional alignment
with Brown in both directions rather than trying to algorithmically
preserve some categories. That decision, not this script, is what
settles the scope; see the issue this fixes for the full writeup.

Every one of these is a single-character, same-length swap -- no
token/offset restructuring, unlike #9/#10: only `text` (and, where
the same position's `lemmas` entry happens to carry the same letter,
`lemmas`) changes, so `tokens`/`pos`/`oewn_key`/`wn16_key`/`wn30_key`
are untouched.

`case-mismatch-fixes.yaml` lists all 976 confirmed fixes as (file,
sentence, offset, target) -- `offset` is the character position within
that sentence's own `text` (not the whole merged document), `target`
is the single character it should become. Generated once, offline,
against `nltk.corpus.brown`; this script has no runtime NLTK
dependency and just applies that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "case-mismatch-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_LEMMAS_LINE = re.compile(r"(?m)^    lemmas: (\[.*\])$")

_WIDTH = 10**9


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, str]]]]:
    """Return {filename: {sentence_id: [(offset, target_char), ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["offset"], entry["target"]))
    return by_file


def fix_sentence(sent: dict, fixes: list[tuple[int, str]]) -> dict | None:
    """Apply a sentence's (offset, target_char) fixes to `text` (and, where
    it lines up, `lemmas`).

    Returns a new sentence dict, or None if every fix was already
    applied (e.g. a second run).
    """
    text = sent["text"]
    applied = False
    text_chars = list(text)
    for offset, target in fixes:
        if offset >= len(text_chars) or text_chars[offset] == target:
            continue  # already applied, or out of range on a stale manifest entry
        text_chars[offset] = target
        applied = True

    if not applied:
        return None

    new_text = "".join(text_chars)

    # Keep `lemmas` consistent where a lemma entry contains the exact
    # same substring at the same relative position as the fixed word in
    # `text` (lemmas inconsistently preserve surface case in this corpus,
    # e.g. `So_that` vs `so_that` -- only touch it where it already
    # mirrors the surface, same as #9 did for em dashes).
    tokens = sent["tokens"]
    lemmas = list(sent["lemmas"])
    lemmas_changed = False
    for i, (s, e) in enumerate(tokens):
        old_word = text[s:e]
        new_word = new_text[s:e]
        if old_word == new_word:
            continue
        lemma = lemmas[i]
        if lemma == old_word:
            lemmas[i] = new_word
            lemmas_changed = True

    new_sent = dict(sent)
    new_sent["text"] = new_text
    if lemmas_changed:
        new_sent["lemmas"] = lemmas
    return new_sent


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def _render_lemmas(lemmas: list[str]) -> str:
    body = ", ".join(
        yaml.safe_dump(v, default_style='"', allow_unicode=True, width=_WIDTH).rstrip("\n") for v in lemmas
    )
    return f"    lemmas: [{body}]"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested case fixes.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, dict] = {}
    for sid, fixes in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        new_sent = fix_sentence(sent, fixes)
        if new_sent is not None:
            changed[sid] = new_sent

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
            new_sent = changed[doc_id]
            text_repl = _render_text(new_sent["text"])
            chunk, n = _TEXT_BLOCK.subn(lambda m: text_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text block for {doc_id!r}")
            if "lemmas" in new_sent and new_sent["lemmas"] != data[doc_id]["lemmas"]:
                lemmas_repl = _render_lemmas(new_sent["lemmas"])
                chunk, n = _LEMMAS_LINE.subn(lambda m: lemmas_repl, chunk, count=1)
                if n != 1:
                    raise RuntimeError(f"{path}: could not find lemmas line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align this corpus's capitalization with Brown "
        "wherever they differ by exactly one character's case, per "
        "case-mismatch-fixes.yaml (generalizes #13)."
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
