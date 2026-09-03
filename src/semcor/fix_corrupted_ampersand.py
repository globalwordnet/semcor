"""Restore literal '&' characters corrupted to '+' (fixes #17).

#8/#9/#11 already established that this corpus decodes one Brown
transcription escape convention: a literal `&` in the tagged corpus
marks a non-sentence-final abbreviation period (`Mr&` -> `Mr.`). That
convention means the original transcription needed a *different* escape
for an actual, literal ampersand character in running text -- and it
used `+`. This corpus never decoded that second convention, so `A & M`
(Texas A&M) stayed as `A_+_M` instead of becoming `A & M`.

An exhaustive scan of every token in `data/*.yaml` containing a literal
`+` (81 total, not a sample) found 76 confirmed corruptions -- virtually
all proper-noun ampersands (`Chesapeake + Ohio`, `Smith + Wesson`,
`Standard + Poor's`, `AF + AM`, ...), each checked against
`nltk.corpus.brown` (which does decode this convention) and local
context. A whole-document diff against `nltk.corpus.brown` alone (the
method #9/#13/#15/#16 use) isn't reliable here on its own: its greedy
longest-match behavior missed a second `B + O` mention sitting close to
an already-resolved one in the same sentence (`br-a27.yaml`, `j50s`) --
caught only by cross-checking against this direct token scan instead.
`sls.hawaii.edu`'s raw dump (the more faithful source #16 used) isn't
the right comparison for *this* fix either: it has the identical
undecoded `+` in all 76 places, since it preserves the same escape
convention this corpus does.

The remaining 5 `+`-containing tokens are genuine, unrelated plus signs
-- three statistical correlation/discrepancy values (`+ .50`, `+ .04`,
`+ .7`), one sentence explaining the `+` symbol itself, and one school
grade (`B + student`) -- and are correctly left untouched.

Every fix is a single-character, same-length swap, same as #28: no
token/offset restructuring, so `tokens`/`pos`/`oewn_key`/`wn16_key`/
`wn30_key` are untouched. `corrupted-ampersand-fixes.yaml` lists all 78
confirmed `+` character offsets (two tokens have more than one); the
target character is always `&`, so it isn't repeated per entry.
Generated once, offline, against `nltk.corpus.brown`; this script has
no runtime NLTK dependency and just applies that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "corrupted-ampersand-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_LEMMAS_LINE = re.compile(r"(?m)^    lemmas: (\[.*\])$")

_WIDTH = 10**9
_TARGET = "&"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[int]]]:
    """Return {filename: {sentence_id: [offset, ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append(entry["offset"])
    return by_file


def fix_sentence(sent: dict, offsets: list[int]) -> dict | None:
    """Swap each offset's character in `text` from '+' to '&' (and, where
    it lines up, `lemmas`).

    Returns a new sentence dict, or None if every fix was already
    applied (e.g. a second run).
    """
    text = sent["text"]
    applied = False
    text_chars = list(text)
    for offset in offsets:
        if offset >= len(text_chars) or text_chars[offset] != "+":
            continue  # already applied, or out of range on a stale manifest entry
        text_chars[offset] = _TARGET
        applied = True

    if not applied:
        return None

    new_text = "".join(text_chars)

    # Keep `lemmas` consistent where a lemma entry mirrors the fixed
    # word's old surface at the same token -- either exactly, or (unlike
    # #28, which only checks the exact case) lowercased, since some
    # lemmas here are a plain lowercased mirror of the surface (`R_+_D`
    # -> lemma `r_+_d`) rather than a semantic WordNet lemma.
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
        elif lemma == old_word.lower():
            lemmas[i] = new_word.lower()
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
    """Apply this file's manifested '+' -> '&' fixes.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, dict] = {}
    for sid, offsets in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        new_sent = fix_sentence(sent, offsets)
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
            if "lemmas" in new_sent:
                lemmas_repl = _render_lemmas(new_sent["lemmas"])
                chunk, n = _LEMMAS_LINE.subn(lambda m: lemmas_repl, chunk, count=1)
                if n != 1:
                    raise RuntimeError(f"{path}: could not find lemmas line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore literal '&' characters corrupted to '+' in "
        "data/*.yaml's `text` (fixes #17), per corrupted-ampersand-fixes.yaml."
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
