"""Restore thousands-separator commas stripped from numbers (fixes #15).

Whatever step normalized numbers in this corpus dropped the `,` digit-group
separator along with merging the digit groups into a single space-free
token, e.g. Brown's `126,000` became this corpus's `126000`.

Cross-checking every instance against `nltk.corpus.brown` (same whole-document
`difflib` alignment `semcor-verify-brown` uses, see #12) found 411 confirmed
single-token fixes, but a naive "replace this corpus's token with Brown's
aligned word" doesn't work uniformly:

- For most, this corpus's token and Brown's word are the same digits, just
  missing commas (`126000` -> `126,000`).
- For 168, Brown's tokenizer merged a leading `$` into the number as one
  word (`$1,200`), but this corpus (consistent with how `$` is tokenized
  everywhere else in this corpus) keeps `$` as its own separate token
  already -- using Brown's word verbatim would duplicate the `$`.
- For 11, Brown tokenizes a whole hyphenated compound modifier as one word
  (`75,000-ton`, `$10,000-per-year`, `BOD/day/1,000`) that this corpus
  already splits into several tokens, the same shape of tokenization
  difference #9 found and left out of scope for the em-dash fix.

`thousands-separator-fixes.yaml`'s `replacement` for each entry is already
just the matching digit run pulled out of Brown's aligned word (not the
whole word) -- generated once, offline, against `nltk.corpus.brown`; this
script has no runtime NLTK dependency and just applies that manifest.

A confirmed ~52 further missing commas are ordinary sentence commas (list
items, appositives) with no shared cause -- not part of this fix, see the
issue this fixes for the follow-up.

Every fix here grows one token in place -- unlike #9's em-dash fix, there's
no spurious whitespace padding to absorb, so this only ever expands the
token's own span (never touching a neighbouring gap), shifts every
downstream token's offsets by the length delta, and keeps that token's own
`lemmas` entry consistent with the new surface. `pos`/`oewn_key`/`wn16_key`/
`wn30_key` are untouched.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "thousands-separator-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")
_LEMMAS_LINE = re.compile(r"(?m)^    lemmas: (\[.*\])$")

_WIDTH = 10**9


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, str]]]]:
    """Return {filename: {sentence_id: [(token_index, replacement), ...]}},
    keyed by basename (e.g. `br-a01.yaml`) rather than the manifest's own
    relative path, so lookups don't depend on the caller's cwd.
    """
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["index"], entry["replacement"]))
    return by_file


def apply_fixes(
    text: str, tokens: list[list[int]], lemmas: list[str], fixes: list[tuple[int, str]]
) -> tuple[str, list[list[int]], list[str]]:
    """Apply a sentence's (token_index, replacement) fixes.

    Each fix replaces exactly the token's own span with `replacement` --
    unlike #9's em-dash fix, there's no spurious padding to absorb here
    (any whitespace next to the token is legitimate and must be left
    alone), so the region is always the token's own (start, end), never
    extended into a neighbouring gap.
    """
    regions = []  # (left, right, replacement, token_index), sorted, non-overlapping
    for index, replacement in sorted(fixes):
        s, e = tokens[index]
        regions.append((s, e, replacement, index))

    parts = []
    cursor = 0
    for left, right, replacement, _ in regions:
        parts.append(text[cursor:left])
        parts.append(replacement)
        cursor = right
    parts.append(text[cursor:])
    new_text = "".join(parts)

    def shift(offset: int) -> int:
        delta = 0
        for left, right, replacement, _ in regions:
            if right <= offset:
                delta += len(replacement) - (right - left)
            else:
                break
        return offset + delta

    replaced_at = {index: (left, right, replacement) for left, right, replacement, index in regions}
    new_tokens = []
    new_lemmas = list(lemmas)
    for i, (s, e) in enumerate(tokens):
        if i in replaced_at:
            left, right, replacement = replaced_at[i]
            delta = sum(
                len(r) - (rt - l) for l, rt, r, idx2 in regions if rt <= left
            )
            new_s = left + delta
            new_e = new_s + len(replacement)
            new_tokens.append([new_s, new_e])
            new_lemmas[i] = replacement
        else:
            new_tokens.append([shift(s), shift(e)])

    return new_text, new_tokens, new_lemmas


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def _render_tokens(tokens: list[list[int]]) -> str:
    tokens_yaml = yaml.safe_dump(tokens, default_flow_style=True, allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    tokens: {tokens_yaml}"


def _render_lemmas(lemmas: list[str]) -> str:
    # default_style='"' matches every other string list in these files
    # (lemmas, pos, *_key values); plain safe_dump would pick unquoted or
    # single-quoted scalars per PyYAML's own quoting-minimization rules,
    # which round-trips fine but looks inconsistent next to untouched rows.
    lemmas_yaml = yaml.safe_dump(
        lemmas, default_flow_style=True, default_style='"', allow_unicode=True, width=_WIDTH
    ).rstrip("\n")
    return f"    lemmas: {lemmas_yaml}"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested fixes.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, tuple[str, list[list[int]], list[str]]] = {}
    for sid, fixes in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        new_text, new_tokens, new_lemmas = apply_fixes(
            sent["text"], sent["tokens"], sent["lemmas"], fixes
        )
        # Already applied (e.g. a second run): skip so reporting/writes
        # stay honest about what actually changed, same as the other
        # fix-* scripts here being idempotent.
        if (new_text, new_tokens, new_lemmas) == (sent["text"], sent["tokens"], sent["lemmas"]):
            continue
        changed[sid] = (new_text, new_tokens, new_lemmas)

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
            new_text, new_tokens, new_lemmas = changed[doc_id]
            text_repl = _render_text(new_text)
            chunk, n = _TEXT_BLOCK.subn(lambda m: text_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text block for {doc_id!r}")
            tokens_repl = _render_tokens(new_tokens)
            chunk, n = _TOKENS_LINE.subn(lambda m: tokens_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find tokens line for {doc_id!r}")
            lemmas_repl = _render_lemmas(new_lemmas)
            chunk, n = _LEMMAS_LINE.subn(lambda m: lemmas_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find lemmas line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore thousands-separator commas stripped from "
        "numbers in data/*.yaml's `text` (fixes #15), per "
        "thousands-separator-fixes.yaml."
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
