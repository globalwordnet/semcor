"""Correct underscore-joined lexemes that should be hyphenated (fixes #43).

Part of #43's scan for hyphens dropped from compound modifiers: this
corpus recognizes 54 hyphenated compounds (`self-acceptance`,
`spring-training`, `pinch-hitters`, ...) as single WordNet-sensed
multiword lexemes -- correctly, unlike the ~702-instance remainder #43
also found (two ordinary separate words with the hyphen missing
entirely, tracked separately, split into #46 since almost all of those
turn out to already carry a real sense on both sides and would need an
individual editorial call to merge) -- but joins them with `_` instead of
the real `-` Brown's text has, e.g. `self_acceptance` where Brown has
`self-acceptance`.

Confirmed against `src/semcor/brown-nolines.txt` (the same reference
`semcor-compare-brown-nolines` uses -- not `nltk.corpus.brown`, which
carries its own divergent tokenization this repo stopped trusting as
ground truth per PR #19) via a context-window word search: for each
underscore-joined token, its parts (rejoined with `-`) had to match a
single word at a unique position in the reference, agreeing with at
least one side's worth of the token's own immediate neighbouring words.

Since `_` and `-` are both one character, correcting this is a same-
length, in-place substitution: the token's span in `tokens` is unchanged,
only the literal characters at that span in `text` are corrected.
`lemmas` gets the same `_` -> `-` substitution applied to *its own*
existing value, not overwritten with the corrected surface outright --
lemmatization can differ arbitrarily from the surface (e.g. surface
`re_arguing` pairs with lemma `re-argue`, already hyphenated, nothing to
fix there), so only a literal underscore in the lemma itself is
corrected. `oewn_key` never encodes spelling (only a synset ID), and
`wn16_key`/`wn30_key` sense keys are spec'd to always use `_` for
multiword lemmas regardless of surface spelling -- so none of the three
sense-key layers change.

`src/semcor/underscore-hyphen-lexeme-fixes.yaml` lists all 54 confirmed
`{file, sentence, index, replacement}` fixes -- kept next to this module
since nothing else needs it; generated once, offline. This script has no
NLTK dependency and just applies that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "underscore-hyphen-lexeme-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_LEMMAS_LINE = re.compile(r"(?m)^    lemmas: (\[.*\])$")

_WIDTH = 10**9


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, str]]]]:
    """Return {filename: {sentence_id: [(index, replacement), ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["index"], entry["replacement"]))
    return by_file


def fix_sentence(sent: dict, fixes: list[tuple[int, str]]) -> dict | None:
    """Apply a sentence's (index, replacement) underscore -> hyphen fixes.

    `index` is only an ordering hint, not trusted to still point at the
    right token (harmless here since this fix never changes token count,
    but kept consistent with every other `fix_*` module's convention of
    locating by surface from an advancing cursor rather than trusting a
    stored index, so a second run is idempotent and a stale manifest
    entry doesn't silently corrupt an unrelated token).

    Returns a new sentence dict, or None if every fix was already
    applied (e.g. a second run).
    """
    text = sent["text"]
    tokens = sent["tokens"]
    lemmas = sent["lemmas"]
    surfaces = [text[s:e] for s, e in tokens]

    applied = []  # (token_idx, replacement), using *current* indices
    cursor = 0
    for _, replacement in sorted(fixes, key=lambda f: f[0]):
        underscored = replacement.replace("-", "_")
        found = None
        for i in range(cursor, len(tokens)):
            if surfaces[i] == replacement:
                cursor = i + 1
                break  # already fixed -- nothing to do
            if surfaces[i] == underscored:
                found = i
                break
        if found is None:
            continue  # already applied, or genuinely gone
        applied.append((found, replacement))
        cursor = found + 1

    if not applied:
        return None

    parts = []
    new_lemmas = list(lemmas)
    cursor = 0
    for idx, replacement in sorted(applied):
        s, e = tokens[idx]
        parts.append(text[cursor:s])
        parts.append(replacement)
        cursor = e
        # The lemma doesn't necessarily mirror the surface (lemmatization
        # can differ arbitrarily, e.g. surface `re_arguing` / lemma
        # `re-argue` -- already hyphenated, nothing to fix); only swap `_`
        # for `-` *within* whatever lemma is already there, never replace
        # it outright with the corrected surface.
        new_lemmas[idx] = lemmas[idx].replace("_", "-")
    parts.append(text[cursor:])
    new_text = "".join(parts)

    new_sent = dict(sent)
    new_sent["text"] = new_text
    new_sent["lemmas"] = new_lemmas
    return new_sent


def _dump_str(value: str) -> str:
    return yaml.safe_dump(value, default_style='"', allow_unicode=True, width=_WIDTH).rstrip("\n")


def _render_str_list(key: str, values: list[str]) -> str:
    body = ", ".join(_dump_str(v) for v in values)
    return f"    {key}: [{body}]"


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested underscore -> hyphen corrections.

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
            lemmas_repl = _render_str_list("lemmas", new_sent["lemmas"])
            chunk, n = _LEMMAS_LINE.subn(lambda m: lemmas_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find lemmas line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correct underscore-joined lexemes that should be "
        "hyphenated in data/*.yaml (fixes #43), per "
        "underscore-hyphen-lexeme-fixes.yaml."
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
