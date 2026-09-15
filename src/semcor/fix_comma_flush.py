"""Move a comma flush against the word before it inside a fused MWE
token (fixes #62), e.g. `Opelika_,_Ala.` -> `Opelika,_Ala.`.

This corpus fuses multi-word proper nouns and titles into a single token
by joining their words with underscores (e.g. `Fulton_County_Grand_Jury`)
-- a deliberate, otherwise-correct convention (see also
`semcor-fix-genitive-gap`, #57, for the same underscore-as-space
convention breaking down for `'s`). It breaks down the same way for a
comma: a comma is always flush against the word before it and followed
by a space, never the reverse, but 20 confirmed tokens have an
underscore on *both* sides of an embedded comma (`Opelika_,_Ala.`,
`Boulder_,_Colorado`, `Hallmark_Cards_,_Inc.`) -- the leading underscore
is always wrong.

Detection is structural (any token whose text contains `_,_`) but every
one of the 20 was individually confirmed against
`src/semcor/brown-nolines.txt`: replacing `_,_` with `,_` and rendering
every remaining `_` as a space reproduces a real, verbatim substring of
Brown's own text. 11 further raw candidates are deliberately excluded
because fixing the comma alone still wouldn't match the reference --
6 also embed a numeric range needing its own flush hyphen (`400_-_401`,
the same shape as #9's number ranges), 4 are Selective Service
classification codes (`4_,_-_D`) using an entirely different escape
convention in the reference, and 1 (`Norman_B._Small_,_Jr.`) is missing
a comma entirely compared to Brown's real text -- a content gap, not a
spacing bug. None of those are in the manifest.

Every fix is a same-token, in-place edit: `str.replace("_,_", ",_")`
deletes one character per embedded comma (some tokens have more than
one, e.g. `Hark_,_Hark_,_the_Lark`), shrinking that one token's own span
and shifting every later token in the sentence left to match -- the same
mechanics as `semcor-fix-caret-diaeresis`'s embedded (no-merge) case.
Token *count* never changes, so `lemmas`/`pos`/every sense-key layer are
completely untouched.

`src/semcor/comma-flush-fixes.yaml` lists all 20 confirmed `{file,
sentence, index}` fixes -- generated once, offline, against
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

MANIFEST_PATH = Path(__file__).resolve().parent / "comma-flush-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")

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


def fix_sentence(
    text: str, tokens: list[list[int]], indices: list[int]
) -> tuple[str, list[list[int]]] | None:
    """Apply a sentence's comma-flush fixes.

    Returns (new_text, new_tokens), or None if every fix was already
    applied (e.g. a second run).
    """
    applied = False
    for index in sorted(indices):
        s, e = tokens[index]
        old = text[s:e]
        if "_,_" not in old:
            if "," in old:
                continue  # already fixed -- no-op
            raise RuntimeError(
                f"manifest entry (index={index}) doesn't match current "
                f"text -- refusing to edit"
            )

        new = old.replace("_,_", ",_")
        removed = len(old) - len(new)
        new_end = s + len(new)

        text = text[:s] + new + text[e:]
        tokens = [
            list(t) if i != index else [s, new_end]
            for i, t in enumerate(tokens)
        ]
        for i in range(index + 1, len(tokens)):
            tokens[i] = [tokens[i][0] - removed, tokens[i][1] - removed]
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
    """Apply this file's manifested comma-flush corrections.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, tuple[str, list[list[int]]]] = {}
    for sid, indices in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        result = fix_sentence(sent["text"], sent["tokens"], indices)
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
        description="Move a comma flush against the preceding word inside "
        "a fused MWE token in data/*.yaml (fixes #62), per "
        "comma-flush-fixes.yaml."
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
