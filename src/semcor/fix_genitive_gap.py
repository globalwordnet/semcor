"""Remove a spurious gap before a fused `'s_...` run (fixes #57).

This corpus fuses multi-word proper nouns and idioms into a single token
by joining their words with underscores (e.g. `Fulton_County_Grand_Jury`)
-- a deliberate, otherwise-correct convention. But when a genitive or
contraction `'s` ends up fused to the word that *follows* it instead of
staying flush with the word it actually belongs to, the corpus keeps a
stray space or underscore immediately before the `'s`, e.g. `Al 's_Little_
Cafe` for Brown's `Al's Little Cafe`, or (entirely inside one token)
`Fulton_Tax_Commissioner_'s_Office` for Brown's `Fulton Tax Commissioner's
Office`. `'s` is never preceded by whitespace in real English, so this is
purely structural: wherever a sentence's `text` contains `'s_` preceded by
a literal space or underscore, that one character is always wrong and
should be deleted.

Detection needs no Brown/NLTK reference to decide *that* a fix applies (no
English text ever has a space before `'s`), though every candidate found
this way was independently confirmed against `src/semcor/brown-nolines.txt`
when the manifest was built. This is unrelated to whether the fused run is
its own separate token (the gap sits *between* two tokens, as with
`semcor-fix-hyphen-dropped-word-pair`) or sits in the middle of one larger
token (the gap is internal to a single span, as with
`Fulton_Tax_Commissioner_'s_Office` above) -- deleting one character and
shifting every later offset left by one, exactly like
`semcor-fix-doubled-n-contraction`, handles both uniformly since token
spans are just integer offsets into `text`.

`src/semcor/genitive-gap-fixes.yaml` lists all 31 confirmed `{file,
sentence, pos}` fixes (`pos` is the character offset, in the sentence's
current `text`, of the space/underscore to delete) -- generated once,
offline, by a scan for `[ _]'s_` verified against `brown-nolines.txt`; this
script has no NLTK dependency and just applies that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "genitive-gap-fixes.yaml"

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
    """Apply a sentence's genitive-gap fixes.

    Returns (new_text, new_tokens), or None if every fix was already
    applied (e.g. a second run).
    """
    applied = False
    for pos in sorted(positions):
        if text[pos] in (" ", "_") and text[pos + 1 : pos + 3] == "'s":
            text = text[:pos] + text[pos + 1 :]
            tokens = [
                [s - 1 if s > pos else s, e - 1 if e > pos else e]
                for s, e in tokens
            ]
            applied = True
            continue
        if text[pos : pos + 2] == "'s":
            continue  # already fixed -- no-op
        raise RuntimeError(
            f"manifest entry (pos={pos}) doesn't match current text "
            f"-- refusing to edit"
        )

    if not applied:
        return None
    return text, tokens


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def _render_tokens(tokens: list[list[int]]) -> str:
    tokens_yaml = yaml.safe_dump(tokens, default_flow_style=True, allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    tokens: {tokens_yaml}"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested genitive-gap corrections.

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
        description="Remove a spurious gap before a fused 's_... run "
        "in data/*.yaml (fixes #57), per genitive-gap-fixes.yaml."
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
