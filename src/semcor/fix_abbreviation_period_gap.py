"""Close a spurious gap between an abbreviation period and the
sentence-final period (fixes #61).

When a sentence ends right after an abbreviation (`Jr.`, `etc.`, `D.C.`),
Brown's own transcription keeps both the abbreviation's own period and
the grammatical sentence-final period as two adjacent, flush characters
(`Jr..`) rather than eliding one -- the same convention that already
motivates `compare_brown_nolines.decode_reference_text` treating a lone
'&' as "a non-sentence-final abbreviation period" (see #9/#11/#17):
when the abbreviation's period *is* also sentence-final, Brown's own
source keeps it a literal '.', immediately followed by the sentence's
own closing '.'.

This corpus already tokenizes that second period as its own token (the
same structural choice this corpus makes everywhere else, e.g. #42's
`can`/`n't` split), but 100 confirmed instances have a stray space
between the two periods instead of being flush -- `Jr. .` where Brown's
own text has `Jr..`. Detection: token `i` ends with `.` (and is longer
than one character, so it's a real abbreviation, not the period itself),
token `i + 1` is exactly `.`, with exactly one space between them.

Verified against `src/semcor/brown-nolines.txt` with the same
context-window word search #43/#8's follow-up use (unique match
required, using both sides of the gap and pulling extra context from
neighbouring sentences when the current one runs out). This confirms
100 of 154 raw candidates; the other 54 are left alone -- mostly
multi-word underscore-joined abbreviations (`N._Y.`, `D._C.`, `U._S.`)
where Brown's real text has no space between the parts either
(`N.Y..`, not `N. Y..`), a related but distinct bug in how this corpus's
underscore-joining convention handles this class of abbreviation.

Just like `semcor-fix-genitive-gap`, this is a pure single-character
deletion between two existing tokens: token *count* never changes, only
the deleted position's own token and everything after it shift left by
one to close the gap. `lemmas`/`pos`/every sense-key layer are completely
untouched -- only `text` and the shifted `tokens` offsets change.

`src/semcor/abbreviation-period-gap-fixes.yaml` lists all 100 confirmed
`{file, sentence, pos}` fixes (`pos` is the character offset of the
space to delete) -- generated once, offline, against `brown-nolines.txt`,
this script has no NLTK dependency and just applies that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "abbreviation-period-gap-fixes.yaml"

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
    """Apply a sentence's abbreviation-period-gap fixes.

    Returns (new_text, new_tokens), or None if every fix was already
    applied (e.g. a second run).
    """
    applied = False
    for pos in sorted(positions):
        if text[pos] == " " and text[pos + 1] == ".":
            text = text[:pos] + text[pos + 1 :]
            tokens = [
                [s - 1 if s > pos else s, e - 1 if e > pos else e]
                for s, e in tokens
            ]
            applied = True
            continue
        if text[pos] == ".":
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
    body = ", ".join(f"[{s}, {e}]" for s, e in tokens)
    return f"    tokens: [{body}]"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested abbreviation-period-gap corrections.

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
        description="Close a spurious gap between an abbreviation period "
        "and the sentence-final period in data/*.yaml (fixes #61), per "
        "abbreviation-period-gap-fixes.yaml."
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
