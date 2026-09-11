"""Move an opening single quote flush against the word it introduces,
not the word before it (fixes #67).

When a single quote marks emphasized/scare-quoted or dialect-elided
speech (`more 'pro' letters than 'con'`, `Ah says 'Uh coahse...`), the
*opening* quote is frequently flush against the *preceding* word instead
of the word it actually introduces (`more' pro'` instead of
`more 'pro'`) -- the same shape as #8's double-quote gap bug, but for
single quotes, and specifically the opening side (the closing side,
flush against the word it closes, is already correct: `pro'` is right).

Detection: a standalone `'` token tagged `POS` (Brown's own tagger's
closest approximation for a bare apostrophe -- `''` closing-quote and
`POS` possessive are the two tags it can get) that's flush before and
gapped after. Most `POS`-tagged candidates are correctly-placed plural
possessives (`boys' toys`) with no scare-quote at all; verified against
`src/semcor/brown-nolines.txt` via context-window matching, requiring
the reference to confirm a literal quote character at the matched
position, which filters those out. 45 confirmed this way (~75 more
left unresolved, no unique context match, for a follow-up look).

Since the gap is always exactly one space, the fix is a same-length
swap of the quote and the space immediately after it: `text[s:s+2]`
(`"' "`) becomes `" '"`. Only this one token's own span shifts (from
`[s, s+1)` to `[s+1, s+2)`); every other token, including the word the
quote now introduces, keeps its existing span untouched. `lemmas`/`pos`/
every sense-key layer are completely unaffected.

`src/semcor/single-quote-gap-fixes.yaml` lists all 45 confirmed `{file,
sentence, index}` fixes (`index` is the quote token's own index) --
generated once, offline, against `brown-nolines.txt`, this script has no
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

MANIFEST_PATH = Path(__file__).resolve().parent / "single-quote-gap-fixes.yaml"

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
    """Apply a sentence's single-quote-gap fixes.

    Returns (new_text, new_tokens), or None if every fix was already
    applied (e.g. a second run).
    """
    applied = False
    tokens = [list(t) for t in tokens]
    for index in indices:
        s, e = tokens[index]
        if text[s] != "'":
            raise RuntimeError(
                f"manifest entry (index={index}) doesn't match current text "
                f"-- refusing to edit"
            )
        if text[s : s + 2] != "' ":
            continue  # already fixed -- no-op
        text = text[:s] + " '" + text[s + 2 :]
        tokens[index] = [s + 1, e + 1]
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
    """Apply this file's manifested single-quote-gap corrections.

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
        description="Move an opening single quote flush against the word "
        "it introduces in data/*.yaml (fixes #67), per "
        "single-quote-gap-fixes.yaml."
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
