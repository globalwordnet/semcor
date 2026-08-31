"""Restore sentence-initial capitalization lost after a dialogue-closing
quote + `.`/`?`/`!` (fixes #13).

When a new clause starts immediately after a `"` followed by sentence-
terminal punctuation -- whether that's later in the same (merged)
SemCor sentence, or the very next one -- its first letter is lowercase
in this corpus's `text` where Brown has it capitalized. Detected as:
three tokens in a row, in document order (crossing a sentence-entry
boundary is fine -- SemCor's segmentation doesn't always match
Brown's), where the first is `"`, the second is `.`/`?`/`!`, and the
third starts with a lowercase letter.

That pattern alone isn't quite enough to trust blindly: of the 160
places it occurs across the corpus, 152 were confirmed correct by
locating the same local context in `nltk.corpus.brown` and checking
Brown's capitalization there, but one (`data/fiction_mystery/br-l12.yaml`
sentence `Rakw`) turned out to be misleading -- Brown has "His voice
shook", and this corpus is missing the word "His" entirely, not just
its capital; blindly capitalizing "voice" would still be wrong, just
differently. Seven more couldn't be confirmed either way (usually a
multiword-joined neighbor, e.g. `pointed_out`, breaking the exact-match
search against Brown's separately-tokenized words). _SKIP lists all
eight so this only touches the 152 independently confirmed instances;
see #13 for the verification method if the skipped ones are ever worth
another look.

Each fix is a single-character case change: `tokens` offsets never
shift, so unlike fix_spurious_spacing.py, only `text` needs rewriting.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)

_TERMINAL = {".", "?", "!"}

# (filename, sentence_id, token_index) confirmed NOT safe to auto-fix --
# see module docstring.
_SKIP: set[tuple[str, str, int]] = {
    ("br-l12.yaml", "Rakw", 0),  # contradicted: real issue is a missing word ("His"), not a case fix
    ("br-n11.yaml", "JFXF", 13),  # unresolved: multiword-joined neighbor breaks verification
    ("br-k08.yaml", "jI8f", 10),  # unresolved: multiword-joined neighbor breaks verification
    ("br-k11.yaml", "QW+j", 8),  # unresolved: multiword-joined neighbor breaks verification
    ("br-k25.yaml", "5iw4", 24),  # unresolved: multiword-joined neighbor breaks verification
    ("br-l16.yaml", "0CHu", 0),  # unresolved: context too short/common to verify uniquely
    ("br-l16.yaml", "it0F", 0),  # unresolved: context too short/common to verify uniquely
    ("br-r02.yaml", "Zyca", 16),  # unresolved: multiword-joined neighbor breaks verification
}


def find_fixes(data: dict, filename: str) -> dict[str, list[int]]:
    """Return {sentence_id: [char_offsets]} for each sentence needing the
    character at each offset in `text` capitalized -- a sentence can need
    more than one (e.g. `data/fiction_romance/br-p05.yaml` sentence
    `DaBb` needs two), so this must accumulate a list, not overwrite.
    """
    ids = [k for k in data if k != "_meta" and isinstance(data[k], dict)]
    fixes: dict[str, list[int]] = {}
    prev_two: list[str] = []  # last up-to-2 token surfaces seen so far

    for sid in ids:
        sent = data[sid]
        text = sent.get("text") or ""
        tokens = sent.get("tokens") or []
        surfaces = [text[s:e] for s, e in tokens]

        for i, surf in enumerate(surfaces):
            window = (prev_two + surfaces[:i])[-2:]
            if (
                len(window) == 2
                and window[0] == '"'
                and window[1] in _TERMINAL
                and surf
                and surf[0].isalpha()
                and surf[0].islower()
                and (filename, sid, i) not in _SKIP
            ):
                fixes.setdefault(sid, []).append(tokens[i][0])  # first char of this token
            prev_two = (prev_two + [surf])[-2:]

    return fixes


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=10**9).rstrip("\n")
    return f"    text: {text_yaml}"


def fix_file(path: Path, dry_run: bool = False) -> list[str]:
    """Capitalize lost sentence-initial letters in `path`.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    fixes = find_fixes(data, path.name)
    if not fixes or dry_run:
        return list(fixes)

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
        if doc_id in fixes:
            new_text = data[doc_id]["text"]
            for offset in fixes[doc_id]:
                new_text = new_text[:offset] + new_text[offset].upper() + new_text[offset + 1 :]
            replacement = _render_text(new_text)
            chunk, n = _TEXT_BLOCK.subn(lambda m: replacement, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text block for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(fixes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore sentence-initial capitalization lost after a "
        "dialogue-closing quote + ./?/! in data/*.yaml's `text` (fixes #13)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to update (default: data/)",
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

    total_fixes = 0
    changed_files = 0
    for path in files:
        fixed = fix_file(path, dry_run=args.dry_run)
        if fixed:
            changed_files += 1
            total_fixes += len(fixed)

    verb = "Would fix" if args.dry_run else "Fixed"
    print(f"{verb} {total_fixes} sentence(s) across {changed_files} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
