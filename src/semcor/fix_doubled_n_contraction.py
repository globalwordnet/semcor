"""Fix negated `can`/`won't` split as `cann't`/`wonn't` (fixes #42).

When a negated `can` or `won't` splits into two tokens, this corpus keeps
the modal's whole spelling as the first token (`can`, `won`) instead of
the correct Penn-Treebank-style split (`ca`, `wo`), then starts the
second token at `n't` anyway -- duplicating the shared `n` and rendering
as `cann't`/`wonn't` (6 characters) once the two flush token spans are
concatenated, instead of Brown's real 5-character `can't`/`won't`.

Detection is purely structural, no Brown/NLTK reference needed: a token
pair `(i, i+1)` is this bug iff the spans are flush
(`tokens[i][1] == tokens[i+1][0]`), `tokens[i+1]`'s surface is exactly
`n't`, and `tokens[i]`'s surface is exactly `can` or `won`. This
naturally excludes two other, unrelated anomalies noted in #42 (a `cai`
typo in `data/fiction_adventure/br-n16.yaml`, a `could`-lemma-but-
different-surface case in `data/fiction_romance/br-p09.yaml`), since
neither has surface `can`/`won` at that position -- no manual exclusion
needed.

Token *count* never changes (no merge/split, just resizing one token's
span and shifting every later offset in the sentence left by one
character), so this is simpler than #24/#43's token-merge fixes and
doesn't need their advancing-cursor relocation: an `index` from the
manifest still points at the same token after an earlier fix in the same
sentence has run, just with shifted span *values*. `lemmas`/`pos`/every
sense-key layer are untouched -- the lemma content (`"can"`/`"n't"`, or
whatever `"win"`/`"will"` lemmatization quirk is already there for
`won't`, a separate, unrelated inconsistency out of scope here) was
already correct; only the surface character span was wrong.

`src/semcor/doubled-n-contraction-fixes.yaml` lists all 166 confirmed
`{file, sentence, index, word}` fixes (111 `can`, 55 `won`) -- generated
once, offline, by a scratch scan of this same structural rule; this
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

MANIFEST_PATH = Path(__file__).resolve().parent / "doubled-n-contraction-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")

_WIDTH = 10**9


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, str]]]]:
    """Return {filename: {sentence_id: [(index, word), ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["index"], entry["word"]))
    return by_file


def fix_sentence(
    text: str, tokens: list[list[int]], fixes: list[tuple[int, str]]
) -> tuple[str, list[list[int]]] | None:
    """Apply a sentence's (index, word) doubled-n fixes.

    Returns (new_text, new_tokens), or None if every fix was already
    applied (e.g. a second run).
    """
    applied = False
    for index, word in sorted(fixes):
        s0, e0 = tokens[index]
        s1, e1 = tokens[index + 1] if index + 1 < len(tokens) else (None, None)

        if text[s0:e0] == word[:-1] and s1 == e0 and text[s1:e1] == "n't":
            continue  # already fixed -- no-op

        if not (text[s0:e0] == word and s1 == e0 and text[s1:e1] == "n't"):
            raise RuntimeError(
                f"manifest entry (index={index}, word={word!r}) doesn't match "
                f"current tokens -- refusing to edit"
            )

        del_pos = e0 - 1  # the duplicated 'n'
        text = text[:del_pos] + text[del_pos + 1 :]
        tokens = [
            [s - 1 if s > del_pos else s, e - 1 if e > del_pos else e]
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
    tokens_yaml = yaml.safe_dump(tokens, default_flow_style=True, allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    tokens: {tokens_yaml}"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested doubled-n corrections.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, tuple[str, list[list[int]]]] = {}
    for sid, fixes in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        result = fix_sentence(sent["text"], sent["tokens"], fixes)
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
        description="Fix negated can/won't split as cann't/wonn't in "
        "data/*.yaml (fixes #42), per doubled-n-contraction-fixes.yaml."
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
