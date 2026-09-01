"""Restore em dashes (and a handful of plain number-range hyphens)
corrupted into a single, space-padded `-` (fixes #9).

Brown tokenizes an em dash as its own two-character token `--`, sitting
flush against its neighbours with no surrounding space. This corpus
reduced every one of those to a single `-` padded with a space on each
side -- wrong character *and* two spurious spaces.

Naively, "every token whose surface is exactly `-`, surrounded by
spaces" sounds like the whole fix. It isn't: cross-checking all 2,939
such tokens against `nltk.corpus.brown` (same method as #13 -- locate
the token's local context in Brown, confirm what's actually there)
found three different underlying patterns, not one:

- 1,868 are genuine em dashes: Brown has `--` at that position. Fixed
  here.
- 17 are number ranges (`10-16`, `100-230`, `1960-1962`, ...) where
  Brown keeps a plain `-` as its own token, not doubled. Also fixed
  here, using the confirmed single-`-` form.
- ~1,054 are a different bug entirely: a compound modifier that's a
  *single* token in Brown (`10-16 Mc` is one thing; the actual example
  found was `The 80-hp motor`, one token `80-hp`) got split into three
  SemCor tokens (`80`, `-`, `hp`) with spurious spaces added, the same
  way `11:30` got split into three in #8. Fixing that means merging
  tokens back together, not just editing whitespace/characters, which
  is a different shape of fix -- left for a separate issue rather than
  folded in here.

The remaining ~unconfirmed instances (context didn't match Brown
uniquely, usually because of a multiword-joined neighbour like
`pointed_out`) are left untouched rather than guessed at.

`em-dash-fixes.yaml` lists exactly the 1,885 confirmed (file, sentence,
index, replacement) fixes to apply -- generated once, offline, by the
verification process above; this script has no runtime dependency on
NLTK/Brown, it just applies that fixed manifest. Each fix expands the
token's own span (`-` -> `--`, or leaves it `-`) and closes the
surrounding gaps in one move, updating `tokens` offsets for everything
after it in the sentence and updating that token's own `lemmas` entry
to match (kept consistent with `text`, as it already was before this
ran -- every other punctuation token's lemma is a literal copy of its
surface, same as this one's was when it was still a lone `-`).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "em-dash-fixes.yaml"

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

    Each fix replaces the region from the previous token's end (or this
    token's own start, if there's no gap to close) to the next token's
    start (or this token's own end) with `replacement`, absorbing the
    spurious padding either side along with the character itself.
    """
    regions = []  # (left, right, replacement, token_index), sorted, non-overlapping
    for index, replacement in sorted(fixes):
        s, e = tokens[index]
        left = tokens[index - 1][1] if index > 0 and tokens[index - 1][1] < s else s
        right = tokens[index + 1][0] if index + 1 < len(tokens) and tokens[index + 1][0] > e else e
        regions.append((left, right, replacement, index))

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
        description="Restore em dashes (and confirmed number-range "
        "hyphens) corrupted into a space-padded single '-' in "
        "data/*.yaml's `text` (fixes #9), per em-dash-fixes.yaml."
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
