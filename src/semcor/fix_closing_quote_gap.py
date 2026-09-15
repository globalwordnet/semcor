"""Move a closing quote (or quote pair) flush against the word it closes,
and put a closing pair in the correct nesting order.

This is the mirror image of `semcor-fix-single-quote-gap` (#67): that
fix moves an *opening* single quote flush against the word it
introduces; a closing quote or quote-pair has the exact same "gap
landed on the wrong side" bug, just at the other end of the quoted span
(`sepulchred ".` where Brown's real text has `sepulchred".`). It also
folds in `semcor-fix-quote-order` (#72)'s ordering rule -- Brown's real
text always closes the *inner* single quote before the *outer* double
quote (`'perhaps'".`, never `'perhaps"'.` or, with the gap this fix
targets, `'perhaps "'.`) -- since a gapped closing pair is frequently
*also* mis-ordered, and fixing one without the other would still leave
a divergence.

Detection swept every sentence for a `'`/`"` token with exactly one
space before it (never the gap-then-word `"'"` `POS`-tagged shape
#67 already owns) where what follows is either nothing (end of
sentence) or closing-position punctuation (`.`, `,`, `?`, `!`, `:`,
`;`, `)`, `-`) -- immediately followed by another `'`/`"` token flush
against it, if there is one. A run preceded by a purely numeric token
is skipped outright (an inches mark like `1 "`, a separate and
unrelated bug). Every candidate that survived was verified individually
against `src/semcor/brown-nolines.txt`: reconstruct the fixed target
(gap closed, pair reordered to `'` before `"`) together with a few
words of preceding context, and require a unique match in the
reference. 77 confirmed this way.

The fix touches only the quote run's own character(s) and its own
token span(s) -- the run's *content* is replaced (`'"'` -> `'\\'"'`) but
its overall position is anchored to right after the preceding token, so
only tokens from the run onward shift, by the same one character the
gap removal saves. `lemmas` for the run's token(s) are updated to match
(a lemma here is just the token's own surface character, same as any
other punctuation token in this corpus); `pos` is set to `''` for every
token in the run, matching the closing-quote tag this corpus already
uses everywhere else a quote closes a quotation (some of these 77 had
picked up `` `` `` -- the opening-quote tag -- instead, on the token
that's actually closing something). No sense-key layer needed touching
across any of the 77 (verified: no candidate token ever carries one, on
either side of the swap).

`src/semcor/closing-quote-gap-fixes.yaml` lists all 77 confirmed {file,
sentence, index} fixes (`index` is the run's first token's own index,
against the *pre*-fix `tokens` -- since the fix never changes a
sentence's token *count*, only content and character offsets, this
stays valid to re-derive against already-fixed data too, which is how
this script's idempotency check works) -- generated once, offline,
against `brown-nolines.txt`, this script has no NLTK dependency and
just applies that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "closing-quote-gap-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_LINE = {key: re.compile(rf"(?m)^    {key}: (\[.*\])$") for key in ("tokens", "lemmas", "pos")}
_WIDTH = 10**9

_QUOTE_CHARS = ("'", '"')


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[int]]]:
    """Return {filename: {sentence_id: [index, ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append(entry["index"])
    return by_file


def _dump_str(value: str) -> str:
    return yaml.safe_dump(value, default_style='"', allow_unicode=True, width=_WIDTH).rstrip("\n")


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def _render_tokens(tokens: list[list[int]]) -> str:
    return "    tokens: [" + ", ".join(f"[{s}, {e}]" for s, e in tokens) + "]"


def _render_str_list(key: str, values: list[str]) -> str:
    return f"    {key}: [" + ", ".join(_dump_str(v) for v in values) + "]"


def fix_sentence(
    text: str, tokens: list[list[int]], pos: list[str], lemmas: list[str], indices: list[int]
) -> tuple[str, list[list[int]], list[str], list[str]] | None:
    """Apply a sentence's closing-quote-gap fixes.

    Returns (new_text, new_tokens, new_pos, new_lemmas), or None if every
    fix was already applied (e.g. a second run).
    """
    applied = False
    for i in sorted(indices, reverse=True):
        tok = text[tokens[i][0] : tokens[i][1]]
        if tok not in _QUOTE_CHARS:
            raise RuntimeError(f"manifest index {i} isn't a quote token ({tok!r}) -- refusing to edit")

        pair_end = None
        if i + 1 < len(tokens):
            s2, e2 = tokens[i + 1]
            tok2 = text[s2:e2]
            if tok2 in _QUOTE_CHARS and s2 == tokens[i][1]:
                pair_end = i + 1

        prev_end = tokens[i - 1][1]
        run_start = tokens[i][0]
        run_end = tokens[pair_end][1] if pair_end is not None else tokens[i][1]
        gap = run_start - prev_end

        if gap == 0 and pair_end is None:
            continue  # already fixed -- no-op
        if gap == 0 and pair_end is not None and text[run_start] == "'" and text[run_start + 1] == '"':
            continue  # already fixed -- no-op
        if gap not in (0, 1):
            raise RuntimeError(f"index {i}: gap of {gap} chars, expected 0 or 1 -- refusing to edit")

        new_content = "'\"" if pair_end is not None else tok
        text = text[:prev_end] + new_content + text[run_end:]
        delta = len(new_content) - (run_end - prev_end)
        end_idx = pair_end if pair_end is not None else i

        new_tokens = list(tokens)
        cur = prev_end
        for k, _ch in enumerate(new_content):
            new_tokens[i + k] = [cur, cur + 1]
            cur += 1
        for j in range(end_idx + 1, len(new_tokens)):
            s, e = new_tokens[j]
            new_tokens[j] = [s + delta, e + delta]

        pos = list(pos)
        lemmas = list(lemmas)
        if pair_end is not None:
            pos[i], pos[pair_end] = "''", "''"
            lemmas[i], lemmas[pair_end] = "'", '"'
        else:
            pos[i] = "''"
            lemmas[i] = tok

        tokens = new_tokens
        applied = True

    if not applied:
        return None
    return text, tokens, pos, lemmas


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested closing-quote-gap corrections.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, tuple[str, list[list[int]], list[str], list[str]]] = {}
    for sid, indices in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        result = fix_sentence(sent["text"], sent["tokens"], sent["pos"], sent["lemmas"], indices)
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
            new_text, new_tokens, new_pos, new_lemmas = changed[doc_id]
            chunk, n = _TEXT_BLOCK.subn(lambda m: _render_text(new_text), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text block for {doc_id!r}")
            chunk, n = _LINE["tokens"].subn(lambda m: _render_tokens(new_tokens), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find tokens line for {doc_id!r}")
            chunk, n = _LINE["pos"].subn(lambda m: _render_str_list("pos", new_pos), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find pos line for {doc_id!r}")
            chunk, n = _LINE["lemmas"].subn(lambda m: _render_str_list("lemmas", new_lemmas), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find lemmas line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move a closing quote (or quote pair) flush against "
        "the word it closes, and correct a mis-ordered closing pair, per "
        "closing-quote-gap-fixes.yaml."
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

    manifest = load_manifest()

    if args.paths:
        files = []
        for p in args.paths:
            files.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])
    else:
        files = find_yaml_files(DATA_DIR)

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
