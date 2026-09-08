"""Close spurious whitespace gaps in `text` that don't exist in the real
Brown Corpus text (fixes #8).

Three specific, narrow patterns, each verified against
`browntag_nolines.txt`/`nltk.corpus.brown` in #8:

- a space inserted just inside an opening/closing quote
- a space inserted around a `:` between two all-digit tokens (`11: 30`)
- a space inserted between two short letter(s)+period fragments that are
  actually one abbreviation split across tokens (`a. m.`, `i. e.`)

Quote direction is *not* taken from `pos` (`` ` ` `` vs `''`): per #14,
this corpus already collapsed both into a single `"` character, so
there's nothing left in the token's own text to tell open from close,
and spot-checking found the `pos` tag itself isn't reliable either (two
quotes in the same pair are sometimes both tagged `''`).

The obvious alternative -- toggle open/close across every `"` in a
document, in sentence order -- turns out to be unsafe and was tried and
reverted: real Brown prose has both (a) sentences with a stray,
genuinely never-closed quote (1961 newspaper typesetting slips), which
flips the toggle for every quote in the rest of the document once hit,
confirmed to silently turn correct fixes backwards from that point on;
and (b) quotes nested inside a quote using the same `"` glyph (a quoted
term inside a longer quotation), where simple alternation gets the
inner pair backwards regardless -- also confirmed on a real sentence.
Neither is reliably distinguishable from the ordinary case by any
signal available here.

So this only fixes a sentence's quotes when it contains *exactly two*
`"` tokens: an unambiguous, self-contained open/close pair, with no
cross-sentence state and no nesting question to get wrong. That's
3,686 of the corpus's 6,613 quote-containing sentences; the rest (a
lone quote continuing from/into another sentence, or more than two in
one sentence, which could be sequential pairs or nesting) are left for
a follow-up pass -- see #8's tracking issue for whether that's worth
building out further.

Each fix is a whitespace-only edit: the character content of every
token is unchanged, only the gap *between* certain adjacent token pairs
is removed, and every `tokens` offset at or after a closed gap shifts
left to match. `semcor-check-tokens` (a sample of token text keyed by
sentence + index, not absolute offset) is unaffected by this by design.

Edits are targeted substitutions on each document's raw text block, not
a YAML parse/dump round-trip for the whole file (see
fix_leading_space.py for why -- varied quoting/line-wrapping a generic
dumper won't reproduce). Only sentences with at least one gap to close
have their `text`/`tokens` lines regenerated; everything else in the
file, including unaffected sentences, is left byte-for-byte untouched.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

# A top-level document (or `_meta`) key: unindented, ending the line.
_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
# `text`'s value, possibly wrapped across several lines -- non-greedy and
# stopping at the lookahead so DOTALL can't run past it. `tokens` (always
# the next key, and always a single line) is matched separately, and
# *without* DOTALL, so its own `.*` can't cross a newline either -- both
# matter: an earlier version combined them into one DOTALL pattern, whose
# greedy `.*` for tokens' value ran straight through the following
# wn16_key/wn30_key/oewn_key lines to the *last* `]` in the chunk,
# silently deleting them.
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")

_ABBREV_FRAGMENT = re.compile(r"^[A-Za-z]{1,4}\.$")

# Emitting a single unwrapped line sidesteps needing to reproduce the
# original dumper's line-folding exactly; PyYAML round-trips it fine
# either way, and it keeps the diff for an edited sentence readable
# (one changed line, not a whole reflowed paragraph).
_WIDTH = 10**9


def _gap_after(tokens: list[list[int]], i: int) -> tuple[int, int] | None:
    if i + 1 >= len(tokens):
        return None
    end, start = tokens[i][1], tokens[i + 1][0]
    return (end, start) if start > end else None


def _gap_before(tokens: list[list[int]], i: int) -> tuple[int, int] | None:
    if i - 1 < 0:
        return None
    end, start = tokens[i - 1][1], tokens[i][0]
    return (end, start) if start > end else None


def find_gaps(sent: dict) -> list[tuple[int, int]]:
    """Return the (start, end) character ranges in `sent['text']` that
    should be deleted -- each one a gap between two adjacent tokens that
    shouldn't have any whitespace between them.
    """
    text = sent.get("text") or ""
    tokens = sent.get("tokens") or []
    gaps: set[tuple[int, int]] = set()

    surfaces = [text[s:e] for s, e in tokens]
    quote_indices = [i for i, surf in enumerate(surfaces) if surf == '"']

    # Only an unambiguous, self-contained pair -- see module docstring
    # for why a lone quote (continues from/into another sentence) or
    # more than two (sequential pairs vs. nesting, indistinguishable
    # here) aren't safe to guess at.
    if len(quote_indices) == 2:
        open_i, close_i = quote_indices
        g = _gap_after(tokens, open_i)
        if g:
            gaps.add(g)
        g = _gap_before(tokens, close_i)
        if g:
            gaps.add(g)

    for i, surf in enumerate(surfaces):
        if surf == ":":
            prev_surf = surfaces[i - 1] if i > 0 else ""
            next_surf = surfaces[i + 1] if i + 1 < len(surfaces) else ""
            if prev_surf.isdigit() and next_surf.isdigit():
                gb = _gap_before(tokens, i)
                if gb:
                    gaps.add(gb)
                ga = _gap_after(tokens, i)
                if ga:
                    gaps.add(ga)

        if _ABBREV_FRAGMENT.match(surf) and i + 1 < len(surfaces):
            if _ABBREV_FRAGMENT.match(surfaces[i + 1]):
                g = _gap_after(tokens, i)
                if g:
                    gaps.add(g)

    return sorted(gaps)


def apply_gaps(
    text: str, tokens: list[list[int]], gaps: list[tuple[int, int]]
) -> tuple[str, list[list[int]]]:
    if not gaps:
        return text, tokens

    parts = []
    cursor = 0
    for start, end in gaps:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    new_text = "".join(parts)

    def shift(offset: int) -> int:
        removed = 0
        for start, end in gaps:
            if start < offset:
                removed += min(end, offset) - start
            else:
                break
        return offset - removed

    new_tokens = [[shift(s), shift(e)] for s, e in tokens]
    return new_text, new_tokens


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def _render_tokens(tokens: list[list[int]]) -> str:
    tokens_yaml = yaml.safe_dump(tokens, default_flow_style=True, allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    tokens: {tokens_yaml}"


def fix_file(path: Path, dry_run: bool = False) -> list[tuple[str, int]]:
    """Close spurious gaps in `path`.

    Returns a list of (sentence_id, gaps_closed) for every sentence
    changed (or that would change, if `dry_run`).
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    affected: dict[str, tuple[str, list[list[int]]]] = {}
    counts: list[tuple[str, int]] = []
    for doc_id, doc in data.items():
        if doc_id == "_meta" or not isinstance(doc, dict):
            continue
        gaps = find_gaps(doc)
        if not gaps:
            continue
        new_text, new_tokens = apply_gaps(doc.get("text") or "", doc.get("tokens") or [], gaps)
        affected[doc_id] = (new_text, new_tokens)
        counts.append((doc_id, len(gaps)))

    if not affected or dry_run:
        return counts

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
        if doc_id in affected:
            new_text, new_tokens = affected[doc_id]
            # Function repls, not string ones -- some sentences contain
            # literal backslashes (e.g. math notation in br-j03.yaml), and
            # a string repl would have re try to interpret `\g<...>` etc.
            # as backreferences.
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
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Close spurious whitespace gaps around quotes, "
        "digit:digit colons, and split abbreviations in data/*.yaml's "
        "`text` (fixes #8)."
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

    total_gaps = 0
    total_sentences = 0
    changed_files = 0
    for path in files:
        counts = fix_file(path, dry_run=args.dry_run)
        if counts:
            changed_files += 1
            total_sentences += len(counts)
            total_gaps += sum(n for _, n in counts)

    verb = "Would close" if args.dry_run else "Closed"
    print(
        f"{verb} {total_gaps} gap(s) across {total_sentences} sentence(s) "
        f"in {changed_files} file(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
