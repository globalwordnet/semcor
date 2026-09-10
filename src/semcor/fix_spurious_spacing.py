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

So a sentence's quotes are only fixed by *guessing* structure when it
contains exactly two `"` tokens: an unambiguous, self-contained pair,
with no cross-sentence state and no nesting question to get wrong.
That covers 3,686 of the corpus's 6,613 quote-containing sentences.

The rest don't need a structural guess at all, though: instead of
inferring open/close direction, `quote-gap-fixes.yaml` verifies each
quote's *actual* spacing directly against `brown-nolines.txt` -- the
same reference `semcor-compare-brown-nolines` already trusts. Since
that file also collapsed both quote directions to a bare `"` (same
loss, per #14), it can't disambiguate open-vs-close either -- but it
*does* preserve real spacing, which is the only thing this fix needs:
building a word-context window around each quote (reusing this
corpus's own whitespace/underscore conventions) and requiring a
*unique* matching position in the reference confirms, per quote and
per side independently, whether that specific gap should close --
without ever needing to know whether the sentence's quotes are nested,
sequential pairs, or one continuing from/into another sentence. 2,750
such gaps (1,362 before, 1,388 after) were confirmed this way and are
listed in the manifest; the rest (no unique context match, or the
reference confirms a real gap belongs there too) are left untouched.

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
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "quote-gap-fixes.yaml"

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


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, str]]]]:
    """Return {filename: {sentence_id: [(index, side), ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["index"], entry["side"]))
    return by_file


def find_gaps(sent: dict, manifest_fixes: list[tuple[int, str]] | None = None) -> list[tuple[int, int]]:
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
    # for why guessing structurally isn't safe for a lone quote or more
    # than two. `manifest_fixes` (verified against Brown, not guessed)
    # covers those cases instead, just below.
    if len(quote_indices) == 2:
        open_i, close_i = quote_indices
        g = _gap_after(tokens, open_i)
        if g:
            gaps.add(g)
        g = _gap_before(tokens, close_i)
        if g:
            gaps.add(g)

    for index, side in manifest_fixes or []:
        if index >= len(tokens) or text[tokens[index][0]:tokens[index][1]] != '"':
            continue  # stale manifest entry -- skip rather than corrupt
        g = _gap_before(tokens, index) if side == "before" else _gap_after(tokens, index)
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


def fix_file(
    path: Path, manifest: dict, dry_run: bool = False
) -> list[tuple[str, int]]:
    """Close spurious gaps in `path`.

    Returns a list of (sentence_id, gaps_closed) for every sentence
    changed (or that would change, if `dry_run`).
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    file_fixes = manifest.get(path.name, {})
    affected: dict[str, tuple[str, list[list[int]]]] = {}
    counts: list[tuple[str, int]] = []
    for doc_id, doc in data.items():
        if doc_id == "_meta" or not isinstance(doc, dict):
            continue
        gaps = find_gaps(doc, file_fixes.get(doc_id))
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
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Reference-verified quote-gap manifest to read (default: %(default)s)",
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

    total_gaps = 0
    total_sentences = 0
    changed_files = 0
    for path in files:
        counts = fix_file(path, manifest, dry_run=args.dry_run)
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
