"""Split the remaining `semcor-compare-brown-nolines` divergences into two
human-review queues instead of one undifferentiated diff.

Once the systematic, always-safe cases are handled by
`compare_brown_nolines.decode_reference_text` (typesetting escapes) and by
targeted `fix_*` scripts (real corpus bugs), what's left needs a person to
look at each case and decide -- but not all of it needs the *same* kind of
decision, so this splits it two ways:

- Brown has words this corpus doesn't have anywhere nearby, at all (a
  dropped subheadline like `#MERGER PROPOSED#`, or a swallowed prefix like
  `multi-` in `multi-million-dollar`) -- real content this corpus is
  missing and that needs new token/lemma/pos/sense annotation to add, not
  just a text edit. Written to `brown-nolines-missing.csv`.
- Every other divergence -- same content, different surface form
  (`"amount to` vs. `amount "to`, `Red` vs. `(Red)`), or content only this
  corpus's side has -- is a smaller judgment call: accept Brown's reading,
  keep this corpus's own, or flag it as a `fix_*` candidate. Written to
  `brown-nolines-review.csv` with a blank `accept` column for a human to
  fill in (`yes`/`no`/a note).

Classification uses the same per-document word alignment
`compare_brown_nolines.main` does (`difflib.SequenceMatcher` over
`our_doc_words_with_sents` vs. `_decode_and_split` + `_adopt_our_casing`),
not the unified diff text: opcode boundaries carry exactly the (sentence
ID, word-index) context a CSV row needs, which re-parsing `.diff` hunks
would have to reconstruct less reliably.

An `insert` opcode (Brown has words at this position this corpus has
none of at all) always goes to the missing-content queue. A `replace`
opcode is routed there too when Brown's side, with all whitespace/hyphens
stripped, contains this corpus's side as a substring and is strictly
longer -- the `multi-million-dollar` shape: real content embedded in a
reformatted word, not just a reformatting of the same content. Every
other opcode (`replace` that isn't a strict superset, and `delete` --
this corpus has content Brown's side doesn't) goes to the smaller-review
queue.

A third, much smaller output -- `brown-nolines-large-mismatches.csv` --
catches the opcodes wide enough (`_LARGE_OPCODE_WORDS`) that they're
almost certainly not a real content divergence at all, but a whole
document failing to align with its reference span (a bad
`brown-nolines-offsets.yaml` entry). Left for separate investigation
rather than dumped into the review queue as an unreadable thousand-word
CSV cell.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
from pathlib import Path

from semcor.compare_brown_nolines import (
    _adopt_our_casing,
    _decode_and_split,
    _DEFAULT_NOLINES_FILE,
    _DEFAULT_OFFSETS_FILE,
    brown_fileid_for,
    load_offsets,
    our_doc_words_with_sents,
)
from semcor.validate import DATA_DIR, find_yaml_files

_CONTEXT = 5

# An opcode this wide almost never means "50+ words of real content
# differ" -- every case found while building this script instead traced
# back to a whole document's word list failing to align with its
# reference span at all (likely a bad brown-nolines-offsets.yaml entry
# for that file, a separate, file-level problem `--regenerate-offsets`
# would need to re-derive). Routed to `large_mismatch` instead of the
# per-word review queue so a handful of these don't bury it in
# thousand-word CSV cells.
_LARGE_OPCODE_WORDS = 50


def _alnum(word: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", word).lower()


def _is_content_expansion(minus_words: list[str], plus_words: list[str]) -> bool:
    minus_chars = "".join(_alnum(w) for w in minus_words)
    plus_chars = "".join(_alnum(w) for w in plus_words)
    return bool(minus_chars) and len(plus_chars) > len(minus_chars) and minus_chars in plus_chars


def _context(words: list[str], start: int, end: int, before: bool) -> str:
    if before:
        return " ".join(words[max(0, start - _CONTEXT) : start])
    return " ".join(words[end : end + _CONTEXT])


def extract_doc(
    path: Path,
    ours_words: list[str],
    our_sent_ids: list[str],
    ref_words: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (missing_rows, review_rows, large_mismatch_rows) for one
    document."""
    missing: list[dict] = []
    review: list[dict] = []
    large: list[dict] = []

    matcher = difflib.SequenceMatcher(a=ours_words, b=ref_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        minus_words = ours_words[i1:i2]
        plus_words = ref_words[j1:j2]

        if max(len(minus_words), len(plus_words)) > _LARGE_OPCODE_WORDS:
            large.append(
                {
                    "file": path.stem,
                    "sent_id": our_sent_ids[i1] if i1 < len(our_sent_ids) else "",
                    "type": tag,
                    "ours_word_count": len(minus_words),
                    "reference_word_count": len(plus_words),
                }
            )
            continue
        # An opcode's own left edge (i1) is the anchor sentence: for an
        # insert, i1==i2 so it's the position words would be inserted at;
        # for replace/delete it's the first word actually being replaced.
        anchor = min(i1, len(our_sent_ids) - 1) if our_sent_ids else None
        sent_id = our_sent_ids[anchor] if anchor is not None and anchor >= 0 else ""

        row = {
            "file": path.stem,
            "sent_id": sent_id,
            "before_context": _context(ours_words, i1, i1, before=True),
            "ours": " ".join(minus_words),
            "reference": " ".join(plus_words),
            "after_context": _context(ours_words, i2, i2, before=False),
        }

        if tag == "insert" or (tag == "replace" and _is_content_expansion(minus_words, plus_words)):
            missing.append(row)
        else:
            review_row = dict(row)
            review_row["type"] = tag
            review_row["accept"] = ""
            review.append(review_row)

    return missing, review, large


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split semcor-compare-brown-nolines divergences into "
        "brown-nolines-missing.csv (content this corpus lacks entirely) "
        "and brown-nolines-review.csv (smaller judgment calls, for a "
        "human to mark accept=yes/no)."
    )
    parser.add_argument("data_dir", nargs="?", type=Path, default=DATA_DIR)
    parser.add_argument("--nolines-file", type=Path, default=_DEFAULT_NOLINES_FILE)
    parser.add_argument("--offsets-file", type=Path, default=_DEFAULT_OFFSETS_FILE)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    if not args.nolines_file.exists():
        print(f"error: {args.nolines_file} not found.", file=sys.stderr)
        return 2
    if not args.offsets_file.exists():
        print(f"error: {args.offsets_file} not found.", file=sys.stderr)
        return 2

    with args.nolines_file.open("r", encoding="utf-8") as f:
        nolines_text = f.read()
    offsets = load_offsets(args.offsets_file)

    files = [p for p in find_yaml_files(args.data_dir) if brown_fileid_for(p)]
    files.sort(key=lambda p: brown_fileid_for(p))

    all_missing: list[dict] = []
    all_review: list[dict] = []
    all_large: list[dict] = []

    for path in files:
        fileid = brown_fileid_for(path)
        assert fileid is not None
        if fileid not in offsets:
            continue
        start, end = offsets[fileid]

        ours_words, our_sent_ids = our_doc_words_with_sents(path)
        ref_words, brace_flags = _decode_and_split(nolines_text, start, end)
        _adopt_our_casing(ours_words, ref_words, brace_flags)

        missing, review, large = extract_doc(path, ours_words, our_sent_ids, ref_words)
        all_missing.extend(missing)
        all_review.extend(review)
        all_large.extend(large)

    missing_path = args.out_dir / "brown-nolines-missing.csv"
    review_path = args.out_dir / "brown-nolines-review.csv"
    large_path = args.out_dir / "brown-nolines-large-mismatches.csv"

    with missing_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "sent_id", "before_context", "ours", "reference", "after_context"]
        )
        writer.writeheader()
        writer.writerows(all_missing)

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "sent_id",
                "type",
                "before_context",
                "ours",
                "reference",
                "after_context",
                "accept",
            ],
        )
        writer.writeheader()
        writer.writerows(all_review)

    with large_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "sent_id", "type", "ours_word_count", "reference_word_count"]
        )
        writer.writeheader()
        writer.writerows(all_large)

    print(f"{len(all_missing)} missing-content row(s) -> {missing_path}")
    print(f"{len(all_review)} review row(s) -> {review_path}")
    print(
        f"{len(all_large)} large-mismatch opcode(s) (>{_LARGE_OPCODE_WORDS} words, likely a bad "
        f"brown-nolines-offsets.yaml entry, not a real per-word divergence) -> {large_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
