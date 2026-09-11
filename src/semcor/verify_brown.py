"""Verify data/*.yaml's `text` against the original Brown Corpus (via NLTK).

For each `data/<genre>/br-<letter><nn>.yaml`, this loads the matching
Brown Corpus file through `nltk.corpus.brown` (fileid `c<letter><nn>`,
e.g. `br-a01.yaml` <-> `ca01`) and diffs it against the file's merged
`text`, ignoring whitespace and underscores on both sides -- per #5, the
corpus's own multiword-collocation joins (`take_place`) and Brown's own
tokenization spacing aren't divergences worth reporting here; a
character actually being added, removed, or changed is.

Whitespace-only divergences (extra/missing spaces around quotes,
colons, abbreviation periods, etc.) are deliberately invisible to this
comparison -- see #8/#9 for those, which this script can't and
shouldn't re-detect since it discards whitespace entirely before
comparing.

Emits a Markdown report and exits non-zero if any file diverges, so
this can run as a CI check (see #12). It will currently fail: closing
it out is blocked on the underlying divergences (#8-#11) being fixed,
not on this script.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import nltk
import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

_FILEID_RE = re.compile(r"br-([a-z])(\d+)$")

_CONTEXT_WORDS = 6


def brown_fileid_for(path: Path) -> str | None:
    """Map `data/<genre>/br-a01.yaml` to NLTK's Brown fileid `ca01`."""
    m = _FILEID_RE.match(path.stem)
    if not m:
        return None
    letter, num = m.groups()
    return f"c{letter}{num}"


def ensure_brown_downloaded() -> None:
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)


_PUNCT_ONLY = re.compile(r"[^\w\s]+")


def _collapse_boundary_duplicates(words: list[str]) -> list[str]:
    """Brown Corpus's own convention doubles a sentence's final `;`/`?`/`!`
    (never `.`) when the next sentence follows without a paragraph break in
    between -- a sentence-boundary marker baked into the source files, not
    a divergence from SemCor (which, like any normal detokenization, keeps
    only one). Collapse those before comparing so the diff isn't swamped
    with an artifact of the reference corpus's own format.
    """
    result: list[str] = []
    for w in words:
        if result and result[-1] == w and _PUNCT_ONLY.fullmatch(w):
            continue
        result.append(w)
    return result


_OPEN_QUOTE = "``"
_CLOSE_QUOTE = "''"


def _normalize_quotes(words: list[str]) -> list[str]:
    """Collapse NLTK's tokenized `` ` ` ``/`''` quote convention to a plain
    `"`, matching this corpus's own rendering.

    That convention belongs to NLTK's own tokenization of Brown, not the
    original Brown Corpus source text -- a more faithful plaintext
    rendering of Brown (e.g. the bley-vroman/brown_nolines.txt
    reformatting) already uses a plain `"` for both open and close, same
    as this corpus. Without this, quote marks dominate every report (see
    #14): ~11,780 of the ~21,000 divergences found before this
    normalization were nothing but this convention mismatch.
    """
    return ['"' if w in (_OPEN_QUOTE, _CLOSE_QUOTE) else w for w in words]


def load_semcor_doc(path: Path) -> tuple[str, list[tuple[str, int, int]]]:
    """Return a file's merged `text` plus (sentence_id, start, end) spans
    into it, in file order -- mirroring how merge.py concatenates
    per-sentence `text` into one document.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    parts: list[str] = []
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for sent_id, sent in data.items():
        if sent_id == "_meta" or not isinstance(sent, dict):
            continue
        text = sent.get("text") or ""
        parts.append(text)
        spans.append((sent_id, offset, offset + len(text)))
        offset += len(text)
    return "".join(parts), spans


def _strip_norm(text: str) -> tuple[str, list[int]]:
    """Strip whitespace/underscores, keeping a map back to original offsets."""
    chars: list[str] = []
    orig_offsets: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace() or ch == "_":
            continue
        chars.append(ch)
        orig_offsets.append(i)
    return "".join(chars), orig_offsets


def _word_norm(words: list[str]) -> tuple[str, list[int]]:
    """Concatenate word tokens (already whitespace-free individually),
    keeping a map back to word indices."""
    chars: list[str] = []
    word_of_char: list[int] = []
    for wi, w in enumerate(words):
        for ch in w:
            chars.append(ch)
            word_of_char.append(wi)
    return "".join(chars), word_of_char


def _sentence_for_offset(spans: list[tuple[str, int, int]], offset: int) -> str:
    for sent_id, start, end in spans:
        if start <= offset < end or (offset == end and offset == start):
            return sent_id
    return spans[-1][0] if spans else "?"


class Divergence:
    def __init__(
        self,
        sentence_id: str,
        brown_context: str,
        semcor_context: str,
        pattern: tuple[str, str],
    ) -> None:
        self.sentence_id = sentence_id
        self.brown_context = brown_context
        self.semcor_context = semcor_context
        # The raw (brown, semcor) substitution itself, stripped of
        # surrounding context -- e.g. ("``", '"') for every instance of the
        # curly-vs-straight opening quote convention. Divergences sharing a
        # pattern are usually all instances of the same systematic (already
        # understood) difference, so the report aggregates by this rather
        # than listing each one -- see PATTERN_THRESHOLD below.
        self.pattern = pattern


def diff_file(path: Path, words: list[str]) -> list[Divergence]:
    doc_text, spans = load_semcor_doc(path)

    semcor_norm, semcor_offsets = _strip_norm(doc_text)
    brown_norm, brown_word_idx = _word_norm(words)

    divergences: list[Divergence] = []
    matcher = difflib.SequenceMatcher(None, brown_norm, semcor_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        # Context on the Brown side, in words.
        if i1 < i2:
            w_start, w_end = brown_word_idx[i1], brown_word_idx[i2 - 1]
        elif i1 < len(brown_word_idx):
            w_start = w_end = brown_word_idx[i1]
        elif brown_word_idx:
            w_start = w_end = brown_word_idx[-1]
        else:
            w_start = w_end = 0
        lo = max(0, w_start - _CONTEXT_WORDS)
        hi = min(len(words), w_end + 1 + _CONTEXT_WORDS)
        brown_context = " ".join(words[lo:hi])

        # Context on the SemCor side, from the original (unstripped) text.
        if j1 < j2:
            o_start, o_end = semcor_offsets[j1], semcor_offsets[j2 - 1] + 1
        elif j1 < len(semcor_offsets):
            o_start = o_end = semcor_offsets[j1]
        elif semcor_offsets:
            o_start = o_end = semcor_offsets[-1] + 1
        else:
            o_start = o_end = 0
        ctx_lo = max(0, o_start - 40)
        ctx_hi = min(len(doc_text), o_end + 40)
        semcor_context = doc_text[ctx_lo:ctx_hi].replace("\n", " ")

        sentence_id = _sentence_for_offset(spans, o_start)
        pattern = (brown_norm[i1:i2], semcor_norm[j1:j2])
        divergences.append(
            Divergence(sentence_id, brown_context, semcor_context, pattern)
        )

    return divergences


# A (brown, semcor) substitution pattern occurring at least this many times
# across the whole corpus is treated as a systematic, already-understood
# difference (e.g. curly vs. straight quotes) rather than something worth
# listing instance-by-instance: it's rolled up into one aggregate entry
# instead, so per-file listings stay focused on less-common divergences.
PATTERN_THRESHOLD = 30


def build_report(
    results: dict[Path, list[Divergence]], max_examples: int
) -> str:
    lines = [
        "# Brown Corpus verification report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Compares each `data/<genre>/br-*.yaml`'s merged `text` against the "
        "matching file in NLTK's Brown Corpus, ignoring whitespace and "
        "underscores on both sides (see #5, #12).",
        "",
    ]

    total_files = len(results)
    diverging = {p: d for p, d in results.items() if d}
    total_divergences = sum(len(d) for d in diverging.values())

    lines.append(
        f"**{len(diverging)} / {total_files} files diverge "
        f"({total_divergences} divergence(s) total).**"
    )
    lines.append("")

    if not diverging:
        lines.append("No divergences found.")
        return "\n".join(lines) + "\n"

    pattern_counts: dict[tuple[str, str], int] = {}
    pattern_example: dict[tuple[str, str], tuple[Path, Divergence]] = {}
    for path, divs in diverging.items():
        for d in divs:
            pattern_counts[d.pattern] = pattern_counts.get(d.pattern, 0) + 1
            pattern_example.setdefault(d.pattern, (path, d))

    common_patterns = {
        p for p, count in pattern_counts.items() if count >= PATTERN_THRESHOLD
    }

    lines.append("| File | Divergences |")
    lines.append("|---|---|")
    for path in sorted(diverging):
        lines.append(f"| `{path}` | {len(diverging[path])} |")
    lines.append("")

    if common_patterns:
        lines.append(
            f"## Common patterns (≥{PATTERN_THRESHOLD} occurrences)"
        )
        lines.append("")
        lines.append(
            "These account for most of the divergence count above but are "
            "each a single systematic difference, so they're rolled up "
            "here instead of listed per occurrence below."
        )
        lines.append("")
        lines.append("| Count | Brown | SemCor | Example |")
        lines.append("|---|---|---|---|")
        for pattern in sorted(
            common_patterns, key=lambda p: pattern_counts[p], reverse=True
        ):
            brown_raw, semcor_raw = pattern
            count = pattern_counts[pattern]
            ex_path, ex_div = pattern_example[pattern]
            lines.append(
                f"| {count} | `{brown_raw!r}` | `{semcor_raw!r}` | "
                f"`{ex_path}` sentence `{ex_div.sentence_id}` |"
            )
        lines.append("")

    for path in sorted(diverging):
        divs = [d for d in diverging[path] if d.pattern not in common_patterns]
        if not divs:
            continue
        lines.append(f"## `{path}`")
        lines.append("")
        for d in divs[:max_examples]:
            lines.append(f"- sentence `{d.sentence_id}`:")
            lines.append(f"  - Brown: `{d.brown_context}`")
            lines.append(f"  - SemCor: `{d.semcor_context}`")
        if len(divs) > max_examples:
            lines.append(f"- ... and {len(divs) - max_examples} more")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify data/*.yaml's text against NLTK's Brown Corpus "
        "and produce a Markdown divergence report."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to check (default: data/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the Markdown report here instead of stdout",
    )
    parser.add_argument(
        "--max-examples-per-file",
        type=int,
        default=20,
        help="Cap the number of example divergences listed per file "
        "(default: %(default)s); the summary table always counts all of them",
    )
    args = parser.parse_args()

    if args.paths:
        files = []
        for p in args.paths:
            files.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])
    else:
        files = find_yaml_files(DATA_DIR)

    try:
        cwd = Path.cwd()
        files = [
            p.relative_to(cwd) if p.is_absolute() else p for p in files
        ]
    except ValueError:
        pass  # files aren't under cwd (e.g. run from elsewhere) -- keep absolute

    ensure_brown_downloaded()
    from nltk.corpus import brown

    results: dict[Path, list[Divergence]] = {}
    skipped: list[Path] = []
    for path in files:
        fileid = brown_fileid_for(path)
        if fileid is None or fileid not in brown.fileids():
            skipped.append(path)
            continue
        words = _normalize_quotes(_collapse_boundary_duplicates(list(brown.words(fileids=[fileid]))))
        results[path] = diff_file(path, words)

    report = build_report(results, args.max_examples_per_file)
    if skipped:
        report += (
            f"\n{len(skipped)} file(s) had no matching Brown fileid and were "
            "skipped: " + ", ".join(str(p) for p in skipped) + "\n"
        )

    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)

    total_divergences = sum(len(d) for d in results.values())
    return 1 if total_divergences else 0


if __name__ == "__main__":
    sys.exit(main())
