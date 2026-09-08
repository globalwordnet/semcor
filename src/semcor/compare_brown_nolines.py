"""Compare this corpus's plain text against a local copy of
http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt -- a plaintext
reformatting of the whole 500-file Brown Corpus with original line-wrapping
removed, used elsewhere in this repo (#16/#18) as a more faithful reference
than `nltk.corpus.brown`'s own tokenized rendering (which normalizes
quotes/spacing in ways that read as false divergences -- see #14). The
comparison itself never touches NLTK: it's this corpus's own `text` layer
against `brown_nolines.txt`, nothing else.

`brown_nolines.txt` has no per-file markers at all -- it's one continuous
stream of text for all 500 Brown files, in Brown's own canonical order, with
no indication of where one file ends and the next begins. Since SemCor (and
so this corpus) only contains 352 of those 500 files, comparing the whole
reference file directly against this corpus's text would be dominated by the
~150 whole files this corpus never had in the first place -- not a
divergence in anything this corpus actually contains, just noise.

So each of the 500 Brown fileids' [start, end) byte span in
`brown_nolines.txt` needs locating first. That's a one-time, offline step
(`--regenerate-offsets`, needs a local `nltk` install) that anchors on each
file's first sentence (from `nltk.corpus.brown` -- used only to get each
file's approximate opening words as a search anchor, never as the text
being compared) and searches for those words with a cursor that only moves
forward, so files are found in the same left-to-right order they appear in
the reference text. The result is committed as
`src/semcor/brown-nolines-offsets.yaml` -- next to the module that reads
it, since nothing outside this one script has any use for it, unlike the
repo-root `*-fixes.yaml` manifests that `semcor-validate`/CI and this
corpus's own data all depend on -- so normal runs, and CI, never need
NLTK at all, the same generate-the-manifest-once-offline pattern those
manifests already use.

`brown-nolines.txt` itself is committed too, next to its offsets file
(`src/semcor/brown-nolines.txt`), rather than fetched at CI time from
`www.sls.hawaii.edu`: that host has proven unreliable from GitHub-hosted
runners (repeated `curl: (28) Failed to connect`), and there's no reason
this comparison should depend on a third party's uptime at all -- the
reference text doesn't change. `--nolines-file`/`$SEMCOR_BROWN_NOLINES_FILE`
still exist to point at a different copy if the upstream file is ever
updated.

Since `brown_nolines.txt` doesn't consistently render one sentence per
line -- some paragraphs run several sentences together on one line, and
paragraph-vs-sentence line breaks don't line up with this corpus's own
`paragraph` layer either -- comparing at the line level would mostly measure
re-wrapping differences, not real content differences. Instead, each
document (both sides) is rendered one word per line, so the comparison is a
word-level diff: a real difference shows up as a small, localized run of
added/removed word-lines rather than cascading through the rest of the
paragraph the way a fixed-width rewrap would.

The reference side also undoes the original 1961 transcription's own
escaping -- shared by this corpus's source format and by
`brown_nolines.txt` alike (see #9/#11/#17): a run of three '&' is really an
ellipsis, a lone '&' is really a non-sentence-final abbreviation period, and
'+' is really a literal '&'. Without this, every instance already fixed on
this corpus's side would look like a divergence again here.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

_FILEID_RE = re.compile(r"br-([a-z])(\d+)$")

_ANCHOR_WORDS = 6
_SEARCH_WINDOW = 60_000

_DEFAULT_NOLINES_FILE = Path(
    os.environ.get(
        "SEMCOR_BROWN_NOLINES_FILE",
        str(Path(__file__).resolve().parent / "brown-nolines.txt"),
    )
)
_DEFAULT_OFFSETS_FILE = Path(__file__).resolve().parent / "brown-nolines-offsets.yaml"


def brown_fileid_for(path: Path) -> str | None:
    """Map `data/<genre>/br-a01.yaml` to NLTK's Brown fileid `ca01`."""
    m = _FILEID_RE.match(path.stem)
    if not m:
        return None
    letter, num = m.groups()
    return f"c{letter}{num}"


def _alnum_words(sent: list[str], limit: int) -> list[str]:
    out: list[str] = []
    for w in sent:
        if re.search(r"[A-Za-z0-9]", w):
            out.append(w)
        if len(out) >= limit:
            break
    return out


def _escape_anchor_word(word: str) -> str:
    # A '.'-ending word (an abbreviation, per NLTK's tokenization) needs to
    # also match its raw '&' spelling -- see the module docstring.
    if word.endswith(".") and len(word) > 1:
        return re.escape(word[:-1]) + r"[.&]"
    return re.escape(word)


def _anchor_pattern(words: list[str]) -> re.Pattern[str]:
    # Allow up to 20 non-word characters (including '_', used for this
    # transcription's italics/caps markup) between anchor words, so the
    # anchor still matches across intervening punctuation/markup.
    return re.compile(
        r"[\W_]{0,20}".join(_escape_anchor_word(w) for w in words), re.IGNORECASE
    )


def locate_file_boundaries(nolines_text: str, fileids: list[str]) -> dict[str, int]:
    """Find each Brown fileid's start offset in `nolines_text`.

    Needs `nltk` (only for `nltk.corpus.brown`'s per-file first sentence, as
    a search anchor -- see the module docstring); only called from
    `--regenerate-offsets`, never during a normal comparison run.

    Searches with a cursor that only advances, in `fileids` order (NLTK's
    own fileid order already matches brown_nolines.txt's), bounded to a
    fixed window ahead of the cursor -- an unbounded search combined with a
    short, permissive anchor risks matching a much later, unrelated
    occurrence of common words and silently corrupting every subsequent
    file's position.
    """
    from nltk.corpus import brown

    starts: dict[str, int] = {}
    cursor = 0
    missing: list[str] = []
    for fileid in fileids:
        first_sent = brown.sents(fileids=[fileid])[0]
        words = _alnum_words(first_sent, _ANCHOR_WORDS)
        window = nolines_text[cursor : cursor + _SEARCH_WINDOW]
        found = False
        for k in range(min(_ANCHOR_WORDS, len(words)), 0, -1):
            m = _anchor_pattern(words[:k]).search(window)
            if m:
                starts[fileid] = cursor + m.start()
                cursor = cursor + m.start() + 1
                found = True
                break
        if not found:
            missing.append(fileid)

    if missing:
        raise RuntimeError(
            f"Could not locate {len(missing)} Brown file(s) in the reference "
            f"text: {', '.join(missing[:10])}"
            + (", ..." if len(missing) > 10 else "")
        )
    return starts


def regenerate_offsets(nolines_file: Path, offsets_file: Path) -> None:
    """Recompute brown-nolines-offsets.yaml from scratch. Needs `nltk`
    installed locally (`uv pip install nltk`, or `pip install nltk` in the
    project's venv) and its 'brown' corpus downloaded -- not otherwise a
    dependency of this project, see the module docstring.
    """
    import nltk

    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    with nolines_file.open("r", encoding="utf-8") as f:
        nolines_text = f.read()

    fileids = brown.fileids()
    starts = locate_file_boundaries(nolines_text, fileids)
    spans = {
        fileid: [
            starts[fileid],
            starts[fileids[i + 1]] if i + 1 < len(fileids) else len(nolines_text),
        ]
        for i, fileid in enumerate(fileids)
    }

    with offsets_file.open("w", encoding="utf-8") as f:
        f.write(
            "# Byte offsets [start, end) for each Brown Corpus fileid within a local\n"
            "# copy of http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt (read\n"
            "# with universal newlines, i.e. CRLF already normalized to LF).\n"
            "#\n"
            "# Generated once, offline, via:\n"
            "#   uv run semcor-compare-brown-nolines --regenerate-offsets\n"
            "# (needs nltk installed locally; not a runtime dependency otherwise --\n"
            "# see this module's docstring in src/semcor/compare_brown_nolines.py)\n"
        )
        yaml.safe_dump(spans, f, default_flow_style=None, sort_keys=False)
    print(f"Wrote {len(spans)} file offsets to {offsets_file}.")


def load_offsets(offsets_file: Path) -> dict[str, tuple[int, int]]:
    with offsets_file.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {fileid: (start, end) for fileid, (start, end) in raw.items()}


_ELLIPSIS_RUN_RE = re.compile(r"&\s*&\s*&")


def decode_reference_text(text: str) -> str:
    """Undo brown_nolines.txt's transcription escapes -- see the module
    docstring."""
    text = _ELLIPSIS_RUN_RE.sub("...", text)
    text = text.replace("&", ".").replace("+", "&")
    return text


def reference_doc_words(nolines_text: str, start: int, end: int) -> list[str]:
    return decode_reference_text(nolines_text[start:end]).split()


def our_doc_words(path: Path) -> list[str]:
    """A document's sentences' `text`, in file order, as a flat word list.

    Underscore-joined multiword tokens are rendered back to spaces first --
    that's this corpus's own annotation convention, not a real Brown
    Corpus character, so keeping it would show up as a spurious divergence
    on every single-word occurrence of a multiword collocation.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    words: list[str] = []
    for sent_id, sent in data.items():
        if sent_id == "_meta" or not isinstance(sent, dict):
            continue
        text = (sent.get("text") or "").replace("_", " ")
        words.extend(text.split())
    return words


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render data/ as plain text in brown_nolines.txt's own "
        "document order and diff it, word-per-line, against a local copy "
        "of that reference file."
    )
    parser.add_argument("data_dir", nargs="?", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--nolines-file",
        type=Path,
        default=_DEFAULT_NOLINES_FILE,
        help="Local copy of brown_nolines.txt "
        "(default: %(default)s, or $SEMCOR_BROWN_NOLINES_FILE)",
    )
    parser.add_argument(
        "--offsets-file",
        type=Path,
        default=_DEFAULT_OFFSETS_FILE,
        help="Manifest of each Brown fileid's byte span in --nolines-file "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Where to write the generated .txt/.diff files (default: %(default)s)",
    )
    parser.add_argument(
        "--regenerate-offsets",
        action="store_true",
        help="Recompute --offsets-file from --nolines-file and exit "
        "(needs nltk installed locally; see the module docstring)",
    )
    args = parser.parse_args()

    if not args.nolines_file.exists():
        print(
            f"error: {args.nolines_file} not found.\n\n"
            "A copy is committed at src/semcor/brown-nolines.txt; pass "
            "--nolines-file or set $SEMCOR_BROWN_NOLINES_FILE to point "
            "elsewhere, or fetch a fresh copy with e.g.:\n"
            f"  curl -o {args.nolines_file} "
            "http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt",
            file=sys.stderr,
        )
        return 2

    if args.regenerate_offsets:
        regenerate_offsets(args.nolines_file, args.offsets_file)
        return 0

    if not args.offsets_file.exists():
        print(
            f"error: {args.offsets_file} not found.\n\n"
            "Generate it once with:\n"
            "  uv run semcor-compare-brown-nolines --regenerate-offsets",
            file=sys.stderr,
        )
        return 2

    with args.nolines_file.open("r", encoding="utf-8") as f:
        nolines_text = f.read()
    offsets = load_offsets(args.offsets_file)

    files = [p for p in find_yaml_files(args.data_dir) if brown_fileid_for(p)]
    files.sort(key=lambda p: brown_fileid_for(p))

    missing = [p for p in files if brown_fileid_for(p) not in offsets]
    if missing:
        print(
            f"error: {len(missing)} file(s) have no entry in {args.offsets_file}: "
            f"{', '.join(str(p) for p in missing[:10])}",
            file=sys.stderr,
        )
        return 2

    ours_path = args.out_dir / "brown-nolines-ours.txt"
    ref_path = args.out_dir / "brown-nolines-reference.txt"
    diff_path = args.out_dir / "brown-nolines.diff"

    changed_docs = 0
    with (
        ours_path.open("w", encoding="utf-8") as ours_f,
        ref_path.open("w", encoding="utf-8") as ref_f,
    ):
        for path in files:
            fileid = brown_fileid_for(path)
            assert fileid is not None
            header = f"=== {path.stem} ({fileid}) ===\n"
            start, end = offsets[fileid]

            ours_words = our_doc_words(path)
            ref_words = reference_doc_words(nolines_text, start, end)
            if ours_words != ref_words:
                changed_docs += 1

            ours_f.write(header)
            ours_f.write("\n".join(ours_words))
            ours_f.write("\n\n")

            ref_f.write(header)
            ref_f.write("\n".join(ref_words))
            ref_f.write("\n\n")

    result = subprocess.run(
        ["diff", "-u", str(ours_path), str(ref_path)],
        capture_output=True,
        text=True,
    )
    diff_path.write_text(result.stdout, encoding="utf-8")

    divergent_words = sum(
        1
        for line in result.stdout.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )

    print(f"Compared {len(files)} document(s) against {args.nolines_file}.")
    print(f"{divergent_words} divergent word-line(s) across {changed_docs} document(s).")
    print(f"Generated: {ours_path}, {ref_path}")
    print(f"Full diff: {diff_path}")
    return 1 if divergent_words else 0


if __name__ == "__main__":
    sys.exit(main())
