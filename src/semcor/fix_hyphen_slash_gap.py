"""Close (or fill) the gaps between existing tokens with the hyphen,
slash, or other connector Brown's real text actually has there --
`1 - 1 2` -> `1-1/2`, `3 - to 3` -> `3-to-3`, `22 - year old` ->
`22-year-old` -- without merging, splitting, or otherwise touching the
tokens themselves.

A large class of remaining brown-nolines.txt divergences turned out to
be pure re-punctuation: this corpus already has every word as its own
token, correctly spelled, in the right order -- Brown's real text just
joins some of them with a hyphen or a slash instead of a space, or with
no separator at all, where this corpus still has a plain space (or,
for a handful of already-hyphenated tokens, an underscore doing a
space's job: `submarine_ball` where Brown's real text prints
`submarine-ball`). None of this is missing or wrong content, so
`semcor-add-missing-brown-content` correctly leaves it alone; it isn't
a single-character swap either, so none of the narrower `fix_*` scripts
above cover it.

Critically, this is *not* the same kind of fix as
`semcor-fix-hyphen-compound-merge`: that script actually merges several
tokens into one (dropping the merged span's own sense annotations, per
its docstring, because a compound modifier isn't "a number" or "a word"
in the same sense its parts were). Doing that here at this scale would
require resolving a real conflict on close to three quarters of the
candidates: `22-year-old` alone carries three separate sense
annotations (`22`'s cardinal sense, `year`, `old`) that a merge into one
token would force a choice between, discarding at least one for good.
Since every word is already its own token here, there's a strictly
better option: leave the tokens exactly as they are -- same count, same
`lemmas`/`pos`/every sense-key layer, same everything -- and only
rewrite the *interstitial* text between them (what's currently a plain
space becomes `-`, `/`, or nothing), plus, for the rare token whose own
text has an underscore standing in for what should print as a hyphen,
that one character within the token. Every sense annotation survives
completely untouched, on the same token it was already on.

Detection walked every `replace` opcode from the same per-document
word alignment `semcor-extract-brown-nolines-review` uses, keeping
opcodes where Brown's side is reconstructable from this corpus's side
by inserting only `-`, `/`, or ` ` at the gaps between words (a
two-pointer walk: every non-connector character must appear in the
same order on both sides). Each survivor was then verified individually
against `brown-nolines.txt` (a context-window search requiring a unique
match) before being accepted -- 451 confirmed this way; 2 more that
matched uniquely but didn't reduce to a pure connector-insertion (one
needs a period moved, not a connector inserted; one needs a token's own
underscore reinterpreted as a real space while an unrelated neighbouring
gap gets a new hyphen) were left out rather than forced through a
model that doesn't fit them.

`src/semcor/hyphen-slash-gap-fixes.yaml` lists all 451 confirmed {file,
sentence, start_idx, end_idx, target} fixes -- `target` is the exact
desired reconstruction of tokens[start_idx:end_idx], space-joined the
same way `extract_brown_nolines_review` joins reference words. The
script re-derives the actual per-gap connectors and any within-token
substitution from `target` against the *current* tokens at apply time
(rather than storing them directly), which is also how it recognizes
an already-applied fix as a no-op: after fixing, searching for the
tokens' own (now-already-correct) text against `target` finds zero-width
gaps, so nothing changes on a second run.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "hyphen-slash-gap-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")
_WIDTH = 10**9

# A token's own '_' can stand for a real word-separating space (this
# corpus's usual multiword convention) or, for the handful of targets
# this fix covers, for a hyphen/slash/comma/period Brown's real text
# has there instead -- matched against `target` to find out which.
_UNDERSCORE_FLEX = "(?:_|[-/.,])?"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[dict]]]:
    """Return {filename: {sentence_id: [entry, ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append(entry)
    return by_file


def _tok_pattern(tok: str) -> str:
    parts = re.split(r"(_)", tok)
    return "".join(re.escape(p) if p != "_" else _UNDERSCORE_FLEX for p in parts)


def _compute_gaps(old_texts: list[str], target: str) -> tuple[list[str], list[str]] | None:
    """Find each old token's (possibly underscore-substituted) text inside
    `target`, in order, and the connector text between consecutive
    matches. Returns (new_old_texts, gaps), or None if `target` isn't
    reconstructable this way from `old_texts`."""
    pos = 0
    gaps: list[str] = []
    new_old_texts: list[str] = []
    for k, tok in enumerate(old_texts):
        m = re.compile(_tok_pattern(tok)).search(target, pos)
        if m is None:
            return None
        found = m.start()
        matched_text = m.group()
        if k > 0:
            gaps.append(target[pos:found])
        pos = found + len(matched_text)
        new_old_texts.append(matched_text)
    if pos != len(target):
        return None
    return new_old_texts, gaps


def fix_sentence(
    text: str, tokens: list[list[int]], entries: list[dict]
) -> tuple[str, list[list[int]]] | None:
    """Apply a sentence's hyphen/slash-gap fixes.

    Returns (new_text, new_tokens), or None if every fix was already
    applied (e.g. a second run).
    """
    applied = False
    for entry in sorted(entries, key=lambda e: -e["start_idx"]):
        start_idx, end_idx, target = entry["start_idx"], entry["end_idx"], entry["target"]
        old_texts = [text[s:e] for s, e in tokens[start_idx:end_idx]]
        result = _compute_gaps(old_texts, target)
        if result is None:
            raise RuntimeError(
                f"tokens[{start_idx}:{end_idx}] {old_texts!r} no longer reconstruct "
                f"{target!r} -- refusing to edit"
            )
        new_old_texts, gaps = result
        if new_old_texts == old_texts and all(g == text[tokens[start_idx + k][1] : tokens[start_idx + k + 1][0]] for k, g in enumerate(gaps)):
            continue  # already fixed -- no-op

        span_start = tokens[start_idx][0]
        span_end = tokens[end_idx - 1][1]
        pieces: list[str] = []
        for k, t in enumerate(new_old_texts):
            pieces.append(t)
            if k < len(gaps):
                pieces.append(gaps[k])
        new_span_text = "".join(pieces)
        text = text[:span_start] + new_span_text + text[span_end:]
        delta = len(new_span_text) - (span_end - span_start)

        new_tok_spans = []
        cur = span_start
        for k, t in enumerate(new_old_texts):
            new_tok_spans.append([cur, cur + len(t)])
            cur += len(t)
            if k < len(gaps):
                cur += len(gaps[k])
        tokens = tokens[:start_idx] + new_tok_spans + [[s + delta, e + delta] for s, e in tokens[end_idx:]]
        applied = True

    if not applied:
        return None
    return text, tokens


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def _render_tokens(tokens: list[list[int]]) -> str:
    return "    tokens: [" + ", ".join(f"[{s}, {e}]" for s, e in tokens) + "]"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested hyphen/slash-gap corrections.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, tuple[str, list[list[int]]]] = {}
    for sid, entries in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        result = fix_sentence(sent["text"], sent["tokens"], entries)
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
            chunk, n = _TEXT_BLOCK.subn(lambda m: _render_text(new_text), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text block for {doc_id!r}")
            chunk, n = _TOKENS_LINE.subn(lambda m: _render_tokens(new_tokens), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find tokens line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill the gaps between existing tokens with the "
        "hyphen/slash/other connector Brown's real text has there, per "
        "hyphen-slash-gap-fixes.yaml."
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
