"""Restore a word split around a literal '^' diaeresis escape (fixes #60).

The original 1961 transcription marks a diaeresis over the *previous*
letter with a literal '^' (`Hammarskjo^ld`, `nai^ve`) -- the same escape
`semcor-compare-brown-nolines` already undoes on the *reference* side
(`brown_nolines.txt`, see `decode_reference_text`). This corpus's own
`text` should never have carried the escape at all (every other instance
in this corpus is already plain ASCII, e.g. `Hammarskjold`), but 34
instances leaked through uncorrected, in two shapes:

- Embedded in one token's surface with no whitespace on either side
  (`La^utner`, `Leverku^hn`, `Du^rer`, `Tonio_Kro^ger`, 8 instances,
  all in `data/belles_lettres/br-g15.yaml`): the token itself is
  otherwise correct, just needs the one stray character deleted --
  `lo == hi` in the manifest below, no token merge.
- Split across multiple tokens by a spurious space around the '^'
  (`Scho ^ nberg`, `nai ^ ve`, `Bo ^ o ^ k` for the double-diaeresis
  `Böök`, and once fused as `^_vdingar` straight onto the next word with
  no token of its own -- 26 instances). Here `lo < hi`: the fix merges
  `tokens[lo..hi]` into one token, the same token-count-changing shape as
  `semcor-fix-hyphen-compound-merge`/`semcor-fix-function-word-merges`.

Both shapes reduce to the same text-level operation once the token range
is known: `re.sub(r"\\s?\\^_?\\s?", "", text[tokens[lo][0]:tokens[hi][1]])`
deletes the caret together with at most one adjacent space on each side
(and the one stray underscore in the `^_vdingar` case) -- verified
against every instance's real-world spelling (`Schoenberg`, `naive`,
`Boeoek`/`Book` for `Böök`, `hoevdingar`/`hovdingar`, ...) collapsing to
plain ASCII, matching how this corpus already spells every other
diaeresis/umlaut letter elsewhere.

For `lo < hi`, at most one token in the range ever carries a sense (two,
for `Lake_Va^ttern`, but both copies are identical) -- never a real
sense-on-both-sides editorial choice like #43/#46's hyphen-pair remainder.
So the merged token keeps that one sense verbatim (`lemmas`/`pos`/every
sense-key layer, taken from whichever index has it, first one if tied);
when no fragment has a sense at all, its `lemmas` entries are themselves
literal spelling fragments (not placeholders), so the merged lemma is
built the same way as the merged surface -- concatenated with the same
caret/space cleanup, e.g. `nai`+`ve` -> `naive`.

`src/semcor/caret-diaeresis-fixes.yaml` lists all 34 confirmed `{file,
sentence, lo, hi}` fixes (`lo`/`hi` are the first/last token index in the
range to collapse into one, inclusive; `lo == hi` for the no-merge
embedded case) -- generated once, offline, by a full scan of every
literal '^' in `data/*.yaml` classified by whether it sits at a token's
very start (merge with the preceding token, chaining through any further
bare '^' tokens) or mid-token (simple in-place deletion).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parent / "caret-diaeresis-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")
_LEMMAS_LINE = re.compile(r"(?m)^    lemmas: (\[.*\])$")
_POS_LINE = re.compile(r"(?m)^    pos: (\[.*\])$")
_SENSE_KEY_LINE = {
    key: re.compile(rf"(?m)^    {key}: (\[.*\])$")
    for key in ("oewn_key", "wn16_key", "wn30_key")
}

_CARET_RE = re.compile(r"\s?\^_?\s?")
_WIDTH = 10**9


def _dump_str(value: str) -> str:
    return yaml.safe_dump(value, default_style='"', allow_unicode=True, width=_WIDTH).rstrip("\n")


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def _render_tokens(tokens: list[list[int]]) -> str:
    body = ", ".join(f"[{s}, {e}]" for s, e in tokens)
    return f"    tokens: [{body}]"


def _render_str_list(key: str, values: list[str]) -> str:
    body = ", ".join(_dump_str(v) for v in values)
    return f"    {key}: [{body}]"


def _render_key_layer(key: str, entries: list[list]) -> str:
    body = ", ".join(f"[{idx}, {_dump_str(val)}]" for idx, val in entries)
    return f"    {key}: [{body}]"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Return {filename: {sentence_id: [(lo, hi), ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["lo"], entry["hi"]))
    return by_file


def _sense_at(sent: dict, idx: int) -> dict[str, str]:
    found = {}
    for key in ("oewn_key", "wn16_key", "wn30_key"):
        for pair in sent.get(key) or []:
            if pair[0] == idx:
                found[key] = pair[1]
    return found


def fix_sentence(sent: dict, ranges: list[tuple[int, int]]) -> bool:
    """Apply a sentence's caret-diaeresis merges/deletions in place.

    Returns True if anything changed (False if every fix was already
    applied, e.g. a second run).
    """
    applied = False
    # Rightmost first: collapsing a range removes (hi - lo) tokens and
    # shifts every later index down, but never affects an earlier range.
    for lo, hi in sorted(ranges, key=lambda r: r[0], reverse=True):
        text = sent["text"]
        tokens = sent["tokens"]
        s, e = tokens[lo][0], tokens[hi][1]
        old_span = text[s:e]
        if "^" not in old_span:
            continue  # already fixed -- no-op
        merged = _CARET_RE.sub("", old_span)
        if not merged:
            raise RuntimeError(f"caret cleanup produced an empty token (lo={lo}, hi={hi})")

        removed_len = len(old_span) - len(merged)
        new_end = s + len(merged)

        sent["text"] = text[:s] + merged + text[e:]
        new_tokens = tokens[: lo + 1] + tokens[hi + 1 :]
        new_tokens[lo] = [s, new_end]
        for i in range(lo + 1, len(new_tokens)):
            new_tokens[i] = [new_tokens[i][0] - removed_len, new_tokens[i][1] - removed_len]
        sent["tokens"] = new_tokens

        primary = next((i for i in range(lo, hi + 1) if _sense_at(sent, i)), None)
        lemmas = sent["lemmas"]
        if primary is not None:
            merged_lemma = lemmas[primary]
        else:
            merged_lemma = _CARET_RE.sub("", "".join(lemmas[lo : hi + 1]))
        pos = sent["pos"]
        merged_pos = pos[primary] if primary is not None else pos[lo]

        sent["lemmas"] = lemmas[:lo] + [merged_lemma] + lemmas[hi + 1 :]
        sent["pos"] = pos[:lo] + [merged_pos] + pos[hi + 1 :]

        for key in ("oewn_key", "wn16_key", "wn30_key"):
            arr = sent.get(key) or []
            kept_value = None
            if primary is not None:
                for idx, value in arr:
                    if idx == primary:
                        kept_value = value
                        break
            new_arr = []
            for idx, value in arr:
                if lo <= idx <= hi:
                    continue
                new_arr.append([idx - (hi - lo) if idx > hi else idx, value])
            if kept_value is not None:
                new_arr.append([lo, kept_value])
                new_arr.sort(key=lambda p: p[0])
            sent[key] = new_arr

        applied = True

    return applied


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested caret-diaeresis fixes.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: set[str] = set()
    for sid, ranges in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        if fix_sentence(sent, ranges):
            changed.add(sid)

    if not changed or dry_run:
        return sorted(changed)

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
            sent = data[doc_id]
            chunk, n = _TEXT_BLOCK.subn(lambda m: _render_text(sent["text"]), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text block for {doc_id!r}")
            chunk, n = _TOKENS_LINE.subn(lambda m: _render_tokens(sent["tokens"]), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find tokens line for {doc_id!r}")
            chunk, n = _LEMMAS_LINE.subn(lambda m: _render_str_list("lemmas", sent["lemmas"]), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find lemmas line for {doc_id!r}")
            chunk, n = _POS_LINE.subn(lambda m: _render_str_list("pos", sent["pos"]), chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find pos line for {doc_id!r}")
            for key, pattern in _SENSE_KEY_LINE.items():
                if pattern.search(chunk) is None:
                    continue
                chunk, n = pattern.subn(
                    lambda m, key=key: _render_key_layer(key, sent.get(key) or []), chunk, count=1
                )
                if n != 1:
                    raise RuntimeError(f"{path}: could not find {key} line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return sorted(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore a word split around a literal '^' diaeresis "
        "escape in data/*.yaml (fixes #60), per caret-diaeresis-fixes.yaml."
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
