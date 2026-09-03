"""Merge over-split hyphenated compound tokens back into one (fixes #24).

Brown sometimes tokenizes a hyphenated compound modifier as a *single*
token (`80-hp`, `1787-89`, `3-cm`), but this corpus split some of these
into three tokens with spurious spaces (`80`, `-`, `hp` --
`text: '80 - hp'`) -- the same surface symptom as #9's em-dash bug, but
fixing it means merging tokens back together, not editing whitespace or
a character.

#9 already found and fixed the two other patterns behind a lone `-`
surrounded by spaces (genuine em dashes, and separate-token number
ranges); of the ~1,054 it couldn't confirm either way, this session
built a proper context-window search against `nltk.corpus.brown` (2
tokens of real context on each side of the hyphen, excluding its
immediate neighbours since one of *those* is what might be swallowed
into the compound) to classify the rest. That found 315 confirmed
compound merges, plus a larger number of em dashes #9's narrower
neighbour-only search had missed, and a handful of unrelated anomalies
-- see the issue this fixes for the full breakdown.

This only covers 289 of those 315. First, the *left* token must be
purely numeric (`80`, `1787`, `3,000`, ...): the other 13 are
word-prefixed (`AFL-CIO`, `radio-TV`, `Class-D`, ...) and are left for a
follow-up issue -- unlike a number, whose only possible sense is its own
generic cardinal/quantity identity (confirmed by checking every
candidate's existing sense annotations, and always safe to drop once
merged into a compound that isn't "a number" anymore), a word carries a
real, specific sense that a merge would need an individual editorial
call to resolve (e.g. `Class-D`'s `class%1:14:05::`, `radio-TV`'s
`radio%1:10:00::` -- both real content senses, not boilerplate). Of the
remaining 302 digit-prefixed candidates, 13 more turned out to be a
byte-for-byte mismatch between this corpus's own digit+`-`+word content
and Brown's merged surface -- an abbreviation period Brown's word has
that this corpus tokenizes as a separate `.` token (`24-hr` vs.
`24-hr.`), a `per-cent`/`per_cent` punctuation difference, a fraction
written `1/2` vs. `1_2`, and one case (`Ab63711-r`) where Brown's
compound actually spans a *fourth* token this corpus splits off
separately too. Merging those would mean fabricating characters this
corpus doesn't already have rather than just closing whitespace gaps,
so they're excluded rather than guessed at.

Every fix here merges 3 tokens (number, `-`, word) into 1, closing the
two whitespace gaps in `text` and shifting later `tokens`/`oewn_key`/
`wn16_key`/`wn30_key` indices to match -- the same token-count-and-length
change #16's placeholder-marker merge makes, with one addition #16 never
needed: the number token's own sense annotation (if any -- always its
generic cardinal/quantity identity, per the digit-only scope above) is
dropped rather than carried over, while the word token's sense (if any)
follows onto the merged token exactly like #16's trailing-token case.
Neither token's sense ever conflicts with a sense on the hyphen itself
(checked -- 0 of the 289 have one).

`hyphen-compound-merge-fixes.yaml` lists all 289 confirmed `{file,
sentence, index, replacement, pos}` fixes -- `index` is the *left*
(numeric) token's index, `replacement` is Brown's merged surface, `pos`
is Brown's tag at that position converted to this corpus's Penn
Treebank convention (`NP` -> `NNP`, `-TL`/`-HL` suffixes stripped, same
conversion #10 already does for its own Brown-tag lookups). Generated
once, offline, against `nltk.corpus.brown`; this script has no runtime
NLTK dependency and just applies that manifest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "hyphen-compound-merge-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_LINE = {
    key: re.compile(rf"(?m)^    {key}: (\[.*\])$")
    for key in ("tokens", "lemmas", "pos", "oewn_key", "wn16_key", "wn30_key")
}

_WIDTH = 10**9


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, str, str]]]]:
    """Return {filename: {sentence_id: [(left_index, replacement, pos), ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, str, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["index"], entry["replacement"], entry["pos"]))
    return by_file


def merge_sentence(sent: dict, fixes: list[tuple[int, str, str]]) -> dict | None:
    """Apply a sentence's (left_index, replacement, pos) merges.

    Each fix merges three tokens (number, '-', word) into one. `left_index`
    is only an ordering hint, not trusted to still point at the right
    token: a sentence with more than one fix has every index after the
    first merge shifted by -2, so locating each fix by scanning for its
    surface from a cursor that only advances (same reasoning as #10's
    `split_sentence`) is what actually finds the right tokens, on both
    the first run and an idempotent second one.

    Returns a new sentence dict, or None if every fix was already
    applied (e.g. a second run).
    """
    text = sent["text"]
    tokens = sent["tokens"]
    pos = sent["pos"]
    lemmas = sent["lemmas"]
    key_layers = {k: sent.get(k) or [] for k in ("oewn_key", "wn16_key", "wn30_key")}

    surfaces = [text[s:e] for s, e in tokens]

    merges = []  # (left_index, replacement, tag), using *current* indices
    cursor = 0
    for _, replacement, tag in sorted(fixes, key=lambda f: f[0]):
        found = None
        for i in range(cursor, len(tokens)):
            if surfaces[i] == replacement:
                cursor = i + 1
                break  # already merged into one token here -- nothing to do
            if (
                i + 2 < len(tokens)
                and surfaces[i + 1] == "-"
                and surfaces[i] + "-" + surfaces[i + 2] == replacement
            ):
                found = i
                break
        if found is None:
            continue  # already applied, or genuinely gone
        merges.append((found, replacement, tag))
        cursor = found + 3

    if not merges:
        return None

    merges.sort()
    delete_regions = []  # (start, end) char ranges to delete from text
    for left_index, _, _ in merges:
        delete_regions.append((tokens[left_index][1], tokens[left_index + 1][0]))
        delete_regions.append((tokens[left_index + 1][1], tokens[left_index + 2][0]))
    delete_regions.sort()

    parts = []
    cursor = 0
    for start, end in delete_regions:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    new_text = "".join(parts)

    def shift(offset: int) -> int:
        removed = 0
        for start, end in delete_regions:
            if start < offset:
                removed += min(end, offset) - start
            else:
                break
        return offset - removed

    merge_by_left = {left_index: (replacement, tag) for left_index, replacement, tag in merges}
    merge_set = set(merge_by_left)
    index_map: dict[int, int | None] = {}
    new_tokens: list[list[int]] = []
    new_pos: list[str] = []
    new_lemmas: list[str] = []
    i = 0
    new_i = 0
    n = len(tokens)
    while i < n:
        if i in merge_set:
            replacement, tag = merge_by_left[i]
            s0 = tokens[i][0]
            new_s0 = shift(s0)
            new_e2 = new_s0 + len(replacement)
            # Content check, not just length: the manifest's `replacement`
            # must exactly equal what closing the two gaps actually
            # produces here (this corpus's own digit/hyphen/word
            # characters, untouched) -- confirmed offline for every
            # manifest entry, but a mismatch (an abbreviation period
            # Brown's word has that this corpus tokenizes separately, a
            # `per-cent`/`per_cent` punctuation difference, ...) means
            # gluing would silently produce the wrong text, so this
            # raises instead of trusting a stale manifest entry.
            if new_text[new_s0:new_e2] != replacement:
                raise ValueError(
                    f"merged text {new_text[new_s0:new_e2]!r} does not match "
                    f"manifested replacement {replacement!r}"
                )
            new_tokens.append([new_s0, new_e2])
            new_pos.append(tag)
            new_lemmas.append(replacement)
            index_map[i] = None  # left (number) token's own sense is dropped
            index_map[i + 1] = None  # hyphen never carries one (checked)
            index_map[i + 2] = new_i  # word token's sense follows the merge
            i += 3
        else:
            new_tokens.append([shift(tokens[i][0]), shift(tokens[i][1])])
            new_pos.append(pos[i])
            new_lemmas.append(lemmas[i])
            index_map[i] = new_i
            i += 1
        new_i += 1

    new_key_layers = {}
    for key, entries in key_layers.items():
        new_entries = []
        for idx, val in entries:
            mapped = index_map[idx]
            if mapped is None:
                if any(idx == li + 1 for li in merge_set):
                    raise ValueError(
                        f"token {idx} ({key}) has a sense annotation on a hyphen "
                        "being merged -- should have been excluded"
                    )
                continue  # left (number) token's sense: dropped, per module docstring
            new_entries.append([mapped, val])
        new_key_layers[key] = new_entries

    new_sent = dict(sent)
    new_sent["text"] = new_text
    new_sent["tokens"] = new_tokens
    new_sent["pos"] = new_pos
    new_sent["lemmas"] = new_lemmas
    new_sent.update(new_key_layers)
    return new_sent


def _dump_str(value: str) -> str:
    return yaml.safe_dump(value, default_style='"', allow_unicode=True, width=_WIDTH).rstrip("\n")


def _render_tokens(tokens: list[list[int]]) -> str:
    body = ", ".join(f"[{s}, {e}]" for s, e in tokens)
    return f"    tokens: [{body}]"


def _render_str_list(key: str, values: list[str]) -> str:
    body = ", ".join(_dump_str(v) for v in values)
    return f"    {key}: [{body}]"


def _render_key_layer(key: str, entries: list[list]) -> str:
    body = ", ".join(f"[{idx}, {_dump_str(val)}]" for idx, val in entries)
    return f"    {key}: [{body}]"


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=_WIDTH).rstrip("\n")
    return f"    text: {text_yaml}"


def fix_file(path: Path, manifest: dict, dry_run: bool = False) -> list[str]:
    """Apply this file's manifested compound merges.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    file_fixes = manifest.get(path.name)
    if not file_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, dict] = {}
    for sid, fixes in file_fixes.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        new_sent = merge_sentence(sent, fixes)
        if new_sent is not None:
            changed[sid] = new_sent

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
            new_sent = changed[doc_id]
            text_repl = _render_text(new_sent["text"])
            chunk, n = _TEXT_BLOCK.subn(lambda m: text_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find text block for {doc_id!r}")
            renderers = {
                "tokens": lambda: _render_tokens(new_sent["tokens"]),
                "pos": lambda: _render_str_list("pos", new_sent["pos"]),
                "lemmas": lambda: _render_str_list("lemmas", new_sent["lemmas"]),
                "oewn_key": lambda: _render_key_layer("oewn_key", new_sent["oewn_key"]),
                "wn16_key": lambda: _render_key_layer("wn16_key", new_sent["wn16_key"]),
                "wn30_key": lambda: _render_key_layer("wn30_key", new_sent["wn30_key"]),
            }
            for key, render in renderers.items():
                if key not in new_sent:
                    continue
                line_repl = render()
                chunk, n = _LINE[key].subn(lambda m: line_repl, chunk, count=1)
                if n != 1:
                    raise RuntimeError(f"{path}: could not find {key} line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge over-split hyphenated compound tokens back "
        "into one in data/*.yaml (fixes #24), per hyphen-compound-merge-fixes.yaml."
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
