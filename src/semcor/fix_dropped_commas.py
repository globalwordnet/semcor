"""Restore 52 ordinary sentence commas dropped for no shared reason
(fixes #31, split off from #15).

Unlike the rest of this stack, there's no single mechanical cause here
-- re-running the same whole-document `nltk.corpus.brown` alignment
`semcor-verify-brown`/#15 use and isolating pure single-comma deletions
not between two digits turns up 52 ordinary, independent dropped
commas (parenthetical/interruptive commas, appositive and list commas,
a couple of comma-preceded quote attributions) scattered across mixed
genres with no shared trigger. Reading every one in its real sentence
context confirms all 52 are genuine losses Brown has and this corpus
doesn't -- but there turns out to be a clean, uniform *shape* to each
fix even without a shared cause:

- 46 sit at a plain token boundary: the missing comma belongs glued to
  the end of the token before it, with the pre-existing word-gap space
  becoming its trailing space. Fixed here by inserting a brand new `,`
  token there (`insert-token`, `dropped-comma-insert-fixes.yaml`).
- 6 (across 5 sentences -- one token needs two internal commas
  restored) fall *inside* an existing token instead, because the words
  on either side got joined by this corpus's own multiword convention
  before the comma was ever dropped (`Department_of_Health_Education_and_Welfare`
  missing "Health, Education, and Welfare"'s two commas, a movie title,
  a legal citation, a person's name, a plural noun). Fixed here by
  growing that one token's own span in place (`grow-token`,
  `dropped-comma-grow-fixes.yaml`) -- the same "expand one token's own
  span" shape #15/#17 already use, not a new token. `lemmas` updated
  alongside only where it already mirrors the token's own surface
  (exactly or lowercased, #17's extension of #28's check) -- true for
  2 of the 5 tokens (an unsensed legal citation, an unsensed movie
  title); the other 3 have a real WordNet lemma unrelated to the
  literal surface and are left untouched.

3 further instances -- a whole `<digit>,` outline-number sequence
(Brown's `3,`/`7,`/`4,` opening a new point in running prose) missing
*completely*, not just a comma -- are a different, rarer shape of loss
(a whole leading token gone, not an isolated punctuation slip) and are
deliberately not fixed here; see the issue this fixes for the reasoning.

Every `insert-token` fix is located by scanning for the surface of the
token it attaches after, from a cursor that only ever advances (same
reasoning as #10's `split_sentence` -- necessary here too, since a
sentence can have more than one fix and a recurring surface like `**f`
needs fixes matched in original left-to-right order, not just by
content). Every `grow-token` fix's `rel_offset` is relative to its own
token's start, so multiple insertions into the same token (processed
highest-offset-first) never need re-deriving after an earlier one in
the same token shifts things.

Both manifests were generated once, offline, from this session's
investigation; this script has no runtime NLTK dependency and just
applies them.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

INSERT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "dropped-comma-insert-fixes.yaml"
GROW_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "dropped-comma-grow-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_LINE = {
    key: re.compile(rf"(?m)^    {key}: (\[.*\])$")
    for key in ("tokens", "lemmas", "pos", "oewn_key", "wn16_key", "wn30_key")
}

_WIDTH = 10**9
_COMMA = ","


def load_insert_manifest(path: Path = INSERT_MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, str]]]]:
    """Return {filename: {sentence_id: [(index_hint, after_surface), ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["index"], entry["after"]))
    return by_file


def load_grow_manifest(path: Path = GROW_MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Return {filename: {sentence_id: [(index, rel_offset), ...]}}."""
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append((entry["index"], entry["rel_offset"]))
    return by_file


def insert_comma_tokens(sent: dict, fixes: list[tuple[int, str]]) -> bool:
    """Insert a new `,` token after each fix's target token, mutating
    `sent` in place. Returns whether anything actually changed.

    Each fix is `(index_hint, after_surface)`: `after_surface` is the
    exact surface of the token to attach the comma to, captured once at
    manifest-generation time against the true pristine (pre-fix) data --
    it must *not* be re-derived from `sent["tokens"][index_hint - 1]` at
    apply time, since on any run after the first, an earlier fix in this
    same sentence has already permanently shifted the file's layout, and
    re-reading at a stale index would silently pick up a different
    token (or the previously-inserted comma itself).

    `after_surface` alone isn't quite enough to locate the right
    occurrence either: it's an ordinary word (e.g. "God"), and nothing
    stops it recurring *earlier* in the same sentence for an unrelated
    reason -- a scan from the very start of the sentence finds that
    decoy instead. What's reliable is combining it with each fix's own
    rank among this sentence's hints, sorted ascending: every fix with a
    smaller original hint is guaranteed to sit before this one and
    (whether inserted in this call or a previous run) has already added
    exactly one token by the time this fix's real position is reached --
    so `index_hint - 1 + rank` is this fix's exact current token index,
    on a first run and an idempotent second one alike, and the search
    starts there (scanning forward only as a tolerance margin, never
    backward into an earlier decoy).
    """
    ranked = [
        (hint - 1 + rank, after_surface)
        for rank, (hint, after_surface) in enumerate(sorted(fixes, key=lambda f: f[0]))
    ]

    applied = False
    for search_start, after_surface in ranked:
        text = sent["text"]
        tokens = sent["tokens"]

        pos_idx = None
        for i in range(max(0, search_start), len(tokens)):
            if text[tokens[i][0] : tokens[i][1]] == after_surface:
                pos_idx = i
                break
        if pos_idx is None:
            continue  # can't locate -- already applied differently, or gone

        insert_at = pos_idx + 1
        if insert_at < len(tokens) and text[tokens[insert_at][0] : tokens[insert_at][1]] == _COMMA:
            if tokens[pos_idx][1] == tokens[insert_at][0]:
                continue  # already applied: a glued comma already sits right here

        char_pos = tokens[pos_idx][1]
        sent["text"] = text[:char_pos] + _COMMA + text[char_pos:]
        sent["tokens"] = (
            [[s, e] for s, e in tokens[:insert_at]]
            + [[char_pos, char_pos + 1]]
            + [[s + 1, e + 1] for s, e in tokens[insert_at:]]
        )
        sent["pos"] = sent["pos"][:insert_at] + [","] + sent["pos"][insert_at:]
        sent["lemmas"] = sent["lemmas"][:insert_at] + [_COMMA] + sent["lemmas"][insert_at:]
        for key in ("oewn_key", "wn16_key", "wn30_key"):
            for entry in sent.get(key) or []:
                if entry[0] >= insert_at:
                    entry[0] += 1
        applied = True

    return applied


def grow_tokens_with_commas(sent: dict, fixes: list[tuple[int, int]]) -> bool:
    """Grow each `(index, rel_offset)`-targeted token by inserting a `,`
    at that relative position, mutating `sent` in place. Returns
    whether anything actually changed.

    `rel_offset` is relative to the token's own start *as originally
    generated* (against the pristine, pre-fix surface). A token needing
    more than one comma (only `Department_of_Health_Education_and_Welfare`
    does, here) has its offsets processed ascending with a running
    `shift`: the first one lands at its own original position, and every
    later one lands `shift` characters further right, accounting for
    however many earlier-in-this-token commas -- inserted just now, or
    already there from a previous run -- sit before it. Checking a
    later offset's stale, un-shifted position directly against an
    already-fixed surface (tried first, and wrong) finds the wrong
    character and never recognizes the fix as already applied.
    """
    by_index: dict[int, list[int]] = defaultdict(list)
    for index, rel_offset in fixes:
        by_index[index].append(rel_offset)

    applied = False
    for index in sorted(by_index, reverse=True):
        tokens = sent["tokens"]
        if index >= len(tokens):
            continue
        s, e = tokens[index]
        chars = list(sent["text"][s:e])
        old_surface = "".join(chars)

        shift = 0
        growth = 0
        for rel in sorted(by_index[index]):
            pos = rel + shift
            if pos < len(chars) and chars[pos] == _COMMA:
                shift += 1  # already applied here -- still counts for later offsets in this token
                continue
            chars.insert(pos, _COMMA)
            shift += 1
            growth += 1

        if growth == 0:
            continue

        new_surface = "".join(chars)
        new_e = e + growth
        sent["text"] = sent["text"][:s] + new_surface + sent["text"][e:]
        sent["tokens"] = [
            [s, new_e] if i == index else ([ts + growth, te + growth] if ts >= e else [ts, te])
            for i, (ts, te) in enumerate(tokens)
        ]

        lemma = sent["lemmas"][index]
        if lemma == old_surface:
            sent["lemmas"][index] = new_surface
        elif lemma == old_surface.lower():
            sent["lemmas"][index] = new_surface.lower()

        applied = True

    return applied


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


def fix_file(
    path: Path,
    insert_manifest: dict,
    grow_manifest: dict,
    dry_run: bool = False,
) -> list[str]:
    """Apply this file's manifested comma fixes (both kinds).

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    insert_fixes = insert_manifest.get(path.name) or {}
    grow_fixes = grow_manifest.get(path.name) or {}
    if not insert_fixes and not grow_fixes:
        return []

    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, dict] = {}
    for sid in set(insert_fixes) | set(grow_fixes):
        sent = data.get(sid)
        if not isinstance(sent, dict):
            raise RuntimeError(f"{path}: manifested sentence {sid!r} not found")
        new_sent = dict(sent)
        new_sent["tokens"] = list(sent["tokens"])
        new_sent["pos"] = list(sent["pos"])
        new_sent["lemmas"] = list(sent["lemmas"])
        for key in ("oewn_key", "wn16_key", "wn30_key"):
            new_sent[key] = [list(e) for e in (sent.get(key) or [])]

        changed_here = False
        if sid in insert_fixes:
            changed_here |= insert_comma_tokens(new_sent, insert_fixes[sid])
        if sid in grow_fixes:
            changed_here |= grow_tokens_with_commas(new_sent, grow_fixes[sid])

        if changed_here:
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
        description="Restore 52 dropped ordinary sentence commas in "
        "data/*.yaml's `text` (fixes #31), per dropped-comma-insert-fixes.yaml "
        "and dropped-comma-grow-fixes.yaml."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to update (default: data/)",
    )
    parser.add_argument(
        "--insert-manifest",
        type=Path,
        default=INSERT_MANIFEST_PATH,
        help="Insert-token manifest to read (default: %(default)s)",
    )
    parser.add_argument(
        "--grow-manifest",
        type=Path,
        default=GROW_MANIFEST_PATH,
        help="Grow-token manifest to read (default: %(default)s)",
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

    insert_manifest = load_insert_manifest(args.insert_manifest)
    grow_manifest = load_grow_manifest(args.grow_manifest)

    total_sentences = 0
    changed_files = 0
    for path in files:
        changed = fix_file(path, insert_manifest, grow_manifest, dry_run=args.dry_run)
        if changed:
            changed_files += 1
            total_sentences += len(changed)

    verb = "Would fix" if args.dry_run else "Fixed"
    print(f"{verb} {total_sentences} sentence(s) across {changed_files} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
