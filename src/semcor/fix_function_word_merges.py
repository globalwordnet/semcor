"""Split spuriously merged function-word pairs back into two tokens
(fixes #10), e.g. `in_which` (one token, tag `RB`) -> `in`/IN +
`which`/WDT, matching Brown's actual tokenization.

Unlike #8/#9/#13, this changes the *number* of tokens in a sentence,
not just whitespace or a character: `tokens`, `pos`, and `lemmas` each
go from one entry to two, and every `oewn_key`/`wn16_key`/`wn30_key`
annotation at a token index after the split point has to shift by one
to keep pointing at the same word.

Which merges are safe to split, and what tags to split them into, was
decided offline (not at runtime -- this script has no NLTK dependency
of its own). The two questions are answered separately and don't gate
each other: whether to split a merge is decided purely by rule 1-2
below, full stop -- it doesn't depend on whether rule 3's Brown lookup
can pin down precise tags for that specific occurrence, since that
lookup can fail for reasons (a multiword-joined neighbour, an
unrelated typo two words over) that have nothing to do with whether
*this* merge is spurious. An earlier version of this script made that
mistake, gating the whole split on rule 3 succeeding, which silently
left genuine candidates like `in_that` un-split just because Brown
`in`+`that` happened to sit next to something else in this corpus that
had already changed shape.

1. Candidates: a token whose surface is exactly two words joined by
   `_`, both from a closed-class function-word list (prepositions,
   determiners, pronouns, conjunctions, auxiliaries, ...) -- the same
   conservative criterion #10 itself used, since these aren't the kind
   of word pair that forms a genuine lexicalized WordNet entry the way
   `take_place` or `in_order_to` do.
2. Filtered to instances with *no* existing `oewn_key`/`wn16_key`/
   `wn30_key` annotation at that token index -- this is what actually
   decides whether to split, unconditionally. A merge some occurrences
   of a pair carry a sense tag and others don't (e.g. `at_once` is
   sense-tagged 66% of the time, `to_that` never) shows this has to be
   decided per occurrence, not per word pair -- and an existing sense
   annotation is the corpus's own signal that this specific occurrence
   was an intentional multiword unit, worth trusting over any
   dictionary lookup or fixed word-pair list. (A plain lookup against
   Open English Wordnet's entries was tried first and rejected: it
   flags things like `in_this` and `to_that` as "genuine" purely
   because *some* rare sense of that string exists as an entry
   somewhere, but `in_this` in this corpus is essentially always plain
   "in" + "this [something]", not that sense, and `to_that`'s 13
   occurrences are 0% sense-tagged.) All 2,091 candidates surviving
   this filter get split; nothing here leaves one merged.
3. Tags for each split, tried in order of precision:
   a. Each candidate's local context was located in
      `nltk.corpus.brown.tagged_words()` for its own file, to read off
      real (Brown tagset), context-dependent tags -- `so_that` is
      `so_CS that_CS` most of the time but not always, confirming the
      issue's own note. This is unambiguous when it works, but only
      resolved 1,853 of 2,091 (one more found this way had to be
      thrown out: Brown's own tagged corpus has `NIL` -- an upstream
      tagging gap -- for the entire sentence around it).
   b. For the other 238, the exact-context search couldn't find a
      unique match (usually a multiword-joined neighbour breaking it) --
      these fall back to the single most common tag pair for that
      exact adjacent word pair across the *whole* Brown corpus (not
      just this file), e.g. `in that` is tagged `IN DT` in 114 of 141
      Brown occurrences overall. Some pairs are unanimous this way
      (`and then` is `CC RB` in all 62 Brown occurrences); a few are
      genuinely split close to evenly (`that is` is `WPS BEZ` in 14 of
      26 confirmed occurrences here, `DT BEZ` in the other 12) and the
      majority vote is a best-effort guess, not a certainty, for those.
   Brown tags are converted to Penn Treebank in both cases before
   being written to the manifest.

`function-word-merge-fixes.yaml` lists all 2,091 fixes, each as (file,
sentence, index, word, pos1, pos2). `word` is the expected merged
surface (`in_that`, `so_that`, ...); `index` is only a same-sentence
ordering hint for when a sentence has more than one fix, not something
trusted to still point at the right token by the time it's applied --
see `split_sentence`. The split point itself (where the `_` is) and
the two lemmas (the existing merged lemma split the same way,
preserving whatever lemmatization already produced it -- e.g. `had_to`
splits to `had`/`to`, `have_to` splits to `have`/`to`) are derived from
the file at fix time, not stored in the manifest.

Splitting at the `_` only ever swaps that one character for a space --
same length, so unlike #8/#9, no character offset in `text` or
`tokens` ever shifts. Only inserting the second token into `tokens`/
`pos`/`lemmas`, and bumping later `*_key` indices, needs any offset
arithmetic at all.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "function-word-merge-fixes.yaml"

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_LINE = {
    key: re.compile(rf"(?m)^    {key}: (\[.*\])$")
    for key in ("tokens", "lemmas", "pos", "oewn_key", "wn16_key", "wn30_key")
}

_WIDTH = 10**9


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, list[tuple[int, str, str, str]]]]:
    """Return {filename: {sentence_id: [(token_index, word, pos1, pos2), ...]}}.

    `token_index` is only a hint (see `split_sentence`): the expected
    merged surface, `word`, is what actually locates the token to
    split, since a sentence with more than one fix would otherwise
    need its later indices re-derived relative to however many earlier
    splits already happened -- fragile across a second run, or even
    just a different processing order. Matching by surface instead
    sidesteps needing that arithmetic at all.
    """
    with path.open("r", encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    by_file: dict[str, dict[str, list[tuple[int, str, str, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        filename = Path(entry["file"]).name
        by_file[filename][entry["sentence"]].append(
            (entry["index"], entry["word"], entry["pos1"], entry["pos2"])
        )
    return by_file


def split_sentence(sent: dict, fixes: list[tuple[int, str, str, str]]) -> dict:
    """Apply a sentence's (token_index, word, pos1, pos2) splits.

    Returns a new sentence dict with `text`/`tokens`/`pos`/`lemmas`/
    `oewn_key`/`wn16_key`/`wn30_key` all updated.

    Each fix is located by scanning for a token whose surface equals
    `word`, starting from a cursor that only ever advances -- not by
    trusting `token_index` directly. That index is still used to order
    the fixes (so two occurrences of the same merged pair in one
    sentence get matched to the right one, left to right), but never
    to index into `tokens` directly: after any earlier split in this
    same call, or from a previous run, everything from that point on
    has shifted, and re-deriving the exact shift is exactly the
    arithmetic bug this sidesteps (found via a stray `whole_thing`
    momentarily becoming a fix target during testing, when a stale
    index landed on an unrelated token that happened to also contain
    an `_`).
    """
    text = sent["text"]
    tokens = list(sent["tokens"])
    pos = list(sent["pos"])
    lemmas = list(sent["lemmas"])
    key_layers = {k: list(sent.get(k) or []) for k in ("oewn_key", "wn16_key", "wn30_key")}

    text_chars = list(text)
    cursor = 0
    for _, word, pos1, pos2 in sorted(fixes, key=lambda f: f[0]):
        index = None
        for i in range(cursor, len(tokens)):
            s, e = tokens[i]
            if text[s:e] == word:
                index = i
                break
        if index is None:
            continue  # already split (e.g. a second run), or genuinely gone -- nothing to do

        s, e = tokens[index]
        surf = text[s:e]
        if surf.count("_") != 1:
            raise ValueError(f"expected exactly one '_' in {surf!r} (token {index})")
        split_at = s + surf.index("_")
        text_chars[split_at] = " "

        lemma_parts = lemmas[index].split("_")
        if len(lemma_parts) != 2:
            raise ValueError(f"expected a two-part lemma, got {lemmas[index]!r} (token {index})")

        tokens[index : index + 1] = [[s, split_at], [split_at + 1, e]]
        pos[index : index + 1] = [pos1, pos2]
        lemmas[index : index + 1] = lemma_parts

        for layer in key_layers.values():
            for entry in layer:
                if entry[0] == index:
                    raise ValueError(f"token {index} has a sense annotation -- should have been filtered out")
                if entry[0] > index:
                    entry[0] += 1

        cursor = index + 2  # past the two tokens this split just produced

    new_sent = dict(sent)
    new_sent["text"] = "".join(text_chars)
    new_sent["tokens"] = tokens
    new_sent["pos"] = pos
    new_sent["lemmas"] = lemmas
    new_sent.update(key_layers)
    return new_sent


def _dump_str(value: str) -> str:
    # default_style='"' forced per-scalar (not on the whole safe_dump
    # call): applying it to a *list* affects every scalar in it
    # uniformly, including plain integers, which turns e.g. `tokens`'
    # `0` into `!!int "0"`. Doing it one string at a time avoids that
    # while still matching every other string list in these files,
    # which are all double-quoted (plain safe_dump on the whole list
    # would pick unquoted/single-quoted scalars per PyYAML's own
    # minimization rules instead -- see #9, which hit exactly this for
    # `lemmas`).
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
    """Apply this file's manifested splits.

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
        new_sent = split_sentence(sent, fixes)
        # Already applied (e.g. a second run): skip so reporting/writes
        # stay honest about what actually changed.
        if new_sent["tokens"] == sent["tokens"]:
            continue
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
        description="Split spuriously merged function-word pairs back "
        "into two tokens in data/*.yaml (fixes #10), per "
        "function-word-merge-fixes.yaml."
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
