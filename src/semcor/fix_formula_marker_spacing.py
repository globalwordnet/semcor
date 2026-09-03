"""Merge spuriously split `**f`/`**h` placeholder markers back into one
token (fixes #16).

The Brown Corpus's 1961 transcription used a `**<code>` notation as a
placeholder for symbols it couldn't typeset directly: `**f` where a
math/science formula belongs (mostly in the `learned` genre), `**h` as a
paragraph-break marker (mostly in `fiction_general` dialogue). Comparing
against nltk's `brown` corpus alone can't confirm this, since nltk's own
rendering of these codes is inconsistent -- some become ad hoc letter
codes, others are left as raw `**xx`. Cross-checking instead against
http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt (a plaintext dump
that preserves the original transcription's markup verbatim) confirms
this corpus's own text always has the right characters, just split across
tokens with spurious spaces: `**f` became three tokens `*`, `*`, `f`
(`text: '* * f'`) instead of staying glued as one.

An exhaustive scan of every `data/*/*.yaml` file found 602 of these
3-token runs (580 bare `f`, 18 `h`, 4 `f-fold`/`f-inch` compound
modifiers) to merge back into one token, and a handful of differently-
shaped occurrences deliberately left alone (see `_MERGE_TARGET`):

- A next-token surface joined by `_` rather than `-` (`f_Numbers`) is a
  *second*, independent bug -- this corpus's own tokenizer separately
  fused two real words together -- not this fix's spacing issue, and
  rare enough (2 instances) not to build combined merge+split logic for.
- A single, undoubled `*` (`br-a33.yaml`) is an unrelated footnote-style
  marker, not this `**`-code pattern.
- A dangling `* *` with no following token at all (`br-e08.yaml`) is
  real data loss (the placeholder's own letter is missing), not a
  spacing bug -- nothing to merge it into.

All three are tracked in a follow-up issue rather than fixed here.

Unlike #8/#9/#13/#15, this changes both the *number* of tokens in a
sentence (three become one) and the length of `text` (the two gaps
between the three tokens are deleted): every `oewn_key`/`wn16_key`/
`wn30_key` annotation at a token index after a merge point has to shift
down by two, mirroring #10's index-shifting (in the opposite direction --
merging three into one, not splitting one into two). The word token
(index i+2) can carry a real sense annotation of its own (`**f` for a
degree symbol + "F"/Fahrenheit is sense-tagged on "f" in 12 of the 602
merges) and it follows onto the merged token; the two `*` tokens never
carry one (checked exhaustively against every merge candidate), so this
raises rather than silently dropping one if that assumption is ever
wrong.

No manifest and no runtime NLTK dependency: unlike #15, there's no
Brown-side content being pulled in here, so the merge rule is fully
determined by this corpus's own token structure and can be recomputed
from `data/` directly every run.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_LINE = {
    key: re.compile(rf"(?m)^    {key}: (\[.*\])$")
    for key in ("tokens", "lemmas", "pos", "oewn_key", "wn16_key", "wn30_key")
}

_WIDTH = 10**9

# The confirmed word-token shapes that follow a "*", "*" run: a bare
# formula/paragraph code, or a hyphenated compound modifier built on one
# (`f-fold`, `f-inch`). An underscore-joined surface (`f_Numbers`) is a
# different bug -- deliberately not matched, see module docstring.
_MERGE_TARGET = re.compile(r"^[fh](-\S+)?$")


def find_merges(tokens: list[list[int]], surfaces: list[str]) -> list[int]:
    """Return the starting index of each 3-token "*", "*", word run to
    merge into one token.
    """
    merges = []
    i = 0
    n = len(tokens)
    while i + 2 < n:
        if surfaces[i] == "*" and surfaces[i + 1] == "*" and _MERGE_TARGET.match(surfaces[i + 2]):
            merges.append(i)
            i += 3
        else:
            i += 1
    return merges


def merge_sentence(sent: dict, merge_starts: list[int]) -> dict:
    """Merge each (index, index+1, index+2) run in `merge_starts` into a
    single token, closing the two whitespace gaps between them in `text`
    and shifting every later token/annotation to match.
    """
    if not merge_starts:
        return sent

    text = sent["text"]
    tokens = sent["tokens"]
    pos = sent["pos"]
    lemmas = sent["lemmas"]
    key_layers = {k: sent.get(k) or [] for k in ("oewn_key", "wn16_key", "wn30_key")}

    delete_regions = []  # (start, end) char ranges to delete from text
    for i in merge_starts:
        delete_regions.append((tokens[i][1], tokens[i + 1][0]))
        delete_regions.append((tokens[i + 1][1], tokens[i + 2][0]))
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

    merge_set = set(merge_starts)
    index_map: dict[int, int | None] = {}
    new_tokens: list[list[int]] = []
    new_pos: list[str] = []
    new_lemmas: list[str] = []
    i = 0
    new_i = 0
    n = len(tokens)
    while i < n:
        if i in merge_set:
            new_tokens.append([shift(tokens[i][0]), shift(tokens[i + 2][1])])
            new_pos.append(pos[i + 2])
            new_lemmas.append("**" + lemmas[i + 2])
            # The word token (i+2) can legitimately carry a sense
            # annotation of its own (e.g. "**f" for a degree symbol +
            # "F"/Fahrenheit, sense-tagged on "f") -- it follows onto the
            # merged token, same index as i. Neither "*" token (i, i+1)
            # ever does (checked exhaustively against every merge
            # candidate before this script was written) -- i+1 is left
            # unmapped so a stray annotation there still raises below
            # instead of silently vanishing.
            index_map[i] = new_i
            index_map[i + 1] = None
            index_map[i + 2] = new_i
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
                raise ValueError(
                    f"token {idx} ({key}) has a sense annotation inside a "
                    "merged placeholder run -- should have been excluded"
                )
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


def fix_file(path: Path, dry_run: bool = False) -> list[str]:
    """Merge this file's spuriously split placeholder markers.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, dict] = {}
    for sid, sent in data.items():
        if sid == "_meta" or not isinstance(sent, dict):
            continue
        text = sent.get("text")
        tokens = sent.get("tokens")
        if not isinstance(text, str) or not tokens:
            continue
        surfaces = [text[s:e] for s, e in tokens]
        merges = find_merges(tokens, surfaces)
        if not merges:
            continue
        changed[sid] = merge_sentence(sent, merges)

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
        description="Merge spuriously split '**f'/'**h' placeholder "
        "markers back into one token in data/*.yaml's `text` (fixes #16)."
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

    total_sentences = 0
    changed_files = 0
    for path in files:
        changed = fix_file(path, dry_run=args.dry_run)
        if changed:
            changed_files += 1
            total_sentences += len(changed)

    verb = "Would fix" if args.dry_run else "Fixed"
    print(f"{verb} {total_sentences} sentence(s) across {changed_files} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
