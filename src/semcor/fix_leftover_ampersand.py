"""Normalize leftover `& & &` runs to a real ellipsis `...` (fixes #11).

27 sentences across 20 files have a run of exactly three single-`&`
tokens (always tagged `CC` -- the pipeline that produced this corpus
evidently treated a literal `&` as if it meant "and", which none of
these read as). #8/#9 already established this corpus's `&`-as-markup
convention for a non-sentence-final period (`Mr&` -> `Mr.`); these are
the same kind of leftover, just never converted.

Unlike #8/#9/#13, Brown itself isn't a reliable ground truth for what
belongs here: checking these positions against `nltk.corpus.brown`
(same method as elsewhere in this stack) mostly finds *nothing at all*
at the corresponding position -- Brown's plaintext transcription
doesn't preserve a typographic ellipsis as any distinct character, it
just drops it, which is consistent with these being a genuine "trails
off"/editorial-omission mark from the original printed source that
Brown's own processing lost and this corpus's intermediate format
tried (and in this one case, failed) to keep. So this doesn't reduce
to a Brown-lookup problem the way the rest of this stack does; the fix
is a straightforward, uniform normalization instead.

Every run already sits at exactly the gap pattern a real punctuation
mark would: always preceded by a space, and followed by no space when
the next token is a quote or period (otherwise followed by a space) --
i.e. attached the same way any other closing-adjacent punctuation is
elsewhere in this corpus. So the fix only touches the run itself: the
three `&` tokens (and the spaces between them) collapse into one `...`
token, tagged `.` (matching how other sentence-terminal-style
punctuation is tagged here), with the surrounding gaps left exactly as
they already were.
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
_ELLIPSIS = "..."


def find_runs(sent: dict) -> list[tuple[int, int]]:
    """Return [(start, end)] inclusive index ranges of consecutive
    single-`&` tokens in this sentence."""
    text = sent.get("text") or ""
    tokens = sent.get("tokens") or []
    runs = []
    i = 0
    while i < len(tokens):
        s, e = tokens[i]
        if text[s:e] == "&":
            j = i
            while j + 1 < len(tokens):
                s2, e2 = tokens[j + 1]
                if text[s2:e2] != "&":
                    break
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def fix_sentence(sent: dict) -> dict:
    """Collapse every `& & &` run in a sentence into one `...` token.

    Returns a new sentence dict with `text`/`tokens`/`pos`/`lemmas`/
    `oewn_key`/`wn16_key`/`wn30_key` all updated.
    """
    text = sent["text"]
    tokens = list(sent["tokens"])
    pos = list(sent["pos"])
    lemmas = list(sent["lemmas"])
    key_layers = {k: list(sent.get(k) or []) for k in ("oewn_key", "wn16_key", "wn30_key")}

    runs = find_runs(sent)
    if not runs:
        return sent

    text_parts = []
    cursor = 0
    char_delta_before = {}  # token index -> cumulative char delta from earlier runs
    running_char_delta = 0
    running_token_delta = 0
    token_delta_before = {}

    for start, end in runs:
        run_len = end - start + 1
        if run_len != 3:
            raise ValueError(f"expected a run of exactly 3 '&' tokens, got {run_len}")
        for layer_name, layer in key_layers.items():
            for idx, _ in layer:
                if start <= idx <= end:
                    raise ValueError(f"token {idx} ({layer_name}) has a sense annotation -- unexpected for '&'")

        run_start_char = tokens[start][0]
        run_end_char = tokens[end][1]
        text_parts.append(text[cursor:run_start_char])
        text_parts.append(_ELLIPSIS)
        cursor = run_end_char

        char_delta_before[start] = running_char_delta
        token_delta_before[start] = running_token_delta
        running_char_delta += len(_ELLIPSIS) - (run_end_char - run_start_char)
        running_token_delta += 1 - run_len  # 3 tokens -> 1

    text_parts.append(text[cursor:])
    new_text = "".join(text_parts)

    def char_shift(offset: int) -> int:
        delta = 0
        for start, end in runs:
            if tokens[end][1] <= offset:
                delta += len(_ELLIPSIS) - (tokens[end][1] - tokens[start][0])
            else:
                break
        return offset + delta

    def token_shift(index: int) -> int:
        delta = 0
        for start, end in runs:
            if end < index:
                delta += 1 - (end - start + 1)
            else:
                break
        return index + delta

    new_tokens: list[list[int]] = []
    new_pos: list[str] = []
    new_lemmas: list[str] = []
    run_by_start = dict(runs)
    i = 0
    while i < len(tokens):
        if i in run_by_start:
            end = run_by_start[i]
            new_s = tokens[i][0] + char_delta_before[i]
            new_tokens.append([new_s, new_s + len(_ELLIPSIS)])
            new_pos.append(".")
            new_lemmas.append(_ELLIPSIS)
            i = end + 1
        else:
            s, e = tokens[i]
            new_tokens.append([char_shift(s), char_shift(e)])
            new_pos.append(pos[i])
            new_lemmas.append(lemmas[i])
            i += 1

    for layer in key_layers.values():
        for entry in layer:
            entry[0] = token_shift(entry[0])

    new_sent = dict(sent)
    new_sent["text"] = new_text
    new_sent["tokens"] = new_tokens
    new_sent["pos"] = new_pos
    new_sent["lemmas"] = new_lemmas
    new_sent.update(key_layers)
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
    """Collapse `& & &` runs in `path`.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, dict] = {}
    for sid, sent in data.items():
        if sid == "_meta" or not isinstance(sent, dict):
            continue
        new_sent = fix_sentence(sent)
        if new_sent is not sent:
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
        description="Normalize leftover '& & &' runs to a real ellipsis "
        "'...' in data/*.yaml's text/tokens/pos/lemmas (fixes #11)."
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
