"""Add brown-nolines-missing.csv's gaps to data/*.yaml as new, unannotated
tokens, so semcor-compare-brown-nolines stops flagging them.

brown-nolines-missing.csv (see `extract_brown_nolines_review`) lists two
kinds of gap, both re-derived here from the same per-document word
alignment rather than re-parsed from the CSV text (the CSV is a report,
not a diff format built to round-trip):

- A `difflib` `insert` opcode: Brown has words at this position this
  corpus has none of at all (`#MERGER PROPOSED#`). Always safe to insert
  as new tokens -- no existing token is touched.
- A `replace` opcode `_is_content_expansion` flagged (Brown's side,
  alnum-only, contains this corpus's side as a substring and is longer).
  This shape hides two very different cases, and conflating them would
  be actively harmful:

  1. True fusion (`million`, `dollar` -> `multi-million-dollar`): the
     existing words really are being replaced by one differently-typeset
     word. Rare (checked by hand -- essentially one instance in the
     current gap list) and NOT handled by this script; touching an
     existing token risks orphaning a sense annotation the merge would
     need to carry or drop correctly (see `fix_hyphen_compound_merge`'s
     precedent for how much per-case judgment that takes), which isn't
     something to do at 800-row scale on autopilot.
  2. Subheadline-before-existing-content (`Arnold`, `Palmer` ->
     `#GOLF'S GOLDEN BOY# ARNOLD PALMER`): Brown's side *ends* with the
     same words this corpus already has (case-insensitively) -- the
     ALL-CAPS paragraph-lead convention `_adopt_our_casing` already
     handles, just not here because the opcode bundling the subheadline
     gap in with it made it an uneven-length replace, which
     `_adopt_our_casing`'s own equal-length-only check skips. This is
     really *only* an insert (the leading words Brown's side has that
     this corpus's matching suffix doesn't) with no fusion at all --
     detected via `_common_suffix_len` and reduced to an ordinary insert
     before the same token the existing suffix starts at. The existing
     tokens are never touched, so their senses are untouched too.

Every insert (plain, or reduced from case 2) must land exactly on a
token boundary in this corpus's own `tokens` -- checked via
`_token_index_at`; the ~1% that don't (a word this corpus's own text
already has fused with no space to the token before/after the gap) are
skipped and stay in the diff for manual handling, rather than guessed
at.

New tokens get no wn16_key/wn30_key/oewn_key -- they're real, correct
content, but sense-tagging them is a WSD judgment call for a human, not
this script; that's the entire reason `brown-nolines-missing.csv`
exists as a separate queue in the first place. `pos` is a best-effort
`nltk.pos_tag` call over the new words in their real surrounding context
(the existing words on both sides), mapped back into this corpus's own
Penn Treebank tag set; `lemmas` is the lowercased surface form, matching
this corpus's own convention for tokens it never deep-lemmatizes
(`Friday` -> `friday`, `Atlanta` -> `atlanta`). Both are provisional, the
same way the tokens themselves are -- a future WSD pass over
brown-nolines-missing.csv's rows is expected to refine them, not just
add senses.
"""

from __future__ import annotations

import argparse
import re
import sys
import difflib
from pathlib import Path

import nltk
import yaml

from semcor.compare_brown_nolines import (
    _DEFAULT_NOLINES_FILE,
    _DEFAULT_OFFSETS_FILE,
    _adopt_our_casing,
    _decode_and_split,
    brown_fileid_for,
    load_offsets,
)
from semcor.extract_brown_nolines_review import _LARGE_OPCODE_WORDS, _is_content_expansion
from semcor.validate import DATA_DIR, PENN_TREEBANK_TAGS, _YAML_LOADER, find_yaml_files

_WORD_RE = re.compile(r"\S+")
_CONTEXT_WORDS = 4

# nltk's tagger occasionally emits a tag outside this corpus's Penn
# Treebank set (check_pos_tags in validate.py); mapped to the closest
# equivalent already in PENN_TREEBANK_TAGS rather than left to fail
# validation.
_TAG_FALLBACK = {
    "-LRB-": "(",
    "-RRB-": ")",
    "HYPH": ":",
    "NFP": "SYM",
    "ADD": "FW",
    "XX": "FW",
    "AFX": "JJ",
}


def _ensure_tagger() -> None:
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def _guess_pos(before_words: list[str], new_words: list[str], after_words: list[str]) -> list[str]:
    window = before_words + new_words + after_words
    tagged = nltk.pos_tag(window)
    tags = [t for _w, t in tagged[len(before_words) : len(before_words) + len(new_words)]]
    return [_TAG_FALLBACK.get(t, t) if t not in PENN_TREEBANK_TAGS else t for t in tags]


def _guess_lemmas(new_words: list[str]) -> list[str]:
    return [w.lower() for w in new_words]


def _common_suffix_len(a: list[str], b: list[str]) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[-1 - n].lower() == b[-1 - n].lower():
        n += 1
    return n


def our_doc_words_with_spans(path: Path):
    """words, sent_ids, char-spans (in that sentence's own underscore->space
    text) and the parsed per-sentence dict, for one document."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)
    words: list[str] = []
    sent_ids: list[str] = []
    spans: list[tuple[int, int]] = []
    for sent_id, sent in data.items():
        if sent_id == "_meta" or not isinstance(sent, dict):
            continue
        text = (sent.get("text") or "").replace("_", " ")
        for m in _WORD_RE.finditer(text):
            words.append(m.group())
            sent_ids.append(sent_id)
            spans.append((m.start(), m.end()))
    return words, sent_ids, spans, data


def _token_index_at(tokens: list[list[int]], char_pos: int, side: str) -> int | None:
    for i, (s, e) in enumerate(tokens):
        if side == "start" and s == char_pos:
            return i
        if side == "end" and e == char_pos:
            return i
    return None


def find_edits(
    ours_words: list[str],
    sent_ids: list[str],
    spans: list[tuple[int, int]],
    data: dict,
    ref_words: list[str],
) -> list[dict]:
    """Return insert edits: [{sent_id, insert_idx, new_words}], `insert_idx`
    a token-array index (== len(tokens) means "append at the end")."""
    edits: list[dict] = []
    matcher = difflib.SequenceMatcher(a=ours_words, b=ref_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        minus_words = ours_words[i1:i2]
        plus_words = ref_words[j1:j2]
        if max(len(minus_words), len(plus_words)) > _LARGE_OPCODE_WORDS:
            continue

        if tag == "insert":
            new_words = plus_words
            anchor_i1 = i1  # boundary is right before word i1 (or end of doc)
        elif tag == "replace" and _is_content_expansion(minus_words, plus_words):
            suf = _common_suffix_len(minus_words, plus_words)
            if suf < len(minus_words):
                continue  # true fusion or another irregular shape -- skip
            new_words = plus_words[: len(plus_words) - len(minus_words)]
            anchor_i1 = i1  # insert before the (untouched) existing span
        else:
            continue

        if not new_words:
            continue

        if anchor_i1 < len(sent_ids):
            sid = sent_ids[anchor_i1]
            tokens = data[sid].get("tokens") or []
            idx = _token_index_at(tokens, spans[anchor_i1][0], "start")
            if idx is None:
                continue
        elif anchor_i1 > 0:
            sid = sent_ids[anchor_i1 - 1]
            tokens = data[sid].get("tokens") or []
            idx = _token_index_at(tokens, spans[anchor_i1 - 1][1], "end")
            if idx is None:
                continue
            idx += 1
        else:
            continue  # empty document

        before_ctx = ours_words[max(0, anchor_i1 - _CONTEXT_WORDS) : anchor_i1]
        after_ctx = ours_words[anchor_i1 : anchor_i1 + _CONTEXT_WORDS]
        edits.append(
            {"sent_id": sid, "insert_idx": idx, "new_words": new_words, "before_ctx": before_ctx, "after_ctx": after_ctx}
        )
    return edits


# ---- applying edits to one sentence -----------------------------------

_DOC_BOUNDARY = re.compile(r"(?m)^(\S+):[ \t]*$")
_TEXT_BLOCK = re.compile(r"(?m)^    text: .*?(?=\n    tokens: )", re.S)
_LINE = {
    key: re.compile(rf"(?m)^    {key}: (\[.*\])$")
    for key in ("tokens", "lemmas", "pos", "oewn_key", "wn16_key", "wn30_key")
}
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


def apply_edits_to_sentence(sent: dict, edits: list[dict]) -> dict:
    """Apply this sentence's insert edits (already sorted descending by
    insert_idx by the caller) and return a new sentence dict."""
    text = sent["text"]
    tokens = list(sent["tokens"])
    pos = list(sent["pos"])
    lemmas = list(sent["lemmas"])
    key_layers = {k: [list(e) for e in (sent.get(k) or [])] for k in ("oewn_key", "wn16_key", "wn30_key")}

    for edit in edits:
        idx = edit["insert_idx"]
        new_words = edit["new_words"]
        n = len(new_words)
        new_pos = _guess_pos(edit["before_ctx"], new_words, edit["after_ctx"])
        new_lemmas = _guess_lemmas(new_words)

        if idx < len(tokens):
            char_pos = tokens[idx][0]
            insert_text = " ".join(new_words) + " "
            word_start = char_pos
        else:
            char_pos = tokens[-1][1] if tokens else 0
            insert_text = " " + " ".join(new_words)
            word_start = char_pos + 1

        text = text[:char_pos] + insert_text + text[char_pos:]

        new_token_spans: list[list[int]] = []
        cur = word_start
        for w in new_words:
            new_token_spans.append([cur, cur + len(w)])
            cur += len(w) + 1

        delta = len(insert_text)
        tokens = (
            tokens[:idx]
            + new_token_spans
            + [[s + delta, e + delta] for s, e in tokens[idx:]]
        )
        pos = pos[:idx] + new_pos + pos[idx:]
        lemmas = lemmas[:idx] + new_lemmas + lemmas[idx:]
        for layer in key_layers.values():
            for entry in layer:
                if entry[0] >= idx:
                    entry[0] += n

    new_sent = dict(sent)
    new_sent["text"] = text
    new_sent["tokens"] = tokens
    new_sent["pos"] = pos
    new_sent["lemmas"] = lemmas
    new_sent.update(key_layers)
    return new_sent


def fix_file(path: Path, edits_by_sent: dict[str, list[dict]], dry_run: bool = False) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, dict] = {}
    for sid, edits in edits_by_sent.items():
        sent = data.get(sid)
        if not isinstance(sent, dict):
            continue
        edits_sorted = sorted(edits, key=lambda e: e["insert_idx"], reverse=True)
        changed[sid] = apply_edits_to_sentence(sent, edits_sorted)

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
        description="Insert brown-nolines-missing.csv's gaps into data/*.yaml "
        "as new, unannotated tokens."
    )
    parser.add_argument("data_dir", nargs="?", type=Path, default=DATA_DIR)
    parser.add_argument("--nolines-file", type=Path, default=_DEFAULT_NOLINES_FILE)
    parser.add_argument("--offsets-file", type=Path, default=_DEFAULT_OFFSETS_FILE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _ensure_tagger()

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

    total_words = 0
    total_sentences = 0
    changed_files = 0

    for path in files:
        fileid = brown_fileid_for(path)
        assert fileid is not None
        if fileid not in offsets:
            continue
        start, end = offsets[fileid]

        ours_words, sent_ids, spans, data = our_doc_words_with_spans(path)
        ref_words, brace_flags = _decode_and_split(nolines_text, start, end)
        _adopt_our_casing(ours_words, ref_words, brace_flags)

        edits = find_edits(ours_words, sent_ids, spans, data, ref_words)
        if not edits:
            continue

        edits_by_sent: dict[str, list[dict]] = {}
        for e in edits:
            edits_by_sent.setdefault(e["sent_id"], []).append(e)

        changed = fix_file(path, edits_by_sent, dry_run=args.dry_run)
        if changed:
            changed_files += 1
            total_sentences += len(changed)
            total_words += sum(len(e["new_words"]) for e in edits)

    verb = "Would insert" if args.dry_run else "Inserted"
    print(f"{verb} {total_words} word(s) across {total_sentences} sentence(s) in {changed_files} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
