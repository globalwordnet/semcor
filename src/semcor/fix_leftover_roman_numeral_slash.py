"""Strip a leftover roman-numeral slash escape from `text`/`tokens`.

brown_nolines.txt (see `compare_brown_nolines.decode_reference_text`)
writes a roman numeral as a plain Arabic digit preceded by a literal '/'
('World War /2,' = "World War II,", 'Louis /15,' = "Louis XV,") -- an
escape this corpus's own construction pipeline normalizes away almost
everywhere. Five sentences across four files never got that
normalization, leaving the raw escape sitting in `text` (and, for the
one case in `data/religion/br-d03.yaml`, doubled into `James_/1_1`
instead of `James_/1` -- some earlier pass appended the digit a second
time rather than replacing the escape). The lemma layer already carries
a semantic lemma (`person`), not this word's surface form, so only
`text` and `tokens` need fixing; `pos`/`lemmas`/`*_key` stay aligned by
index either way.

Edits are targeted substitutions on each document's raw text block, not
a YAML parse/dump round-trip (see `fix_leading_space.py`'s docstring for
why): only the exact characters that need to change -- inside one
token's `text` span, and every `tokens` offset from that point on in the
same sentence -- are touched.
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
_TOKENS_LINE = re.compile(r"(?m)^    tokens: (\[.*\])$")

# 'James_/1_1' -> 'James_1': the one doubled case (see module docstring),
# checked first so the generic rule below doesn't leave the duplicate
# digit behind.
_SLASH_NUM_DUP_RE = re.compile(r"/(\d+)_\1\b")
_SLASH_NUM_RE = re.compile(r"(?<!\d)/(?=\d)")


def _fix_token_text(token_text: str) -> str:
    fixed = _SLASH_NUM_DUP_RE.sub(r"\1", token_text)
    if fixed == token_text:
        fixed = _SLASH_NUM_RE.sub("", token_text)
    return fixed


def find_fix(sent: dict) -> tuple[int, str] | None:
    """Return (token_index, new_token_text) for the first affected token
    in `sent`, or None if it has none."""
    text = sent.get("text") or ""
    for i, (s, e) in enumerate(sent.get("tokens") or []):
        token_text = text[s:e]
        fixed = _fix_token_text(token_text)
        if fixed != token_text:
            return i, fixed
    return None


def _dump_str(value: str) -> str:
    return yaml.safe_dump(value, default_style="'", allow_unicode=True, width=10**9).rstrip("\n")


def _render_text(text: str) -> str:
    text_yaml = yaml.safe_dump(text, default_style="'", allow_unicode=True, width=10**9).rstrip("\n")
    return f"    text: {text_yaml}"


def _render_tokens(tokens: list[list[int]]) -> str:
    body = ", ".join(f"[{s}, {e}]" for s, e in tokens)
    return f"    tokens: [{body}]"


def fix_file(path: Path, dry_run: bool = False) -> list[str]:
    """Fix leftover roman-numeral slash escapes in `path`.

    Returns the list of sentence IDs changed (or that would change, if
    `dry_run`).
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    changed: dict[str, dict] = {}
    for sid, sent in data.items():
        if sid == "_meta" or not isinstance(sent, dict):
            continue
        fix = find_fix(sent)
        if fix is None:
            continue
        idx, new_token_text = fix
        text = sent["text"]
        tokens = list(sent["tokens"])
        s, e = tokens[idx]
        delta = len(new_token_text) - (e - s)
        new_text = text[:s] + new_token_text + text[e:]
        new_tokens = [
            [ts, te] if i < idx else [ts + delta, te + delta] if i > idx else [s, e + delta]
            for i, (ts, te) in enumerate(tokens)
        ]
        new_sent = dict(sent)
        new_sent["text"] = new_text
        new_sent["tokens"] = new_tokens
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
            tokens_repl = _render_tokens(new_sent["tokens"])
            chunk, n = _TOKENS_LINE.subn(lambda m: tokens_repl, chunk, count=1)
            if n != 1:
                raise RuntimeError(f"{path}: could not find tokens line for {doc_id!r}")
        pieces.append(chunk)

    path.write_text("".join(pieces), encoding="utf-8")
    return list(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strip leftover roman-numeral slash escapes ('/2,' -> "
        "'2,') from data/*.yaml's text/tokens."
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
