"""Validate the Teanga YAML corpus under data/.

Currently checks:
1. Every file parses as valid YAML.
2. Every file's `_meta` block declares layers with a recognised Teanga
   type (characters/span/seq/element) and well-formed `base`/`data`
   attributes.
3. Every `span` layer's [start, end] offsets fall within its base layer
   (e.g. `tokens` offsets are valid character offsets into `text`), and
   every `element` layer's index refers to a valid entry in its base
   layer (e.g. `wn16_key`/`wn30_key`/`oewn2026_key` indices are valid
   token indices: 0 <= x < len(tokens)).
4. Every `pos` tag is a valid Penn Treebank tag.
5. Every `oewn_key` refers to a real synset in an Open English Wordnet
   checkout (see `load_wordnet`). `wn16_key`/`wn30_key` are not checked
   against OEWN: they're legacy sense keys, preserved as originally
   annotated, and routinely diverge from current OEWN synsets as the
   corpus's `oewn_key` layer is reannotated against new OEWN releases.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from cerberus import Validator

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

WORDNET_DIR = Path(
    os.environ.get("SEMCOR_WORDNET_DIR")
    or Path(__file__).resolve().parents[2] / "external" / "english-wordnet"
)

# libyaml's C loader is ~4x faster than the pure-Python one; both the
# ~45MB OEWN source and the ~45MB corpus are large enough for this to matter.
_YAML_LOADER = yaml.CSafeLoader if getattr(yaml, "__with_libyaml__", False) else yaml.SafeLoader

_META_LAYER_SCHEMA = {
    "type": {
        "type": "string",
        "allowed": ["characters", "span", "seq", "element"],
        "required": True,
    },
    "base": {"type": "string", "required": False},
    "data": {"type": "string", "required": False},
}

_META_SCHEMA = {
    "_meta": {
        "type": "dict",
        "required": True,
        "valuesrules": {"type": "dict", "schema": _META_LAYER_SCHEMA},
    },
}

# Santorini (1990) Penn Treebank tag set, plus its punctuation tags.
PENN_TREEBANK_TAGS = frozenset(
    {
        "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
        "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
        "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
        "VBZ", "WDT", "WP", "WP$", "WRB",
        "#", "$", ".", ",", ":", "``", "''", "(", ")",
    }
)


def find_yaml_files(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(data_dir.rglob("*.yaml"))


def check_yaml_syntax(path: Path) -> tuple[object | None, str | None]:
    """Parse a YAML file, returning (data, None) or (None, error message)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.load(f, Loader=_YAML_LOADER), None
    except yaml.YAMLError as e:
        return None, str(e)


def check_meta_schema(data: dict) -> list[str]:
    # Document bodies are dynamically-keyed (content-hash IDs), so only
    # `_meta` is validated strictly; everything else is left unknown.
    validator = Validator(_META_SCHEMA)
    validator.allow_unknown = True
    if validator.validate(data):
        return []
    return [f"{field}: {errs}" for field, errs in validator.errors.items()]


def _check_span_offsets(
    doc_id: str, layer_name: str, spans: object, base_len: int
) -> list[str]:
    if not isinstance(spans, list):
        return [f"{doc_id}.{layer_name}: expected a list of [start, end] pairs"]
    errors = []
    for i, span in enumerate(spans):
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(x, int) for x in span)
        ):
            errors.append(f"{doc_id}.{layer_name}[{i}]: expected a [start, end] pair, got {span!r}")
            continue
        start, end = span
        if not (0 <= start <= end <= base_len):
            errors.append(
                f"{doc_id}.{layer_name}[{i}]: span [{start}, {end}] out of bounds "
                f"for base of length {base_len}"
            )
    return errors


def _check_element_offsets(
    doc_id: str, layer_name: str, entries: object, base_len: int
) -> list[str]:
    if not isinstance(entries, list):
        return [f"{doc_id}.{layer_name}: expected a list of [index, value] pairs"]
    errors = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], int):
            errors.append(f"{doc_id}.{layer_name}[{i}]: expected an [index, value] pair, got {entry!r}")
            continue
        index = entry[0]
        if not (0 <= index < base_len):
            errors.append(
                f"{doc_id}.{layer_name}[{i}]: index {index} out of bounds "
                f"(0 <= x < {base_len})"
            )
    return errors


def check_layer_offsets(data: dict) -> list[str]:
    meta = data.get("_meta") or {}
    errors = []
    for doc_id, doc in data.items():
        if doc_id == "_meta" or not isinstance(doc, dict):
            continue
        for layer_name, layer_def in meta.items():
            ltype = layer_def.get("type")
            base = layer_def.get("base")
            if base is None or ltype not in ("span", "element") or layer_name not in doc:
                continue
            if base not in doc:
                errors.append(f"{doc_id}.{layer_name}: base layer '{base}' missing from document")
                continue
            base_len = len(doc[base])
            if ltype == "span":
                errors.extend(_check_span_offsets(doc_id, layer_name, doc[layer_name], base_len))
            elif ltype == "element":
                errors.extend(_check_element_offsets(doc_id, layer_name, doc[layer_name], base_len))
    return errors


def check_pos_tags(data: dict) -> list[str]:
    errors = []
    for doc_id, doc in data.items():
        if doc_id == "_meta" or not isinstance(doc, dict):
            continue
        for i, tag in enumerate(doc.get("pos") or []):
            if tag not in PENN_TREEBANK_TAGS:
                errors.append(f"{doc_id}.pos[{i}]: '{tag}' is not a valid Penn Treebank tag")
    return errors


class WordNetIndex:
    """Synset IDs parsed from an Open English Wordnet checkout."""

    def __init__(self, synsets: set[str]) -> None:
        self.synsets = synsets


def load_wordnet(wordnet_dir: Path = WORDNET_DIR) -> WordNetIndex | None:
    """Parse an Open English Wordnet checkout's `src/yaml/` synset sources.

    Returns None if `wordnet_dir` doesn't look like a checkout (so callers
    can skip wordnet-dependent checks rather than fail outright when a
    contributor hasn't set one up locally).
    """
    yaml_dir = wordnet_dir / "src" / "yaml"
    if not yaml_dir.is_dir():
        return None

    synsets: set[str] = set()
    for path in sorted(yaml_dir.glob("*.yaml")):
        # `entries-*.yaml` (lexical entries/sense keys) and `frames.yaml`
        # aren't keyed by synset ID; every other file is.
        if path.name.startswith("entries-") or path.name == "frames.yaml":
            continue
        with path.open("r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=_YAML_LOADER)
        synsets.update(data.keys())

    return WordNetIndex(synsets)


def check_wordnet_ids(data: dict, wordnet: WordNetIndex) -> list[str]:
    # A handful of tokens carry a genuinely ambiguous SemCor annotation:
    # two or more synset IDs joined by ';' (e.g. `oewn-08562388-n;oewn-08185877-n`).
    errors = []
    for doc_id, doc in data.items():
        if doc_id == "_meta" or not isinstance(doc, dict):
            continue
        for i, value in doc.get("oewn_key") or []:
            for key in value.split(";"):
                synset_id = key[len("oewn-"):] if key.startswith("oewn-") else key
                if synset_id not in wordnet.synsets:
                    errors.append(
                        f"{doc_id}.oewn_key[{i}]: synset '{synset_id}' not found in Open English Wordnet"
                    )
    return errors


def validate_file(path: Path, wordnet: WordNetIndex | None = None) -> list[str]:
    data, syntax_error = check_yaml_syntax(path)
    if syntax_error is not None:
        return [f"YAML syntax error: {syntax_error}"]
    if not isinstance(data, dict):
        return ["top-level YAML content is not a mapping"]

    meta_errors = check_meta_schema(data)
    if meta_errors:
        # Layer types/bases aren't trustworthy if _meta itself is malformed.
        return meta_errors

    errors = check_layer_offsets(data) + check_pos_tags(data)
    if wordnet is not None:
        errors += check_wordnet_ids(data, wordnet)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate SemCor Teanga YAML data files."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to validate (default: data/)",
    )
    parser.add_argument(
        "--wordnet-dir",
        type=Path,
        default=WORDNET_DIR,
        help="Path to an Open English Wordnet checkout, for validating "
        "oewn_key (default: %(default)s, or $SEMCOR_WORDNET_DIR)",
    )
    args = parser.parse_args()

    if args.paths:
        files = []
        for p in args.paths:
            files.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])
    else:
        files = find_yaml_files()

    wordnet = load_wordnet(args.wordnet_dir)
    if wordnet is None:
        print(
            f"Note: no Open English Wordnet checkout found at {args.wordnet_dir}, "
            "skipping oewn_key validation.\n",
            file=sys.stderr,
        )

    total_errors = 0
    for path in files:
        errors = validate_file(path, wordnet)
        if errors:
            total_errors += len(errors)
            print(f"{path}:")
            for err in errors:
                print(f"  - {err}")

    print(f"\nChecked {len(files)} file(s), {total_errors} error(s) found.")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
