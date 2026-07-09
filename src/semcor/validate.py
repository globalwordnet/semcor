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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from cerberus import Validator

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

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


def find_yaml_files(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(data_dir.rglob("*.yaml"))


def check_yaml_syntax(path: Path) -> tuple[object | None, str | None]:
    """Parse a YAML file, returning (data, None) or (None, error message)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f), None
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


def validate_file(path: Path) -> list[str]:
    data, syntax_error = check_yaml_syntax(path)
    if syntax_error is not None:
        return [f"YAML syntax error: {syntax_error}"]
    if not isinstance(data, dict):
        return ["top-level YAML content is not a mapping"]

    meta_errors = check_meta_schema(data)
    if meta_errors:
        # Layer types/bases aren't trustworthy if _meta itself is malformed.
        return meta_errors

    return check_layer_offsets(data)


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
    args = parser.parse_args()

    if args.paths:
        files = []
        for p in args.paths:
            files.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])
    else:
        files = find_yaml_files()

    total_errors = 0
    for path in files:
        errors = validate_file(path)
        if errors:
            total_errors += len(errors)
            print(f"{path}:")
            for err in errors:
                print(f"  - {err}")

    print(f"\nChecked {len(files)} file(s), {total_errors} error(s) found.")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
