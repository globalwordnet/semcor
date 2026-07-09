"""Validate the Teanga YAML corpus under data/.

Currently checks:
1. Every file parses as valid YAML.
2. Every file's `_meta` block declares layers with a recognised Teanga
   type (characters/span/seq/element) and well-formed `base`/`data`
   attributes.

Document bodies (the per-sentence entries alongside `_meta`) are not yet
checked against their declared layers; that's the next thing to add here.
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


def validate_file(path: Path) -> list[str]:
    data, syntax_error = check_yaml_syntax(path)
    if syntax_error is not None:
        return [f"YAML syntax error: {syntax_error}"]
    if not isinstance(data, dict):
        return ["top-level YAML content is not a mapping"]
    return check_meta_schema(data)


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
