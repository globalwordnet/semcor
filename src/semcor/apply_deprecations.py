"""Apply Open English Wordnet's synset deprecations to oewn_key in data/.

Open English Wordnet tracks synsets that have been merged away in
`src/deprecations.csv` (columns: ID, ILI, SUPERSEDED_BY, SUPERSEDING_ILI,
REASON). This rewrites data/*.yaml's oewn_key values to follow
ID -> SUPERSEDED_BY, so the corpus keeps tracking OEWN's recommended
migration path even before a deprecated synset is actually dropped from
a release (which is what `semcor-validate` checks against).

Rows where SUPERSEDED_BY lists more than one successor (the synset was
split, not merged) are skipped and reported instead of applied: picking
one successor automatically would be an uncontrolled WSD decision that
belongs to a human annotator, not this script.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

from semcor.validate import DATA_DIR, WORDNET_DIR, find_yaml_files


def load_deprecations(wordnet_dir: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Parse deprecations.csv into a synset ID replacement map.

    Returns (replacements, skipped): `replacements` maps a deprecated
    synset ID to its successor, with chains followed to their end (a
    superseded synset can itself later be superseded) and cycles broken
    (deprecations.csv has at least one mutual pair). `skipped` lists the
    raw rows that split into multiple successors, which aren't applied.
    """
    path = wordnet_dir / "src" / "deprecations.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    direct: dict[str, str] = {}
    skipped: list[dict[str, str]] = []
    for row in rows:
        old_id = row["ID"].removeprefix("ewn-")
        target = row["SUPERSEDED_BY"].strip()
        if "," in target:
            skipped.append(row)
            continue
        direct[old_id] = target.removeprefix("ewn-")

    resolved: dict[str, str] = {}
    for old_id in direct:
        seen = {old_id}
        current = old_id
        while True:
            next_id = direct.get(current)
            if next_id is None or next_id in seen:
                break
            current = next_id
            seen.add(current)
        resolved[old_id] = current

    return resolved, skipped


def _resolve_value(value: str, replacements: dict[str, str]) -> str | None:
    """Apply `replacements` to a (possibly ';'-joined) oewn_key value.

    Returns the new value, or None if none of its parts are deprecated.
    """
    parts = value.split(";")
    new_parts = []
    changed = False
    for part in parts:
        old_id = part.removeprefix("oewn-") if part.startswith("oewn-") else part
        if old_id in replacements:
            new_parts.append(f"oewn-{replacements[old_id]}")
            changed = True
        else:
            new_parts.append(part)
    if not changed:
        return None
    # Drop duplicates a substitution may have created by making two
    # originally-distinct candidates coincide, preserving order.
    return ";".join(dict.fromkeys(new_parts))


def apply_to_file(
    path: Path, replacements: dict[str, str], dry_run: bool = False
) -> list[tuple[str, int, str, str]]:
    """Rewrite oewn_key values in `path` per `replacements`.

    Returns a list of (doc_id, token_index, old_value, new_value) for
    each value that changed (or would change, if `dry_run`). Edits are
    targeted substring substitutions on the raw file text, not a YAML
    parse/dump round-trip, so formatting elsewhere in the file -- key
    order, quoting, flow style -- is left untouched.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    changes: list[tuple[str, int, str, str]] = []
    for doc_id, doc in data.items():
        if doc_id == "_meta" or not isinstance(doc, dict):
            continue
        for i, value in doc.get("oewn_key") or []:
            new_value = _resolve_value(value, replacements)
            if new_value is not None:
                changes.append((doc_id, i, value, new_value))

    if changes and not dry_run:
        text = path.read_text(encoding="utf-8")
        for _, _, old_value, new_value in changes:
            if old_value not in text:
                raise RuntimeError(f"{path}: could not find {old_value!r} to replace")
            text = text.replace(old_value, new_value, 1)
        path.write_text(text, encoding="utf-8")

    return changes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply Open English Wordnet's synset deprecations "
        "(src/deprecations.csv) to oewn_key in data/."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to update (default: data/)",
    )
    parser.add_argument(
        "--wordnet-dir",
        type=Path,
        default=WORDNET_DIR,
        help="Path to an Open English Wordnet checkout, for "
        "src/deprecations.csv (default: %(default)s, or $SEMCOR_WORDNET_DIR)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing anything",
    )
    args = parser.parse_args()

    deprecations_path = args.wordnet_dir / "src" / "deprecations.csv"
    if not deprecations_path.is_file():
        print(
            f"error: no deprecations.csv found at {deprecations_path}\n"
            "Set one up, e.g.:\n"
            "  git clone --depth=1 https://github.com/globalwordnet/english-wordnet "
            f"{args.wordnet_dir}\n"
            "or point --wordnet-dir / $SEMCOR_WORDNET_DIR at an existing checkout.",
            file=sys.stderr,
        )
        return 1

    replacements, skipped = load_deprecations(args.wordnet_dir)

    if args.paths:
        files = []
        for p in args.paths:
            files.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])
    else:
        files = find_yaml_files(DATA_DIR)

    total_changes = 0
    changed_files = 0
    for path in files:
        changes = apply_to_file(path, replacements, dry_run=args.dry_run)
        if changes:
            changed_files += 1
            total_changes += len(changes)
            print(f"{path}:")
            for doc_id, i, old_value, new_value in changes:
                print(f"  - {doc_id}.oewn_key[{i}]: {old_value} -> {new_value}")

    if skipped:
        print(f"\n{len(skipped)} deprecation(s) skipped (split into multiple successors):")
        for row in skipped:
            print(f"  - {row['ID']} -> {row['SUPERSEDED_BY']} ({row['REASON']})")

    verb = "Would change" if args.dry_run else "Changed"
    print(f"\n{verb} {total_changes} oewn_key value(s) across {changed_files} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
