"""Export the corpus under data/ to the UFSAC XML format.

UFSAC (https://github.com/getalp/UFSAC) represents a sense-tagged corpus as

    <corpus>
      <document id="...">
        <paragraph>
          <sentence>
            <word surface_form="..." lemma="..." pos="..." wn30_key="..." />
            ...

`lemma` and the sense-key attributes are only present on words that carry
an annotation; other words have just `surface_form` and `pos`. This mirrors
the shape of UFSAC's own semcor.xml release.

Each Teanga YAML file under data/ becomes one <document>, named after the
file's stem (e.g. `br-a01`). Its sentences (top-level, non-`_meta` keys) are
grouped into <paragraph> elements by runs of equal `paragraph` values, in
file order -- which is also sentence order, since Teanga's dict-based
documents preserve YAML's declaration order.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

import yaml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Order in which sense-key attributes are emitted on a <word>, matching the
# order UFSAC's own semcor.xml uses for wn16_key/wn30_key.
AVAILABLE_KEYS = ("wn16_key", "wn30_key", "oewn_key")


def find_yaml_files(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(data_dir.rglob("*.yaml"))


def build_document(path: Path, keys: list[str]) -> Element:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    doc_elem = Element("document", {"id": path.stem})

    para_elem = None
    current_paragraph = object()  # sentinel, unequal to any real paragraph value
    for doc_id, doc in data.items():
        if doc_id == "_meta" or not isinstance(doc, dict):
            continue

        if doc.get("paragraph") != current_paragraph:
            current_paragraph = doc.get("paragraph")
            para_elem = SubElement(doc_elem, "paragraph")
        sent_elem = SubElement(para_elem, "sentence")

        text = doc.get("text") or ""
        tokens = doc.get("tokens") or []
        lemmas = doc.get("lemmas") or []
        pos_tags = doc.get("pos") or []
        key_maps = {key: dict(doc.get(key) or []) for key in keys}

        for i, (start, end) in enumerate(tokens):
            present = {key: m[i] for key, m in key_maps.items() if i in m}
            attrs = {"surface_form": text[start:end]}
            if present:
                attrs["lemma"] = lemmas[i]
            attrs["pos"] = pos_tags[i]
            for key in AVAILABLE_KEYS:
                if key in present:
                    attrs[key] = present[key]
            SubElement(sent_elem, "word", attrs)

    return doc_elem


def build_corpus(files: list[Path], keys: list[str]) -> Element:
    corpus_elem = Element("corpus")
    for path in files:
        corpus_elem.append(build_document(path, keys))
    return corpus_elem


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the SemCor Teanga YAML corpus to UFSAC XML."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to export (default: data/)",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        choices=AVAILABLE_KEYS,
        default=["oewn_key"],
        help="Which sense-key layer(s) to include as <word> attributes "
        "(default: oewn_key)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    if args.paths:
        files = []
        for p in args.paths:
            files.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])
    else:
        files = find_yaml_files()

    corpus_elem = build_corpus(files, args.keys)
    indent(corpus_elem, space="    ")
    tree = ElementTree(corpus_elem)

    out = args.output if args.output else sys.stdout.buffer
    try:
        tree.write(out, encoding="UTF-8", xml_declaration=False)
    except BrokenPipeError:
        # Piping to e.g. `head` closes stdout early; exit quietly instead of
        # printing a traceback on the write that follows.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
