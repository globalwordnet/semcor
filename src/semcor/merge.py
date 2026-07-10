"""Merge the per-sentence corpus under data/ into a single Teanga YAML
file with one document per Brown Corpus file.

This is the inverse of split_by_document.py in semcor-wordnet-integration,
which explodes one big multi-document corpus into
data/<genre>/<brown_id>.yaml, one sentence per top-level document.

Each data/<genre>/<brown_id>.yaml becomes one merged document, with:

- text: the concatenation of all its sentences' text, in file order
  (which is sentence order, since Teanga's dict-based documents preserve
  YAML's declaration order).
- sentence / paragraph: `div` layers of the character offsets in `text`
  where each sentence / paragraph begins (paragraph boundaries are
  detected from runs of equal `paragraph` values in the per-sentence
  files, the same convention used by semcor.ufsac).
- tokens / lemmas / pos / wn16_key / wn30_key / oewn_key: the
  concatenation of each sentence's layer, with tokens' character
  offsets and the *_key layers' token indices shifted to be relative to
  the merged document rather than each individual sentence.
- brown_id / genre: the source file's stem and parent directory name,
  since Teanga's own document IDs are content hashes with no other
  link back to a Brown Corpus file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import teanga
import yaml

from semcor.validate import DATA_DIR, _YAML_LOADER, find_yaml_files


def new_merged_corpus() -> teanga.Corpus:
    corpus = teanga.Corpus()
    corpus.add_layer_meta("text", layer_type="characters")
    corpus.add_layer_meta("brown_id", layer_type="characters")
    corpus.add_layer_meta("genre", layer_type="characters")
    corpus.add_layer_meta("sentence", layer_type="div", base="text")
    corpus.add_layer_meta("paragraph", layer_type="div", base="text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta("lemmas", layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta("pos", layer_type="seq", base="tokens", data="string")
    corpus.add_layer_meta(
        "wn16_key", layer_type="element", base="tokens", data="string"
    )
    corpus.add_layer_meta(
        "wn30_key", layer_type="element", base="tokens", data="string"
    )
    corpus.add_layer_meta(
        "oewn_key", layer_type="element", base="tokens", data="string"
    )
    return corpus


def merge_file(path: Path, corpus: teanga.Corpus, genre: str) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_YAML_LOADER)

    text_parts: list[str] = []
    sentence_offsets: list[int] = []
    paragraph_offsets: list[int] = []
    tokens: list[tuple[int, int]] = []
    lemmas: list[str] = []
    pos: list[str] = []
    wn16_key: list[list] = []
    wn30_key: list[list] = []
    oewn_key: list[list] = []

    char_offset = 0
    token_offset = 0
    current_paragraph = object()  # sentinel, unequal to any real paragraph value

    for sent_id, sent in data.items():
        if sent_id == "_meta" or not isinstance(sent, dict):
            continue

        text = sent.get("text") or ""
        sent_tokens = sent.get("tokens") or []

        if sent.get("paragraph") != current_paragraph:
            current_paragraph = sent.get("paragraph")
            paragraph_offsets.append(char_offset)
        sentence_offsets.append(char_offset)

        text_parts.append(text)
        tokens.extend((s + char_offset, e + char_offset) for s, e in sent_tokens)
        lemmas.extend(sent.get("lemmas") or [])
        pos.extend(sent.get("pos") or [])
        for i, val in sent.get("wn16_key") or []:
            wn16_key.append([i + token_offset, val])
        for i, val in sent.get("wn30_key") or []:
            wn30_key.append([i + token_offset, val])
        for i, val in sent.get("oewn_key") or []:
            oewn_key.append([i + token_offset, val])

        char_offset += len(text)
        token_offset += len(sent_tokens)

    doc = corpus.add_doc(text="".join(text_parts), brown_id=path.stem, genre=genre)
    doc.sentence = sentence_offsets
    doc.paragraph = paragraph_offsets
    doc.tokens = tokens
    doc.lemmas = lemmas
    doc.pos = pos
    doc.wn16_key = wn16_key
    doc.wn30_key = wn30_key
    doc.oewn_key = oewn_key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge the per-sentence corpus under data/ into a "
        "single Teanga YAML file with one document per Brown Corpus file."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=DATA_DIR,
        help="Directory to merge (default: data/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("semcor.yaml"),
        help="Output file (default: %(default)s)",
    )
    args = parser.parse_args()

    corpus = new_merged_corpus()
    files = find_yaml_files(args.data_dir)
    for path in files:
        merge_file(path, corpus, genre=path.parent.name)

    corpus.to_yaml(str(args.output))
    print(f"Merged {len(files)} file(s) into {args.output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
