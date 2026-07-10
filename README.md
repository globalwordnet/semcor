# SemCor (reannotated)

This repository is a fork and reannotation of [SemCor](https://web.eecs.umich.edu/~mihalcea/downloads.html),
the sense-tagged corpus originally produced at Princeton University. SemCor
consists of a subset of the Brown Corpus in which content words (nouns,
verbs, adjectives and adverbs) are tagged with their lemma, part of speech,
and WordNet sense.

The goal of this fork is to keep SemCor's sense annotations aligned with
current releases of the [Open English Wordnet](https://github.com/globalwordnet/english-wordnet/)
(OEWN), in addition to preserving the original WordNet 1.6 and 3.0 sense
keys, so that the corpus remains usable as training/evaluation data for
WSD systems built against modern wordnets.

## Data format

The corpus is stored as [Teanga](https://github.com/teangaNLP/teanga) YAML.
Teanga models a corpus as a set of documents, each made up of one or more
annotation **layers**. Layers are declared once, in a `_meta` block, and
then instantiated per document. The layer types used here are:

| Type         | Meaning                                                                 |
|--------------|--------------------------------------------------------------------------|
| `characters` | Raw text, indexed by character offset                                    |
| `span`       | Start/end character offsets into a `base` layer (e.g. tokens over text)  |
| `seq`        | One value per index of the `base` layer, in order (e.g. one POS per token) |
| `element`    | Sparse, indexed annotations on the `base` layer (start-index + one)      |

### Layout

Each `.yaml` file under `data/` corresponds to one Brown Corpus document
and is named after the original Brown Corpus file ID (e.g. `br-a01.yaml`).
Files are grouped into directories by Brown Corpus genre category:

```
data/
  press_reportage/
  press_editorial/
  press_reviews/
  religion/
  skill_and_hobbies/
  popular_lore/
  belles_lettres/
  miscellaneous/
  learned/
  fiction_general/
  fiction_mystery/
  fiction_science/
  fiction_adventure/
  fiction_romance/
  humor/
```

### File structure

Every file starts with a `_meta` block declaring its layers:

```yaml
_meta:
    text:
        type: characters
    tokens:
        type: span
        base: text
    lemmas:
        type: seq
        base: tokens
        data: string
    pos:
        type: seq
        base: tokens
        data: string
    paragraph:
        type: characters
    wn16_key:
        type: element
        base: tokens
        data: string
    wn30_key:
        type: element
        base: tokens
        data: string
    oewn_key:
        type: element
        base: tokens
        data: string
```

After `_meta`, each remaining top-level key is a document ID (a short
content hash) representing one sentence, with its layer values:

```yaml
sOM7:
    text: 'The Fulton_County_Grand_Jury said Friday an investigation of...'
    tokens: [[0, 3], [4, 28], [29, 33], ...]
    lemmas: ["The", "group", "say", "friday", ...]
    pos: ["DT", "NN", "VBD", "NNP", ...]
    paragraph: '0'
    wn16_key: [[1, "group%1:03:00::"], [2, "say%2:32:00::"], ...]
    wn30_key: [[1, "group%1:03:00::"], [2, "say%2:32:00::"], ...]
    oewn_key: [[1, "oewn-00031563-n"], [2, "oewn-01011267-v"], ...]
```

- `text` is the raw sentence text; `tokens` are `[start, end]` character
  spans over `text`.
- `lemmas` and `pos` give one value per token, in token order.
- `wn16_key`, `wn30_key`, and `oewn_key` are sparse: each entry is a
  `[token_index, sense_key]` pair, present only for tokens that carry a
  sense annotation. `wn16_key` and `wn30_key` preserve the original
  WordNet 1.6 / 3.0 sense keys from the source SemCor release; `oewn_key`
  gives the corresponding [Open English Wordnet](https://github.com/globalwordnet/english-wordnet/)
  synset ID, and is the layer that gets updated as new OEWN versions are
  released.

## Wordnet alignment

Because word senses and synset inventories change between wordnet
releases, the `oewn_key` layer in this corpus is kept aligned with the
current [Open English Wordnet](https://github.com/globalwordnet/english-wordnet/)
release. As new OEWN versions are published, this repository aims to
update `oewn_key` so downstream WSD work can target the current OEWN
release.

## Running the scripts

Install dependencies with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

### `semcor-validate`

Checks every file under `data/` for YAML syntax, `_meta`/layer-schema
correctness, in-bounds span/element offsets, valid Penn Treebank `pos`
tags, and that every `oewn_key` resolves to a real synset.

The last check needs a local checkout of
[Open English Wordnet](https://github.com/globalwordnet/english-wordnet/).
By default it looks for one at `external/english-wordnet`; set one up
with either a fresh clone or a symlink to an existing checkout:

```sh
git clone --depth=1 https://github.com/globalwordnet/english-wordnet external/english-wordnet
# or, if you already have a checkout elsewhere:
ln -s /path/to/english-wordnet external/english-wordnet
```

`external/` is gitignored, so this is a one-time local setup step, not
something to commit. You can point elsewhere instead, via `--wordnet-dir`
or `$SEMCOR_WORDNET_DIR`. If no checkout can be found, the script exits
with an error rather than silently skipping the `oewn_key` check.

```sh
uv run semcor-validate                       # validate data/
uv run semcor-validate data/humor            # validate one directory/file
uv run semcor-validate --wordnet-dir /path/to/english-wordnet
```

CI runs this on every push to `main` and on every pull request (see
`.github/workflows/validate.yml`), cloning `english-wordnet` fresh each time.

### `semcor-ufsac`

Exports `data/` to the [UFSAC](https://github.com/getalp/UFSAC) XML format.

```sh
uv run semcor-ufsac                               # oewn_key only, to stdout
uv run semcor-ufsac --keys wn16_key wn30_key oewn_key -o semcor.xml
uv run semcor-ufsac data/humor -o humor.xml       # export one directory/file
```

`--keys` selects which sense-key layer(s) to include as `<word>`
attributes (`wn16_key`, `wn30_key`, `oewn_key`); it defaults to `oewn_key`
only.

## License

See [LICENSE.md](LICENSE.md). This resource is derived from the Princeton
WordNet database and is further developed under the Creative Commons
Attribution 4.0 International License; attribution to both Princeton
WordNet and the Open English Wordnet team is required.

## Repository layout

- `data/` — the corpus, one Teanga YAML file per Brown Corpus document,
  grouped by genre.
- `src/` — tooling for working with and updating the corpus.
