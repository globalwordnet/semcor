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
- `text` renders both opening and closing quotation marks as a single
  straight double-quote character (`"`). This is intentional, not a
  lossy normalization: `nltk.corpus.brown` (used as the ground truth
  for `semcor-verify-brown`, #12) represents quotes with its own
  tokenization convention (`` `` `` opening, `''` closing), but that's
  an artifact of NLTK's own tokenized rendering, not the original
  Brown Corpus source text -- a more faithful plaintext rendering of
  Brown (e.g. the widely-used `brown_nolines.txt` reformatting) already
  uses a plain `"` for both, matching this corpus. See #14.

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

The last check needs local checkouts of
[Open English Wordnet](https://github.com/globalwordnet/english-wordnet/)
and [Open English Namenet](https://github.com/globalwordnet/english-namenet/)
(Namenet holds the proper-noun synsets — people, places, organisations,
... — that aren't part of the base wordnet, but that `oewn_key` can still
reference). By default they're looked for at `external/english-wordnet`
and `external/english-namenet`; set them up with either a fresh clone or
a symlink to an existing checkout:

```sh
git clone --depth=1 https://github.com/globalwordnet/english-wordnet external/english-wordnet
git clone --depth=1 https://github.com/globalwordnet/english-namenet external/english-namenet
# or, if you already have checkouts elsewhere:
ln -s /path/to/english-wordnet external/english-wordnet
ln -s /path/to/english-namenet external/english-namenet
```

`external/` is gitignored, so this is a one-time local setup step, not
something to commit. You can point elsewhere instead, via
`--wordnet-dir`/`$SEMCOR_WORDNET_DIR` and `--namenet-dir`/`$SEMCOR_NAMENET_DIR`.
If either checkout can't be found, the script exits with an error rather
than silently skipping the `oewn_key` check.

```sh
uv run semcor-validate                       # validate data/
uv run semcor-validate data/humor            # validate one directory/file
uv run semcor-validate --wordnet-dir /path/to/english-wordnet --namenet-dir /path/to/english-namenet
```

CI runs this on every push to `main` and on every pull request (see
`.github/workflows/validate.yml`), cloning both fresh each time.

### `semcor-apply-deprecations`

Reads Open English Wordnet's `src/deprecations.csv` and rewrites
`oewn_key` values from a deprecated synset to its successor (following
chains, and skipping rows that split into multiple successors -- those
need a human WSD call). Uses the same `--wordnet-dir`/`$SEMCOR_WORDNET_DIR`
checkout as `semcor-validate`.

```sh
uv run semcor-apply-deprecations              # apply to data/
uv run semcor-apply-deprecations --dry-run    # preview without writing
```

Run `semcor-validate` afterwards to confirm the result.

### `semcor-fix-leading-space`

Strips a stray leading space from the `text` layer where present (an
artifact of how sentences were originally split), shifting `tokens`
offsets to match.

```sh
uv run semcor-fix-leading-space              # fix data/
```

Idempotent: documents without a leading space are left untouched, so
it's safe to rerun.

### `semcor-fix-spurious-spacing`

Closes spurious whitespace gaps in `text` that don't exist in the real
Brown Corpus text (fixes #8): a space wrongly inserted just inside an
opening/closing quote, around a `:` between two all-digit tokens
(`11: 30`), or between two short letter(s)+period fragments that are
actually one abbreviation split across tokens (`a. m.`), shifting
`tokens` offsets to match. Only fixes a sentence's quotes when it
contains exactly two -- an unambiguous, self-contained pair -- since
neither a stray never-closed quote nor one nested inside another
(both real, both confirmed to break simple open/close alternation) are
reliably distinguishable from the ordinary case; see the module
docstring for the full reasoning.

```sh
uv run semcor-fix-spurious-spacing              # fix data/
uv run semcor-fix-spurious-spacing --dry-run    # preview without writing
```

Idempotent, like `semcor-fix-leading-space`.

### `semcor-fix-capitalization-after-quote`

Restores sentence-initial capitalization lost after a dialogue-closing
quote + `.`/`?`/`!` (fixes #13), e.g. `"How''s Granny"? and sit` ->
`"How''s Granny"? And sit`. Detects the pattern (`"`, then `.`/`?`/`!`,
then a lowercase-starting word, whether that's later in the same
SemCor sentence or the very next one) across the whole corpus, then
only touches the subset independently confirmed correct against
`nltk.corpus.brown` -- see the module docstring for the handful of
excluded cases and why. Each fix is a single-character case change, so
unlike the other `fix-*` scripts here, `tokens` offsets never need to
shift.

```sh
uv run semcor-fix-capitalization-after-quote              # fix data/
uv run semcor-fix-capitalization-after-quote --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-em-dash`

Restores em dashes corrupted into a single, space-padded `-` (fixes
#9), e.g. `three guns - one in the right pocket` ->
`three guns--one in the right pocket` (Brown's own em-dash token is
`--`, flush against its neighbours). Also restores a small number of
number-range hyphens (`10 - 16` -> `10-16`) found to be a different,
correctly-single-hyphen case during the same check.

Unlike the other `fix-*` scripts, this doesn't re-derive what to fix
from a pattern at runtime -- of 2,939 candidate tokens, cross-checking
each one's context against `nltk.corpus.brown` found a third pattern
(a hyphenated compound that's a single token in Brown, e.g. `80-hp`,
split into three here) that needs token-merging rather than a
whitespace/character edit, so isn't part of this fix at all (tracked
separately as #24). `em-dash-fixes.yaml` lists exactly the 1,885
confirmed fixes from that check; this script only applies that
manifest, with no runtime NLTK dependency.

```sh
uv run semcor-fix-em-dash              # apply em-dash-fixes.yaml to data/
uv run semcor-fix-em-dash --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-function-word-merges`

Splits spuriously merged function-word pairs back into two tokens
(fixes #10), e.g. `in_which` (one token, tag `RB`) -> `in`/IN +
`which`/WDT, matching Brown's actual tokenization. Unlike the other
`fix-*` scripts, this changes the *number* of tokens in a sentence:
`tokens`/`pos`/`lemmas` each go from one entry to two, and every
`oewn_key`/`wn16_key`/`wn30_key` annotation after the split point
shifts by one to keep pointing at the same word.

Which merges are safe to split (and what to split them into) was
decided offline, the same way as #9's `em-dash-fixes.yaml`: a
candidate is a token that's exactly two closed-class function words
joined by `_`, with *no* existing sense annotation at that token index
-- an existing annotation is the corpus's own signal that occurrence
was an intentional multiword unit (some occurrences of the same pair,
e.g. `at_once`, are sense-tagged and some aren't, so this has to be
decided per occurrence, not per word pair). Each survivor was then
confirmed against `nltk.corpus.brown.tagged_words()` to get its real,
context-dependent tags. See the module docstring for the full
reasoning, including why a plain Open English Wordnet entry lookup
was tried and rejected as the filter.

```sh
uv run semcor-fix-function-word-merges              # apply function-word-merge-fixes.yaml to data/
uv run semcor-fix-function-word-merges --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-leftover-ampersand`

Normalizes leftover `& & &` runs (27 sentences across 20 files) to a
real ellipsis `...` (fixes #11), e.g. `sleeping together & & &".` ->
`sleeping together ...".`. Unlike the rest of this stack, Brown isn't
a reliable ground truth here -- checking these positions against
`nltk.corpus.brown` mostly finds nothing at all there, consistent with
this being a genuine "trails off" mark from the original printed
source that Brown's own transcription dropped and this corpus's
intermediate format tried (and, in these 27 cases, failed) to
preserve, rather than spurious markup with a recoverable correct
answer. See the module docstring for the reasoning and the existing
`gap_before`/`gap_after` spacing pattern this fix relies on.

```sh
uv run semcor-fix-leftover-ampersand              # fix data/
uv run semcor-fix-leftover-ampersand --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-case-mismatches`

Aligns capitalization with Brown wherever they differ by exactly one
character's case, in either direction (fixes #28, which generalizes
#13's narrower quote-triggered version). `semcor-verify-brown`'s
whole-document alignment against `nltk.corpus.brown`, filtered to
single-character case-only divergences, found 976 more instances
beyond #13's 152: 861 places this corpus has lost a capital Brown has
(`Despite` -> `despite`, `Larimer St.` -> `Larimer_st.`, plus
embedded-title Title Case and foreign-name particles this corpus had
*correctly* normalized away from Brown's inconsistent title-casing,
e.g. `De Falla` -> `de Falla`), and 115 in the reverse direction --
this corpus adding capitalization Brown's plaintext doesn't have at
all (`the revolt of the moderates` -> `The Revolt Of The Moderates`).
There's no clean rule separating "genuine bug" from "deliberate
improvement over Brown" in that mix, so this fixes both directions
unconditionally, with no exceptions carved out for any of the above
categories -- see the module docstring and #28 for the full reasoning.

Every fix is a single-character, same-length swap: no token/offset
restructuring, unlike #9/#10. Only `text` (and, where the same
position's `lemmas` entry already mirrors the surface case, `lemmas`)
changes. `case-mismatch-fixes.yaml` lists all 976 confirmed fixes;
this script only applies that manifest, with no runtime NLTK
dependency.

```sh
uv run semcor-fix-case-mismatches              # apply case-mismatch-fixes.yaml to data/
uv run semcor-fix-case-mismatches --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

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

### `semcor-merge`

Merges `data/` into a single Teanga YAML file with one document per
Brown Corpus file, instead of one per sentence -- the inverse of
[`split_by_document.py`](https://github.com/jmccrae/semcor-wordnet-integration/blob/main/split_by_document.py),
which produced `data/` in the first place. Sentence/paragraph boundaries
are preserved as `sentence`/`paragraph` layers of character offsets, and
each document carries `brown_id`/`genre` layers, since Teanga's own
document IDs are content hashes with no other link back to a source file.

```sh
uv run semcor-merge                       # merge data/ into ./semcor.yaml
uv run semcor-merge data/humor -o humor-merged.yaml
```

## License

See [LICENSE.md](LICENSE.md). This resource is derived from the Princeton
WordNet database and is further developed under the Creative Commons
Attribution 4.0 International License; attribution to both Princeton
WordNet and the Open English Wordnet team is required.

## Repository layout

- `data/` — the corpus, one Teanga YAML file per Brown Corpus document,
  grouped by genre.
- `src/` — tooling for working with and updating the corpus.
