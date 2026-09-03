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
- `text` sometimes renders a parenthetical as `[...]` (square brackets)
  rather than `(...)`, e.g. `[of urbanization]`. This is also
  intentional: every one of the 71 sentences with a `[`/`]` is a
  textbook editorial-insertion bracket -- a clarification inserted into
  a quotation (`[sic]`, `[of urbanization]`), a news-style source
  citation (`[SR, Mar. 25]`), a bracketed alias (`Joseph [Joey]
  Glimco`), a translator's inserted word (`the [Holy] Spirit`), or
  genuine math/science interval notation (`[0, T]`) -- never a plain
  parenthetical mangled into brackets. `brown_nolines.txt` has the
  identical `[...]` at every sampled position, confirming this corpus's
  brackets are a faithful rendering of the real source punctuation.
  `nltk.corpus.brown` is the lossy side here: across its entire
  500-file corpus it has only 2 literal `[` and 2 `]` characters total
  (vs. 2,435/2,466 `(`/`)`), so it essentially never preserves the
  bracket/paren distinction, which is why `semcor-verify-brown`
  (nltk-based) flags these as divergences. See #18.

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

### `semcor-fix-thousands-separator-commas`

Restores thousands-separator commas stripped from numbers (fixes #15),
e.g. `126000` -> `126,000`. `semcor-verify-brown`'s alignment against
`nltk.corpus.brown` found 411 confirmed single-token fixes across 324
sentences in 114 files. Most are a plain missing comma, but two
tokenization wrinkles meant the fix can't just copy Brown's aligned
word verbatim: 168 cases where Brown merges a `$` prefix into the
number as one word (`$1,200`) while this corpus keeps `$` as its own
token, and 11 cases where Brown tokenizes a whole hyphenated compound
(`75,000-ton`) as one word that this corpus already splits into
several tokens. Both are handled by pulling out just the matching
digit run from Brown's word rather than using it whole -- see the
module docstring.

A further ~52 missing commas are ordinary sentence commas (list items,
appositives) with no shared cause, and are deliberately out of scope
here -- see the issue this fixes for the follow-up.

Every fix grows one token in place, expanding only that token's own
span (never a neighbouring gap, since any surrounding whitespace here
is legitimate, unlike #9's em-dash padding). `thousands-separator-fixes.yaml`
lists all 411 confirmed fixes; this script only applies that manifest,
with no runtime NLTK dependency.

```sh
uv run semcor-fix-thousands-separator-commas              # apply thousands-separator-fixes.yaml to data/
uv run semcor-fix-thousands-separator-commas --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-formula-marker-spacing`

Merges spuriously split `**f`/`**h` placeholder markers back into one
token (fixes #16). Brown's 1961 transcription used `**<code>` as a
placeholder for symbols it couldn't typeset directly -- `**f` where a
math/science formula belongs (mostly `learned`-genre texts), `**h` as a
paragraph-break marker (mostly `fiction_general` dialogue) -- but this
corpus split each one across three tokens with spurious spaces
(`text: '* * f'`) instead of keeping it glued as one. `nltk.corpus.brown`
can't confirm this on its own, since its own rendering of these codes is
inconsistent between occurrences (some become ad hoc letter codes,
others stay as raw `**xx`); this was instead verified against
http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt, a plaintext dump
that preserves the original markup verbatim.

An exhaustive scan of every `data/*.yaml` file found 602 of these 3-token
runs to merge (580 bare `f`, 18 `h`, 4 `f-fold`/`f-inch` compound
modifiers) across 342 sentences in 35 files, and a handful of
differently-shaped occurrences deliberately left alone: an
underscore-joined next token (`f_Numbers`, a second, independent
word-fusion bug), one unrelated footnote-style lone `*`, and one dangling
end-of-document `* *` whose own placeholder letter is missing entirely
(real data loss, not a spacing bug) -- see the issue this fixes for the
follow-up.

Unlike the manifest-driven fixes above, this needs no manifest and no
runtime NLTK dependency: the merge rule is fully determined by this
corpus's own token structure (two adjacent `*` tokens immediately
followed by a matching word token), recomputed from `data/` every run.
Like #10, this changes the *number* of tokens (three become one) and the
length of `text` (the two gaps between them are deleted); every
`oewn_key`/`wn16_key`/`wn30_key` annotation past a merge point shifts
down by two to match, and the trailing word token's own annotation (12
of the 602, e.g. `**f` for a degree symbol + sense-tagged "F"/Fahrenheit)
follows onto the merged token -- see the module docstring.

```sh
uv run semcor-fix-formula-marker-spacing              # fix data/
uv run semcor-fix-formula-marker-spacing --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-corrupted-ampersand`

Restores literal `&` characters corrupted to `+` (fixes #17). #8/#9/#11
established that this corpus decodes one Brown transcription escape
convention -- a literal `&` marks a non-sentence-final abbreviation
period (`Mr&` -> `Mr.`) -- but the original transcription needed a
*different* escape for an actual, literal ampersand in running text,
since `&` was already spoken for; it used `+`. This corpus never decoded
that second convention, so `A & M` (Texas A&M) stayed as `A_+_M`.

An exhaustive scan of every `+`-containing token in `data/*.yaml` (81
total, not a sample) found 76 confirmed corruptions -- virtually all
proper-noun ampersands (`Chesapeake + Ohio`, `Smith + Wesson`,
`Standard + Poor's`, `AF + AM`, ...) -- each checked against
`nltk.corpus.brown` and local context. A whole-document Brown diff
alone (the method the rest of this stack uses) missed one real instance
here (`SequenceMatcher`'s greedy matching absorbed a second, nearby `B +
O` mention into an "equal" block once an earlier one was resolved),
caught only by cross-checking against the direct token scan instead.
`sls.hawaii.edu`'s raw dump (the more faithful source #16 used) isn't
the right comparison for *this* fix: it has the identical undecoded `+`
in all 76 places, since it preserves the same escape convention this
corpus does -- `nltk.corpus.brown`, which decodes it, is the right
ground truth here. The remaining 5 `+`-containing tokens are genuine,
unrelated plus signs (statistical correlation values, an explanation of
the `+` symbol itself, a school grade) and are correctly left alone.

Every fix is a single-character, same-length swap, same as #28: no
token/offset restructuring, so `tokens`/`pos`/`oewn_key`/`wn16_key`/
`wn30_key` are untouched. `corrupted-ampersand-fixes.yaml` lists all 78
confirmed `+` character offsets (two tokens have more than one); this
script only applies that manifest, with no runtime NLTK dependency.

```sh
uv run semcor-fix-corrupted-ampersand              # apply corrupted-ampersand-fixes.yaml to data/
uv run semcor-fix-corrupted-ampersand --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-hyphen-compound-merge`

Merges over-split hyphenated compound tokens back into one (fixes #24).
Brown sometimes tokenizes a hyphenated compound modifier as a single
token (`80-hp`, `1787-89`, `3-cm`), but this corpus split some of these
into three tokens with spurious spaces (`80`, `-`, `hp`) -- the same
surface symptom as #9's em-dash bug, but fixing it means merging tokens
back together, not editing whitespace or a character.

#9 already fixed the two other patterns behind a lone `-` surrounded by
spaces (genuine em dashes and separate-token number ranges); of the
~1,054 it couldn't confirm either way, a context-window search against
`nltk.corpus.brown` (2 tokens of real context on each side of the
hyphen, excluding its immediate neighbours since one of those is what
might be swallowed into the compound -- stronger than #9's
neighbour-only search, and something a whole-document diff can't do at
all here, since stripping whitespace before comparing makes `80 - hp`
and `80-hp` identical strings) found 315 confirmed compound merges. This
fixes 289 of those: the *left* token must be purely numeric (`80`,
`1787`, ...) -- 13 word-prefixed cases (`AFL-CIO`, `radio-TV`,
`Class-D`, ...) are left for a follow-up issue, since a number's only
possible sense is always its own generic cardinal/quantity identity
(safe to drop once merged into a compound that isn't "a number"
anymore) while a word carries a real, specific sense a merge would need
an individual editorial call to resolve -- and a further 13 digit-prefix
candidates turned out to be a byte-for-byte mismatch between this
corpus's own content and Brown's merged surface (an abbreviation period
Brown's word has that this corpus tokenizes separately, `per-cent` vs.
`per_cent`, a `1/2` vs. `1_2` fraction, and one case spanning a *fourth*
token this corpus splits off too) and are excluded rather than
fabricating characters this corpus doesn't have.

Every fix merges 3 tokens (number, `-`, word) into 1, closing the two
whitespace gaps in `text` and shifting later `tokens`/`oewn_key`/
`wn16_key`/`wn30_key` indices to match, the same token-count-and-length
change #16's placeholder-marker merge makes -- with one addition #16
never needed: the number token's own sense (if any -- always its
generic cardinal/quantity identity, per the digit-only scope above) is
dropped, while the word token's sense (if any) follows onto the merged
token like #16's trailing-token case. See the module docstring for the
full reasoning, including why each fix is located by scanning for its
surface from an advancing cursor rather than trusting the manifest's
stored index directly (same reasoning as #10's `split_sentence` --
needed here too, since a sentence with more than one fix has every
later index shifted by a merge before it).

`hyphen-compound-merge-fixes.yaml` lists all 289 confirmed fixes;
generated once, offline, against `nltk.corpus.brown`, this script has no
runtime NLTK dependency and just applies that manifest.

```sh
uv run semcor-fix-hyphen-compound-merge              # apply hyphen-compound-merge-fixes.yaml to data/
uv run semcor-fix-hyphen-compound-merge --dry-run    # preview without writing
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
