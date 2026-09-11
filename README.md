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
- `press_reportage` articles are missing datelines (`Washington,Feb.9--`),
  bylines, and mid-article newspaper subheadlines (`nltk.corpus.brown`
  has `..."no evidence" that any irregularities took place. Ask jail
  deputies On other matters...` -- `Ask jail deputies` is a subheadline
  sitting between two unrelated sentences) that Brown's plaintext still
  has. Unlike the two notes above, this **is** a real, confirmed gap,
  not a false alarm: a whole-document scan found 323 such deletions
  across 38 of the genre's 44 files (86%), split between datelines/
  bylines and the more numerous subheadline pattern. But it's also not
  fixable the way #8-#31 fix things -- there's no corrupted or
  mis-tokenized content in this corpus's own data to recover, because
  the text was apparently never there to begin with: `brown_nolines.txt`
  is missing the identical strings too (unlike #16/#18, where it was the
  more complete source), pointing to the loss predating this corpus's
  own construction, likely from however SemCor's annotators originally
  prepared "clean" running prose from Brown decades ago. Restoring it
  would mean synthesizing new, never-sense-tagged sentences from Brown
  and deciding how they fit into this corpus's existing sentence IDs and
  paragraph numbering -- a content-expansion project, not a bug fix. See
  #32.

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
`tokens` offsets to match.

Quote gaps are closed two ways. A sentence with exactly two `"` tokens
is fixed by structural guess alone -- an unambiguous, self-contained
pair, safe without checking anything else. Any other sentence's quotes
(a lone quote continuing from/into another sentence, more than two in
one, sequential pairs or nesting) aren't guessed at structurally --
neither a stray never-closed quote nor same-glyph nesting (both real,
both confirmed to break simple open/close alternation) are reliably
distinguishable from the ordinary case this way. Instead,
`src/semcor/quote-gap-fixes.yaml` verifies each quote's spacing
directly against `src/semcor/brown-nolines.txt`: since that reference
also collapsed both quote directions to a bare `"` (same loss, per
#14), it can't disambiguate open-vs-close either, but it *does*
preserve real spacing -- a unique word-context match around a quote
tells us, per quote and per side independently, whether Brown's real
text has that exact gap or not, with no need to know whether the
sentence's quotes are nested, sequential, or cross a sentence boundary.
3,568 such gaps (1,871 before a quote, 1,697 after) were confirmed this
way; see the module docstring for the full reasoning, including a
follow-up correction that found 818 gaps a first pass had wrongly left
unresolved (context-matching bugs, not false positives -- the earlier
2,750 were all correct, just incomplete).

```sh
uv run semcor-fix-spurious-spacing              # fix data/
uv run semcor-fix-spurious-spacing --dry-run    # preview without writing
```

Idempotent, like `semcor-fix-leading-space`.

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
separately as #24). `src/semcor/em-dash-fixes.yaml` -- kept next to the
script that reads it, since nothing else needs it -- originally listed
1,885 confirmed fixes from that check.

#38 noted a further ~344 candidate `-` tokens #9's exact-neighbour-only
search couldn't place, because the preceding word is multiword-joined
in this corpus's own data (`social_welfare`) but appears as separate
words in `brown-nolines.txt` (`social welfare-`). A second pass with a
chunk-based, underscore-tolerant context-window search (same technique
as #43/#51's) confirmed 299 more (one candidate excluded: a pre-existing,
unrelated `fun_-` token-corruption anomaly, not a real dash), appended to
the same manifest -- 2,184 entries total. No code changes were needed;
the fix mechanism (glue `-` to the preceding word, leave the gap after
untouched) was already exactly right, only the search needed to be
better. This script only applies the manifest, with no runtime NLTK
dependency.

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
is legitimate, unlike #9's em-dash padding).
`src/semcor/thousands-separator-fixes.yaml` -- kept next to the script
that reads it, since nothing else needs it -- lists all 411 confirmed
fixes; this script only applies that manifest, with no runtime NLTK
dependency.

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
`wn30_key` are untouched. `src/semcor/corrupted-ampersand-fixes.yaml`
-- kept next to the script that reads it, since nothing else needs it
-- lists all 78 confirmed `+` character offsets (two tokens have more
than one); this script only applies that manifest, with no runtime
NLTK dependency.

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

`src/semcor/hyphen-compound-merge-fixes.yaml` lists all 289 confirmed
fixes -- kept next to the script that reads it, since nothing else
needs it; generated once, offline, against `nltk.corpus.brown`, this
script has no runtime NLTK dependency and just applies that manifest.

```sh
uv run semcor-fix-hyphen-compound-merge              # apply hyphen-compound-merge-fixes.yaml to data/
uv run semcor-fix-hyphen-compound-merge --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-underscore-hyphen-lexemes`

Corrects underscore-joined lexemes that should be hyphenated (fixes
#43). This corpus recognizes 54 hyphenated compounds (`self-acceptance`,
`spring-training`, `pinch-hitters`, ...) as single WordNet-sensed
multiword lexemes -- correctly, unlike a much larger, separate class #43
also found (two ordinary, individually-tagged words with the hyphen
missing entirely, e.g. `Yankee hatred` for Brown's `Yankee-hatred`; split
off into #46, since 548 of those 549 turn out to already carry a real
sense on one or both sides, needing an individual editorial call to
decide what happens to it on a merge, the same way #24 needed one for
its word-prefixed exclusions -- only 1 has no sense conflict and is fixed
directly, `data/learned/br-j04.yaml`'s `spin spin` -> `spin-spin`) --
but joins these 54 with `_` instead of the real `-` Brown's text has,
e.g. `self_acceptance` where Brown has `self-acceptance`.

Confirmed against `src/semcor/brown-nolines.txt` (the same reference
`semcor-compare-brown-nolines` uses, not `nltk.corpus.brown`, whose own
divergent tokenization this repo stopped trusting as ground truth per
PR #19) via a context-window word search: each underscore-joined token's
parts, rejoined with `-`, had to match a single word at a unique position
in the reference, agreeing with at least one side's worth of the token's
own immediate neighbouring words.

Since `_` and `-` are both one character, this is a same-length,
in-place substitution: only the literal characters at the token's
existing span in `text` change, plus the same `_` -> `-` swap applied to
whatever `lemmas` already has there (not overwritten with the corrected
surface -- lemmatization can differ arbitrarily from the surface, e.g.
surface `re_arguing` pairs with lemma `re-argue`, already hyphenated,
nothing to fix). `oewn_key`/`wn16_key`/`wn30_key` never change: `oewn_key`
encodes a synset ID, not spelling, and WordNet sense keys are spec'd to
always use `_` for multiword lemmas regardless of surface spelling.

`src/semcor/underscore-hyphen-lexeme-fixes.yaml` lists all 54 confirmed
`{file, sentence, index, replacement}` fixes -- kept next to the script
that reads it, since nothing else needs it; generated once, offline, this
script has no NLTK dependency and just applies that manifest.

```sh
uv run semcor-fix-underscore-hyphen-lexemes              # apply underscore-hyphen-lexeme-fixes.yaml to data/
uv run semcor-fix-underscore-hyphen-lexemes --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-doubled-n-contraction`

Fixes negated `can`/`won't` split as `cann't`/`wonn't` (fixes #42). When
a negated `can` or `won't` splits into two tokens, this corpus keeps the
modal's whole spelling as the first token (`can`, `won`) instead of the
correct Penn-Treebank-style split (`ca`, `wo`), then starts the second
token at `n't` anyway -- duplicating the shared `n` and rendering as
`cann't`/`wonn't` (6 characters) once the two flush token spans are
concatenated, instead of Brown's real 5-character `can't`/`won't`.

Purely structural, no Brown/NLTK reference needed: a token pair `(i,
i+1)` is this bug iff the spans are flush, `tokens[i+1]`'s surface is
exactly `n't`, and `tokens[i]`'s surface is exactly `can` or `won` --
166 confirmed instances (111 `can`, 55 `won`) across 89 files, with no
false positives (this naturally excludes two other, unrelated anomalies
noted in #42 -- a `cai` typo, a `could`-lemma-but-different-surface case
-- since neither has surface `can`/`won` at that position).

Token *count* never changes: this just resizes the first token's span
by one character and shifts every later offset in the sentence left by
one to match. `lemmas`/`pos`/every sense-key layer are untouched -- the
lemma content was already correct (including the pre-existing, unrelated
`win`/`will` lemmatization inconsistency for `won't`, out of scope here);
only the surface character span was wrong.

`src/semcor/doubled-n-contraction-fixes.yaml` lists all 166 confirmed
`{file, sentence, index, word}` fixes -- kept next to the script that
reads it, since nothing else needs it; generated once, offline, this
script has no NLTK dependency and just applies that manifest.

```sh
uv run semcor-fix-doubled-n-contraction              # apply doubled-n-contraction-fixes.yaml to data/
uv run semcor-fix-doubled-n-contraction --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-case-mismatch`

Corrects single-token case mismatches against Brown (fixes #13, #28).
Both issues originally claimed a large number of these (152 and 976
respectively) by comparing against `nltk.corpus.brown`, whose own
tokenized rendering isn't reliable ground truth here -- e.g. #28 cited
`nltk.corpus.brown` rendering `"Twilight of Southern Regionalism"` and
`"the Prix de Rome"` in Title Case, but `src/semcor/brown-nolines.txt`
(the reference this repo actually trusts) already has both lowercase,
matching what this corpus had -- not a bug. Corrected via comments on
#13/#28.

Re-deriving the real scope against `brown-nolines.txt` (word-level
diff, same method `semcor-compare-brown-nolines` uses) and verifying
each candidate individually against real context finds only **8
genuine instances**, in both directions (e.g. `savannah` -> `Savannah`,
but also `Same` -> `same`), covering both issues' actual intent -- not
just #13's narrower quote-terminal trigger.

Every fix is a same-length, case-fold-preserving surface substitution:
token *count*/spans are unaffected (case never changes string length).
`lemmas` is updated too, but only where it's currently an exact,
case-sensitive copy of the token's *old* surface -- several of the 8
already have a different or differently-cased lemma (e.g. a generic
`person`, or an already-lowercase sense-form) that's left untouched.
`oewn_key`/`wn16_key`/`wn30_key` never change: sense keys already
ignore surface casing.

`src/semcor/case-mismatch-fixes.yaml` lists all 8 confirmed
`{file, sentence, index, replacement}` fixes -- hand-verified against
`brown-nolines.txt` individually rather than generated by an
unsupervised scan, given how small the set is.

```sh
uv run semcor-fix-case-mismatch              # apply case-mismatch-fixes.yaml to data/
uv run semcor-fix-case-mismatch --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-hyphen-dropped-word-pair`

Restores a hyphen dropped between two ordinary words (fixes #43/#46),
e.g. `term end` -> `term-end`, `ever growing` -> `ever-growing`. #43
found this shape; #46 split off the bulk of it (549 confirmed) because
almost every instance already carries a real WordNet sense on *both*
words, and merging them into one token (as #16/#24 do for other cases)
would force picking which sense survives -- a real editorial call, not
a mechanical one, and checking `external/english-wordnet`'s own source
data confirms none of these compounds has its own sense to fall back on
instead.

That choice turns out to be unnecessary: a hyphen is exactly one
character, the same width as the space it replaces, so the fix is a
single-character substitution *between* the two existing tokens, not a
merge. Both tokens' own spans, `lemmas`, `pos`, and every sense-key
layer are completely untouched; only `text` changes, at exactly the
gap position. This also covers the one hyphenated *range* found
alongside the compound modifiers (`September October` for Brown's
`September-October`, the same shape as `semcor-fix-hyphen-compound-merge`'s
number ranges like `1960-1962`, which this corpus already keeps as
separate tokens either).

Re-verified against `src/semcor/brown-nolines.txt` with a context-window
word search (unique match required, using both sides of the pair and
pulling extra context from neighbouring sentences when the current one
runs out) finds **522 confirmed instances** (508 with a sense on both
words, 14 with a sense on only one, 0 with neither).

`src/semcor/hyphen-dropped-word-pair-fixes.yaml` lists all 522
confirmed `{file, sentence, index}` fixes -- kept next to the script
that reads it; generated once, offline, against `brown-nolines.txt`,
this script has no NLTK dependency and just applies that manifest.

```sh
uv run semcor-fix-hyphen-dropped-word-pair              # apply hyphen-dropped-word-pair-fixes.yaml to data/
uv run semcor-fix-hyphen-dropped-word-pair --dry-run    # preview without writing
```

Idempotent, like the other `fix-*` scripts.

### `semcor-fix-genitive-gap`

Removes a spurious gap before a fused `'s_...` run (fixes #57). This
corpus fuses multi-word proper nouns and idioms into a single token by
joining their words with underscores (e.g. `Fulton_County_Grand_Jury`) --
a deliberate, otherwise-correct convention. But when a genitive or
contraction `'s` ends up fused to the word that *follows* it instead of
staying flush with the word it actually belongs to, the corpus keeps a
stray space or underscore immediately before the `'s`, e.g. `Al 's_Little_
Cafe` for Brown's `Al's Little Cafe`, or (entirely inside one token)
`Fulton_Tax_Commissioner_'s_Office` for Brown's `Fulton Tax Commissioner's
Office`.

Purely structural, no Brown/NLTK reference needed to decide *that* a fix
applies -- no English text ever has a space before `'s` -- though every
candidate was independently confirmed against `brown-nolines.txt` when
building the manifest. This is unrelated to whether the fused run is its
own separate token (the gap sits *between* two tokens, like
`semcor-fix-hyphen-dropped-word-pair`) or sits in the middle of one larger
token (the gap is internal to a single span, like
`Fulton_Tax_Commissioner_'s_Office` above): deleting one character and
shifting every later offset left by one, exactly like
`semcor-fix-doubled-n-contraction`, handles both the same way since token
spans are just integer offsets into `text`. **31 confirmed instances**
across 20 files. `lemmas`/`pos`/every sense-key layer are untouched --
only `text` and the shifted `tokens` offsets change.

`src/semcor/genitive-gap-fixes.yaml` lists all 31 confirmed `{file,
sentence, pos}` fixes (`pos` is the character offset of the space/
underscore to delete) -- kept next to the script that reads it; generated
once, offline, against `brown-nolines.txt`, this script has no NLTK
dependency and just applies that manifest.

```sh
uv run semcor-fix-genitive-gap              # apply genitive-gap-fixes.yaml to data/
uv run semcor-fix-genitive-gap --dry-run    # preview without writing
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

### `semcor-compare-brown-nolines`

Renders `data/` as plain text, in the same document order as
[`brown_nolines.txt`](http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt)
(a plaintext reformatting of the whole 500-file Brown Corpus, used
elsewhere in this repo -- #16/#18 -- as a more faithful reference than
`nltk.corpus.brown`'s own tokenized rendering), and diffs the two,
word-per-line, with `diff`. Meant as a coarse, whole-corpus health check --
a single number to track across PRs -- rather than a replacement for the
issue-by-issue fixes above.

A copy of `brown_nolines.txt` is committed at
`src/semcor/brown-nolines.txt` -- not fetched at run time -- since the
reference text never changes and `www.sls.hawaii.edu` has proven
unreliable to reach from CI runners. Pass `--nolines-file` or set
`$SEMCOR_BROWN_NOLINES_FILE` to compare against a different copy instead.

`brown_nolines.txt` has no markers between its 500 concatenated files, so
locating where each of this corpus's 352 files starts in it is a one-time,
offline step (needs a local `nltk` install, unlike everything else here)
committed as `src/semcor/brown-nolines-offsets.yaml` (kept next to the
script that reads it, rather than at the repo root like the other
`*-fixes.yaml` manifests, since nothing else needs it); regenerate it
only if `brown_nolines.txt` itself changes:

```sh
uv run semcor-compare-brown-nolines --regenerate-offsets
```

```sh
uv run semcor-compare-brown-nolines              # compare data/ against src/semcor/brown-nolines.txt
```

Writes `brown-nolines-ours.txt`, `brown-nolines-reference.txt`, and
`brown-nolines.diff` (all gitignored) and prints a divergent-word-line
count. That count mixes several things together -- genuinely new,
uncatalogued bugs, the accepted #32 dateline/byline/subheadline gap, and
some tokenization-boundary noise (e.g. quote-adjacent spacing in
heavily-quoted sentences that #8 deliberately left unfixed) -- so read
`brown-nolines.diff` itself to see what's actually driving a given number,
rather than the count alone. Exits non-zero if the count is nonzero.

CI runs this on every push and pull request (see
`.github/workflows/verify-brown.yml`) and uploads the generated
`brown-nolines*` files as a build artifact. It's currently red -- see #5
for the tracking issue and its sub-issues for what's been found so far.

### `semcor-verify-brown`

Diffs each `data/<genre>/br-*.yaml`'s merged `text` against the matching
file in [NLTK](https://www.nltk.org/)'s Brown Corpus (`nltk.corpus.brown`,
downloaded on first run), ignoring whitespace and underscores on both
sides -- underscore-joined multiword collocations (`take_place`) and
Brown's own tokenization spacing aren't divergences worth reporting; a
character actually being added, removed, or changed is (see #5, #12).
Produces a Markdown report and exits non-zero if anything diverges. No
longer wired into CI -- `semcor-compare-brown-nolines` above (diffing
against `brown_nolines.txt` rather than NLTK's own tokenized rendering of
Brown, per #16/#18) does that job now -- but still useful standalone for
its per-file Markdown report.

```sh
uv run semcor-verify-brown                        # check data/, report to stdout
uv run semcor-verify-brown -o report.md           # write the report to a file
uv run semcor-verify-brown data/humor             # check one directory/file
```

### `semcor-check-tokens`

A regression tripwire for `tokens` spans silently drifting out of
alignment with `text`: every fix that edits characters in `text` has to
shift every `tokens` span at or after the edit to match, and
`semcor-validate`'s bounds check can't tell a correctly-shifted span from
one that now points at the wrong word. This checks a fixed sample of
tokens (`token-position-samples.yaml`, ~12 per file, picked by sentence
and index rather than absolute offset) against the current `data/*.yaml`.

```sh
uv run semcor-check-tokens              # verify against token-position-samples.yaml
uv run semcor-check-tokens --generate   # regenerate the fixture from current data/
```

Only re-run `--generate` when a change deliberately affects one of the
sampled tokens (e.g. splitting an over-merged token per #10) -- review
the diff to confirm it's the change you intended, the same as reviewing
any snapshot-test update. CI runs the verify mode as its own job (see
`.github/workflows/validate.yml`), independent of `semcor-validate`'s
job so a `oewn_key` drifting out of sync with upstream Open English
Wordnet can't hide a real tokens/text regression behind it.

## License

See [LICENSE.md](LICENSE.md). This resource is derived from the Princeton
WordNet database and is further developed under the Creative Commons
Attribution 4.0 International License; attribution to both Princeton
WordNet and the Open English Wordnet team is required.

## Repository layout

- `data/` — the corpus, one Teanga YAML file per Brown Corpus document,
  grouped by genre.
- `src/` — tooling for working with and updating the corpus.
