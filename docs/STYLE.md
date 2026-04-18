# Documentation style guide (internal)

This file fixes the writing conventions for the technical documentation
in `docs/`. It exists so that twelve chapters written over multiple
sessions maintain a consistent voice and presentation. Deviations from
this guide are bugs to be fixed in review, not stylistic preferences.

## 1. Voice and tense

### 1.1 Default register

The documentation is written in a **rigorously academic register**.
The reader is assumed to be a technically competent colleague — an
ML engineer, a researcher, a senior student — rather than a general
audience.

### 1.2 Grammatical person

The first person (singular or plural) is **not used** in the main
text. Constructions such as *I trained the model* or *we evaluate* are
rewritten in the passive voice or as nominalised agency:

- Reject: *We trained three models.*
- Accept: *Three models were trained.*

- Reject: *I found that the smallest network generalises best.*
- Accept: *The smallest network was found to generalise best.*

- Reject: *We argue that this is due to overfitting.*
- Accept: *This behaviour is argued to result from overfitting of the
  higher-capacity variants to validation-specific regularities.*

### 1.3 Tense

- **Present tense** is used for statements of fact about the method
  and the artefact: *the model uses a stratified split*, *the code
  ships a CLI*.
- **Past tense** is used for the experimental procedure and its
  outcomes: *the models were trained for 20 epochs*, *the baseline
  reached 98.75 % test accuracy*.
- **Future tense** is avoided except in the explicitly
  forward-looking Chapter 11 (*Future Work*).

### 1.4 Voice

The **passive voice** is the default. The active voice is permitted
only where the subject is itself the object of discussion (e.g., *the
data loader shuffles the training split*, *the optimiser updates the
parameters*).

### 1.5 Hedging

Strong claims require empirical support. Where support is partial or
suggestive, the claim is hedged:

- *suggests* — where one observation is consistent with the claim
- *is consistent with* — where the observation does not contradict
- *indicates* — where the observation is stronger than suggestive
- *is argued to* — introduces an interpretative claim

Avoid *prove* and *demonstrate conclusively* in almost all cases.

## 2. Formatting

### 2.1 Markdown flavour

GitHub Flavored Markdown with native math (MathJax) and Mermaid
support. No MDX extensions.

### 2.2 Headings

- Chapter title is H1, once per file.
- Sections are H2.
- Subsections are H3.
- H4 and deeper are discouraged; rewrite into prose paragraphs.

### 2.3 Emphasis

- `*italic*` for introducing terminology on first use and for
  emphasising single words.
- `**bold**` is reserved for structural markers (warnings, table
  headers) and used sparingly.
- All-caps for emphasis is not used.

### 2.4 Lists

- Bullet lists for enumerations without natural ordering.
- Numbered lists for procedures, rankings, or claims being
  individually defended.
- List items end with a full stop if any item is a complete sentence.

### 2.5 Tables

- Every table has a caption immediately preceding it, of the form
  `**Table 7.1.** Headline metrics across the three architectures.`
- Numerical columns are right-aligned with a trailing `:` in the
  header separator.
- Accuracy values are formatted to two decimal places (`98.75 %`),
  loss values to four decimal places (`0.0383`), parameter counts
  with SI prefixes and one decimal (`6.3 M`).

### 2.6 Figures

- Figures are embedded via `![caption](../reports/figures/...)` with
  the caption text functioning as alt text.
- Each figure is accompanied by a caption block **above** the image,
  of the form `**Figure 7.1.** Validation accuracy per epoch for
  `TrafficSignNet`.`
- All figures referenced must exist in the repository. No placeholders.

### 2.7 Code

- Inline Python identifiers are wrapped in backticks: `Trainer`,
  `src/traffic_signs/models/`.
- Multi-line code uses fenced blocks with language annotation:

  ~~~markdown
  ```python
  model = build_model("traffic_sign_net", num_classes=43, image_size=48)
  ```
  ~~~

- Shell commands use the `bash` language tag even on Windows; the
  reader is expected to translate to PowerShell where needed, and
  this is stated once in Chapter 10.

## 3. Mathematical notation

### 3.1 Variables

| Quantity | Symbol |
|---|---|
| Scalar | $x$, $y$, $n$ |
| Vector | $\mathbf{x}$, $\mathbf{y}$ |
| Matrix | $\mathbf{W}$, $\mathbf{X}$ |
| Tensor (rank > 2) | $\mathcal{X}$, $\mathcal{W}$ |
| Scalar function | $f(\cdot)$ |
| Learned function | $f_\theta(\cdot)$ |
| Set | $\mathcal{D}$, $\mathcal{S}$ |
| Probability | $P(\cdot)$ |
| Expectation | $\mathbb{E}[\cdot]$ |

### 3.2 Indices and sizes

- Dataset size: $N$.
- Number of classes: $K$ (GTSRB: $K = 43$).
- Batch size: $B$.
- Image height, width: $H$, $W$.
- Number of channels: $C$ (GTSRB: $C = 3$).

### 3.3 Display equations

Display equations are numbered within the chapter:

```
$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log P_\theta(y_i \mid \mathbf{x}_i)
\tag{4.1}
$$
```

### 3.4 In-line symbols

GitHub's renderer requires `$...$` for inline math. Dollar signs in
ordinary prose are escaped as `\$`.

## 4. Citations

### 4.1 In-line format

Inline citations use the form *Author et al. (Year)* when outside
parentheses, and *(Author et al., Year)* when inside. For two authors:
*Smith and Jones (2020)*.

### 4.2 Reference keys

BibTeX-style keys follow `<firstauthor><year><keyword>`, all lower-case,
no punctuation. Examples:

- `stallkamp2012man`
- `jaderberg2015spatial`
- `ioffe2015batch`

### 4.3 Linking

In the rendered Markdown, in-line author-year citations link to the
corresponding entry in `references.md`:

```markdown
As shown by [Stallkamp et al. (2012)](references.md#stallkamp2012man),
the benchmark ...
```

This creates a clickable cross-reference in the GitHub view.

### 4.4 Direct quotations

Direct quotations are used sparingly. When used, they are limited to
fewer than 15 words and appear in quotation marks with the source and
page number inline.

## 5. Terminology

### 5.1 Canonical names

| Concept | Canonical name | Reject variants |
|---|---|---|
| Three-way data partition | *training / validation / test sets* | train data, val split, hold-out set |
| The official GTSRB test partition | *the test set* (context-clear) or *the official GTSRB test set* | evaluation split, final-test |
| Per-epoch validation score | *validation accuracy* | val acc, dev score |
| Repository package | *the `traffic_signs` package* | the library, the codebase |
| Model-before-CLI | *the configuration* / *the config* | YAML file (by itself), params |
| The architectures | `TrafficSignNet`, `TrafficSignNet-STN`, `DeepTrafficNet` | variations with hyphens, casing |

### 5.2 Capitalisation

- Architecture names are typeset as `code`.
- Named algorithms and architectures from the literature use Title
  Case without code formatting: *Spatial Transformer Network*, *Batch
  Normalisation*, *ReLU*.
- Section headings use sentence case.

### 5.3 Spelling

British English is used throughout: *normalisation*, *regularisation*,
*behaviour*, *labelled*. The one exception is code identifiers, which
retain their original spelling (e.g. `normalize=True` stays as-is).

## 6. Numerical conventions

| Quantity | Format | Example |
|---|---|---|
| Accuracy | `%`, two decimals | `98.75 %` |
| Loss | four decimals | `0.0383` |
| MCC, κ | four decimals | `0.9870` |
| Parameter count | SI prefix, one decimal | `6.3 M` |
| FLOP count | SI prefix, one decimal | `12.5 G` |
| Runtime | unit-adaptive | `18.8 min`, `42 s/epoch` |
| Epochs | integer | `20` |
| Learning rate | exponent | `1 × 10⁻³` |
| Sample count | thousands separator | `12 630` |

Fractional percentages below one: *0.3 %*, not *.3 %*.

The NIST rule for significant digits is respected; values are not
over-reported. A mean of 98.7504 is reported as *98.75 %*, not
*98.7504 %*.

## 7. Build and export

### 7.1 PDF export via pandoc

The twelve chapters plus `references.md` can be exported as a single
PDF:

```bash
pandoc docs/README.md docs/00_*.md docs/01_*.md docs/02_*.md \
       docs/03_*.md docs/04_*.md docs/05_*.md docs/06_*.md \
       docs/07_*.md docs/08_*.md docs/09_*.md docs/10_*.md \
       docs/11_*.md docs/references.md \
       -o traffic-signs-technical-report.pdf \
       --pdf-engine=xelatex \
       --number-sections \
       --toc \
       --toc-depth=3 \
       --metadata title="Traffic-Sign Recognition on GTSRB" \
       --metadata author="Dietmar Ysop" \
       --metadata date="\today"
```

### 7.2 HTML export

For static-site hosting:

```bash
pandoc docs/*.md -s -o traffic-signs.html --toc
```

### 7.3 Quality gates before commit

1. `markdownlint docs/` for structural validity.
2. A spell-check against British English (`hunspell -d en_GB`).
3. Cross-reference check: every citation has a `references.md` entry.
4. Terminology check: search for rejected variants listed in § 5.1.
