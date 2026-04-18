# Outline (internal planning document)

This document is the working outline for the twelve technical chapters
of the project documentation. It is internal: readers of the
documentation should not rely on this file. Its purpose is to prevent
scope drift, redundancy between chapters, and terminology inconsistency
across the writing sessions.

For each chapter, the following are fixed **before** writing starts:

- **Goal**: the single thesis the chapter defends.
- **Sections**: ordered subheadings.
- **In-scope**: what is discussed.
- **Out-of-scope**: what is intentionally deferred to another chapter.
- **Figures / tables**: named assets that must appear.
- **Citations**: reference keys from `references.md` that must be used.
- **Target length**: in words.

After a chapter is written, its status in `docs/README.md` changes from
_pending_ to _complete_.

---

## 00. Abstract

**Goal.** Compress the whole document into ~250 words: problem,
approach, headline result, implication, contribution, licence.

**Length.** 250–300 words, single paragraph, no sections.

**Citations.** None — abstract is self-contained.

**Out-of-scope.** Code details, implementation choices, limitations
beyond the headline observation.

---

## 01. Introduction

**Goal.** Establish *why* traffic-sign recognition matters as a
benchmark problem, *what* the three-way comparison tests, and *what*
the reader will take away.

**Sections.**
1. Motivation: autonomous perception and the role of classification
   subsystems.
2. The GTSRB benchmark as a testbed.
3. Research questions — in particular, the trade-off between capacity
   and generalisation at the 98–99 % accuracy regime.
4. Contributions (enumerated, three points).
5. Document structure (forward reference to chapters).

**In-scope.** High-level framing, motivation, statement of contributions.

**Out-of-scope.** Dataset details (→ chap. 3), model architectures
(→ chap. 5), literature (→ chap. 2).

**Figures.** None. Introduction is text-only by convention.

**Citations.** `stallkamp2012man`, `cirecan2012multi`, `lecun1998gradient`,
`goodfellow2016deep`, `krizhevsky2012imagenet`.

**Target length.** ~2 000 words.

---

## 02. Related Work

**Goal.** Place the work in the landscape of prior GTSRB benchmarks,
small-scale image classification, and the specific techniques used
(spatial transformers, deep convolutional classifiers).

**Sections.**
1. GTSRB benchmark: historical performance evolution 2012–2024.
2. Shallow convolutional baselines for traffic signs.
3. Spatial transformer networks: motivation and subsequent uses.
4. Regularisation strategies for small-scale classification
   (dropout, batch normalisation, data augmentation).
5. The reproducibility question in applied deep learning.

**In-scope.** Summarised contributions of at least 15 referenced works,
grouped thematically.

**Out-of-scope.** Original contribution of this project (→ chap. 1, 7).

**Figures.** One summary table: reference, architecture family,
reported test accuracy, year.

**Citations.** `stallkamp2012man`, `cirecan2012multi`, `jaderberg2015spatial`,
`ioffe2015batch`, `srivastava2014dropout`, `lecun1998gradient`,
`he2016deep`, `simonyan2015very`, `szegedy2015going`, `krizhevsky2012imagenet`,
`pineau2021improving`, `bouthillier2021accounting`, `wilson2017marginal`,
`kingma2015adam`, `loshchilov2019decoupled`.

**Target length.** ~3 000 words.

---

## 03. Dataset

**Goal.** Describe GTSRB comprehensively: provenance, structure, class
distribution, capture conditions, and known challenges.

**Sections.**
1. Provenance and licensing.
2. Structure: class hierarchy, file layout, intensity of class
   imbalance.
3. Capture conditions: illumination, occlusion, resolution range.
4. Split definition: official train/test partition.
5. Stratified validation carve-out (our protocol).
6. Known challenges: directional sign ambiguity under augmentation,
   inter-class visual similarity among speed limit signs.

**In-scope.** Descriptive statistics, histograms, sample grids,
per-class counts.

**Out-of-scope.** Training protocol (→ chap. 4, 6), augmentation
strategy (→ chap. 6).

**Figures.**
- F3.1: sample grid, one example per class.
- F3.2: histogram of per-class training counts.
- F3.3: examples of directional signs (motivating the flip argument).

**Tables.**
- T3.1: summary statistics — mean, std, min, max samples per class.

**Citations.** `stallkamp2012man`, `houben2013detection`.

**Target length.** ~2 800 words.

---

## 04. Methodology

**Goal.** Formalise the experimental design: splits, metrics,
randomisation, computing environment.

**Sections.**
1. Experimental questions, restated from chapter 1 in testable form.
2. Data splits: train, validation, test; the case against using the
   test set during model selection.
3. Metrics: accuracy, top-5 accuracy, macro F1, Matthews correlation
   coefficient, Cohen's κ. Justification for reporting multiple.
4. Seed protocol and determinism.
5. Hardware and software environment.
6. Limitations of single-run reporting; bridge to v0.3 multi-seed plan.

**In-scope.** Experimental protocol at the abstract level.

**Out-of-scope.** Specific hyperparameters (→ chap. 6),
architecture-specific training details (→ chap. 6).

**Figures.** None — this is a methodological chapter.

**Tables.**
- T4.1: metric definitions with formulae and interpretation.

**Citations.** `chicco2020advantages`, `powers2011evaluation`,
`bouthillier2021accounting`, `dror2018hitchhiker`, `pineau2021improving`.

**Target length.** ~2 500 words.

---

## 05. Architectures

**Goal.** Describe the three model architectures with sufficient
mathematical and implementational precision for reproduction from
first principles.

**Sections.**
1. `TrafficSignNet` — the three-block baseline.
   - Layer-by-layer specification.
   - Parameter count derivation.
   - Design rationale.
2. `TrafficSignNet-STN` — the spatial-transformer variant.
   - The affine spatial transformer operation, mathematically.
   - Localisation network design.
   - Initialisation strategy (identity transform).
3. `DeepTrafficNet` — the deeper variant.
   - Rationale for five blocks.
   - Fully-connected head design.
4. Comparative parameter and FLOP count table.

**In-scope.** Architecture-level details; tensor shapes; parameter
budgets.

**Out-of-scope.** Training procedure (→ chap. 6), empirical
comparison (→ chap. 7).

**Figures.**
- F5.1: block diagram of `TrafficSignNet`.
- F5.2: block diagram of `TrafficSignNet-STN` with STN highlighted.
- F5.3: block diagram of `DeepTrafficNet`.
- F5.4: STN-predicted transforms visualised on sample inputs (deferred
  to an addendum if time permits).

**Tables.**
- T5.1: parameter and FLOP count per model.

**Citations.** `jaderberg2015spatial`, `ioffe2015batch`,
`srivastava2014dropout`, `lecun1998gradient`, `he2016deep`.

**Target length.** ~3 500 words.

---

## 06. Training Setup

**Goal.** Document every knob that can influence training outcomes
with enough detail to reproduce the reported numbers exactly.

**Sections.**
1. Optimiser choice (Adam / AdamW) and learning-rate schedule
   (ReduceLROnPlateau).
2. Loss function: cross-entropy, rationale for not using label
   smoothing in the baseline.
3. Augmentation pipeline — the case against horizontal flip on GTSRB.
4. Batch size, epoch budget, early-stopping criterion.
5. Checkpointing policy.
6. Per-model deviation table (where the three configurations differ).

**In-scope.** All hyperparameters for all three models, with rationale.

**Out-of-scope.** Architecture (→ chap. 5), metrics (→ chap. 4), results
(→ chap. 7).

**Figures.**
- F6.1: augmentation examples, one image transformed five ways.

**Tables.**
- T6.1: consolidated hyperparameter table for all three models.

**Citations.** `kingma2015adam`, `loshchilov2019decoupled`,
`prechelt1998early`, `smith2019super`, `shorten2019survey`.

**Target length.** ~3 000 words.

---

## 07. Results

**Goal.** Present the raw numbers, the confusion structure, and the
per-class behaviour — without interpretation. Interpretation belongs
in chapter 8.

**Sections.**
1. Headline test-set metrics table.
2. Training trajectories: validation accuracy and loss per epoch.
3. Confusion matrices per model.
4. Per-class precision and recall analysis.
5. Consistent-error classes (those failed by all three models).
6. Divergent-error classes (where the three models disagree).

**In-scope.** Numerical results and visualisations.

**Out-of-scope.** Interpretation, implications, discussion of the
validation-test gap (→ chap. 8).

**Figures.**
- F7.1–F7.3: per-model training curves.
- F7.4–F7.6: per-model normalised confusion matrices.
- F7.7: worst five classes per model, grid.

**Tables.**
- T7.1: headline metrics.
- T7.2: per-class accuracy, three columns.
- T7.3: confusion hot-spots (cells with ≥ 5 misclassifications).

**Citations.** None.

**Target length.** ~3 000 words.

---

## 08. Discussion

**Goal.** Interpret the results of chapter 7 — especially the
counterintuitive finding that the smallest architecture generalises
best.

**Sections.**
1. The validation-test gap across the three models.
2. Why additional capacity does not generalise: the double-descent
   framing vs. the augmentation-specific overfitting framing.
3. Implications for the choice of spatial transformer.
4. Per-class error patterns and what they imply about feature learning.
5. Connection to broader literature on overparameterisation and
   implicit regularisation.
6. Caveats: single-seed reporting, single-run variability.

**In-scope.** Interpretation, hypotheses, connection to literature.

**Out-of-scope.** New experiments (→ chap. 11), engineering rationale
(→ chap. 9).

**Figures.** None new; references to figures from chapter 7.

**Citations.** `nakkiran2021deep`, `belkin2019reconciling`,
`zhang2021understanding`, `neyshabur2017implicit`,
`bouthillier2021accounting`.

**Target length.** ~3 500 words.

---

## 09. Engineering

**Goal.** Document the engineering choices that differentiate a
production-grade repository from a research notebook, and the bugs in
the original code base that were uncovered during refactoring.

**Sections.**
1. From notebook to library: design principles.
2. Three bugs uncovered in the original implementation:
   - test-set-as-validation (data leak),
   - `horizontal_flip=True` (label-corrupting augmentation),
   - inconsistent metric definitions across models.
3. Configuration management: typed YAML via Pydantic.
4. Reproducibility mechanisms: seed control, deterministic cuDNN,
   configuration hashing.
5. Test strategy: unit tests, end-to-end smoke test with synthetic
   data.
6. Continuous integration: Ruff, mypy, pytest, three Python versions.

**In-scope.** Engineering rationale and patterns.

**Out-of-scope.** Scientific results (→ chap. 7, 8).

**Figures.**
- F9.1: repository tree diagram, annotated.

**Tables.**
- T9.1: before/after comparison of notebook vs. library metrics.

**Citations.** `wilson2014best`, `sculley2015hidden`,
`pineau2021improving`, `bouthillier2021accounting`.

**Target length.** ~3 200 words.

---

## 10. Reproducibility

**Goal.** Provide sufficient instructions and artefacts that a
third party can reproduce the reported numbers to within the
reported tolerance.

**Sections.**
1. Software environment: exact Python version, dependency pins.
2. Dataset acquisition and directory structure.
3. Hardware considerations: GPU vs CPU, expected runtimes.
4. Command sequence to reproduce each model's numbers.
5. Expected artefacts after a successful run.
6. Tolerance: determinism across hardware, floating-point drift.
7. Known Blackwell (sm_120) compatibility caveats.

**In-scope.** Step-by-step reproduction protocol.

**Out-of-scope.** Scientific justification (→ other chapters).

**Figures.** None.

**Tables.**
- T10.1: expected runtime per model, GPU vs CPU.

**Citations.** `pineau2021improving`, `gundersen2018state`,
`bouthillier2021accounting`.

**Target length.** ~2 200 words.

---

## 11. Future Work

**Goal.** Delineate the planned next releases and their technical
motivation.

**Sections.**
1. Version 0.3.0: ensemble of the three baseline models.
   - Scientific motivation: divergent error patterns observed in
     chapter 7.
   - Engineering plan.
2. Version 0.4.0: `TrafficSignNetV2`.
   - Targeted techniques: label smoothing, 1-cycle LR, class-weighted
     loss, test-time augmentation.
   - Ablation protocol.
3. Open limitations and research questions.

**In-scope.** Planned next steps, with justification.

**Out-of-scope.** Speculation beyond v0.4.

**Figures.** None.

**Citations.** `dietterich2000ensemble`, `hansen1990neural`,
`szegedy2016rethinking`, `smith2019super`, `howard2020fastai`.

**Target length.** ~2 000 words.

---

## Cross-chapter commitments

### Terminology freeze

The following canonical terms are used throughout. Variations are
errors to be caught in review.

| Canonical term | Reject these variants |
|---|---|
| *training set* | training data, train split |
| *validation set* | dev set, held-out (unqualified), val split |
| *test set* | evaluation set, hold-out set |
| *stratified split* | class-balanced split, proportional split |
| *test accuracy* | test acc, accuracy on test |
| *top-5 accuracy* | top-5 acc, top-5 |

### Figure numbering

Figures are numbered within chapter as `F<chapter>.<index>`, e.g.
`F7.2` is the second figure of chapter 7. Table numbering uses the
same scheme with the `T` prefix.

### Citation style

Inline citations use the format *Author et al. (Year)* for readability.
The corresponding entry in `references.md` uses BibTeX keys of the
form `<firstauthor><year><keyword>`, e.g. `stallkamp2012man`.

### Word budget

Total budget: approximately **31 000 words** across twelve chapters.
This corresponds to a short research monograph or a substantial
technical report. Individual chapters may deviate by up to 20 %;
larger deviations trigger a scope review.

### Asset dependencies

Chapters 3, 7 depend on having the dataset locally. Chapters 5, 9
depend on source tree being accessible. Remaining chapters are
self-contained in terms of text.
