# 9. Engineering

The scientific content of this report — the methodology of
chapters 4 and 6, the architectures of chapter 5, and the results
reported and interpreted in chapters 7 and 8 — rests on a software
artefact whose engineering choices are the subject of the present
chapter. The artefact is released as an open-source Python package
under the MIT licence, with a supporting continuous-integration
pipeline, a test suite, and a configuration system that together
are intended to make the experimental claims of this document
independently reproducible.

The chapter is organised into six sections. The first documents
the transition from a single-file research-prototype implementation
to a package-based library, and the design principles that guided
the transition. The second enumerates three defects uncovered in
the original research-prototype code base during the refactoring
process, each of which would have materially affected the reported
numbers had it remained undetected. The third describes the
configuration-management system. The fourth documents the
reproducibility mechanisms, including seed control, deterministic
cuDNN configuration, and checkpoint-hash verification. The fifth
describes the test strategy. The sixth documents the
continuous-integration pipeline.

## 9.1 From research prototype to library

The research-prototype code from which the present artefact derives
was a single Jupyter notebook of approximately 1 100 lines,
containing all three model definitions, a training loop for each,
an evaluation section, and a comparison summary. The notebook
format is a productive one for exploratory research; it is
decidedly unsuitable as a basis for reproducible third-party use.
Three properties of the original notebook militate against
reproducibility, each of which motivates a specific engineering
choice in the refactored package.

First, the notebook mixed configuration with logic. Hyperparameter
values, file system paths, and GPU/CPU selection were interspersed
with the code they governed; changing a batch size required editing
a line of Python rather than altering a declarative artefact. The
refactored package separates these concerns: hyperparameters and
paths are declared in YAML configuration files under `configs/`,
and the code reads these declarations through a typed schema. The
rationale for this separation, which is a specific instance of the
general architectural advice of [Wilson et al.
(2014)](references.md#wilson2014best), is that experiments that
can be described declaratively can also be versioned, diffed, and
exchanged, whereas experiments that are embedded in code become
confused with the implementations they parameterise.

Second, the notebook contained three parallel training loops, one
per model, with partially duplicated code. This duplication, a
common feature of research prototypes, made it difficult to ensure
that the three models were trained under identical conditions; any
attempt to modify the training protocol required parallel edits in
three places, with the attendant risk of divergence. The refactored
package implements a single `Trainer` class in
`src/traffic_signs/training/trainer.py`, which is instantiated with
a configuration object and operates on any model that conforms to
the common `nn.Module` interface. Each of the three architectures
is trained by an identical code path differing only in the model
it wraps. This design choice corresponds to the principle, often
associated with [Sculley et al.
(2015)](references.md#sculley2015hidden), that the most dangerous
form of technical debt in machine-learning systems is the debt
accumulated in the training code itself, where small
inconsistencies between training runs propagate into incomparable
results.

Third, the notebook's output — metrics, plots, saved models — was
written to the current working directory with ad-hoc naming. A
model saved after a promising training run was named
`traffic_sign_model_v1.keras`; after a second run, the first model
was either overwritten or re-saved alongside as `..._v2.keras`,
with no association between the saved file and the configuration
under which it was trained. The refactored package imposes a
deterministic output layout: training writes to
`checkpoints/<model_name>/{best,last}.pt` and
`reports/{figures,metrics}/<model_name>/`, where `<model_name>` is
fixed by the configuration and is invariant across runs of the same
configuration. A separate mechanism, described in § 9.4, associates
each saved checkpoint with a hash of the configuration under which
it was produced, such that checkpoints cannot be silently loaded
under mismatched configurations.

The aggregate effect of these three design choices is a package
whose structure is shown in Figure 9.1 and which supports the
separation of concerns — configuration from logic, training from
evaluation, models from training — that the original notebook did
not.

**Figure 9.1.** Repository directory structure of the refactored
package.

```text
traffic-sign-recognition/
├── configs/                     YAML configurations, one per model
├── data/                        dataset location (gitignored)
├── checkpoints/                 trained weights (gitignored)
├── reports/
│   ├── figures/                 training curves, confusion matrices
│   └── metrics/                 JSON metrics, CSV reports
├── notebooks/
│   └── 00_showcase.ipynb        presentation-only notebook
├── scripts/
│   ├── prepare_test_set.py      one-off: reshape GTSRB test folder
│   └── train_all.py             train and evaluate all three models
├── src/traffic_signs/
│   ├── config.py                Pydantic schema
│   ├── cli.py                   train / eval / predict entry points
│   ├── data/gtsrb.py            stratified split, transforms
│   ├── models/                  three architectures + registry
│   ├── training/trainer.py      unified training loop
│   ├── evaluation/              metrics and report generation
│   ├── inference/predict.py     single-image predictor
│   └── utils/                   seeding, logging, device selection
├── tests/                       pytest suite
├── docs/                        this documentation
├── .github/workflows/ci.yml     CI pipeline
└── pyproject.toml               package metadata + tool configs
```

## 9.2 Three defects uncovered during refactoring

The conversion of the research-prototype notebook into the
refactored package uncovered three defects in the original
implementation. Each was silent in the sense that the notebook
executed without error and produced plausible numerical output;
each would nonetheless have materially affected the reported
numbers had it remained undetected. The three are enumerated in
decreasing order of severity.

### 9.2.1 Test-set exposure during training

The most consequential defect was the use of the official GTSRB
test partition as the validation signal during training. In the
original notebook, the data-loading code instantiated two
`DataLoader` objects — one over the training partition and one
over the test partition — and used the second as the epoch-level
validation set for early stopping, learning-rate scheduling, and
model selection. The consequence is that every number subsequently
reported as a *test accuracy* was in fact a number the model had
been optimised against indirectly, through the intermediate
mechanisms of early stopping and checkpoint selection. The
reported numbers were consequently overoptimistic by an unknown
but systematically non-zero amount.

The refactored data-loading code in
`src/traffic_signs/data/gtsrb.py` constructs a three-way partition
as documented in § 4.2: the test set is held inviolable by the
training pipeline, and a validation set is carved from the
training partition by stratified splitting. This is the protocol
change that [Pineau et al.
(2021)](references.md#pineau2021improving) identifies as one of
the most common corrections required to make reported
machine-learning results trustworthy. Its application here was
necessary and sufficient to make the test-set numbers reported in
chapter 7 reflect genuine generalisation rather than indirect
optimisation.

### 9.2.2 Label-corrupting augmentation

The augmentation pipeline of the original notebook included
`RandomHorizontalFlip` as a default transformation applied to
training samples with probability 0.5. This choice is idiomatic
for many image-classification tasks and is the default
recommendation in numerous tutorials. It is, however, incorrect
for GTSRB, for the reasons enumerated in § 3.6 and § 6.3.1: the
benchmark contains classes — turn-right and turn-left, no-entry,
keep-right and keep-left, dangerous-curve-left and
dangerous-curve-right — whose semantic meaning is
orientation-dependent, such that a horizontal flip produces an
image that, if still interpreted as its original class, is
incorrectly labelled.

The quantitative consequence, estimated in § 6.3.1, is that
approximately 17 % of training samples belong to classes for which
horizontal flip is label-corrupting; applying the flip with
probability 0.5 introduces approximately 8.5 % label noise into
the training signal. Label noise of this magnitude is not
catastrophic — the classifier can compensate in part by treating
the affected classes as more variable than they are — but it is
far from neutral. The refactored augmentation pipeline, documented
in § 6.3, omits horizontal flipping entirely. Configurations are
available that re-enable it for users who wish to experiment with
class-aware flipping, but the default is the label-safe
configuration.

### 9.2.3 Inconsistent metric definitions

The third defect was more subtle and concerns the comparison table
between architectures. In the original notebook, the three
training loops each maintained per-epoch validation metrics, but
the specific definition of validation accuracy differed slightly
between loops. One training loop computed accuracy as a
torchmetrics `Accuracy` object with default parameters; another
computed it as the ratio of correct predictions to total
predictions in the validation batch, with minor differences in how
the reduction was performed; the third computed accuracy as a
running average across validation batches, which is arithmetically
equivalent to the second only if all batches are of the same size
(they were not, for the final partial batch of each epoch).

The absolute differences produced by these three definitions were
small — on the order of 0.1 pp at convergence — but systematic:
the comparison table that the notebook produced in its final cell
was not comparing like with like. The refactored package consolidates
metric computation into the `EvaluationResult` dataclass in
`src/traffic_signs/evaluation/metrics.py`, such that all three
architectures are evaluated by an identical code path on identical
data with identical metric definitions. Every number in Table 7.1
derives from this common code path.

The three defects are summarised in Table 9.1. The table also
records, where quantifiable, the magnitude of the correction each
defect required.

**Table 9.1.** Defects uncovered during refactoring and their
magnitude.

| Defect | Location in original | Correction | Magnitude of effect |
|---|---|---|---|
| Test set used as validation | data loading | Stratified validation carve-out from training partition | Unquantifiable but non-zero overstatement |
| `RandomHorizontalFlip` applied | augmentation | Flip removed by default | ≈ 8.5 % of training samples affected |
| Inconsistent accuracy definitions | three training loops | Unified `EvaluationResult` code path | ≈ 0.1 pp per-metric drift |

None of these defects was a unique failure of the original
research-prototype code base; each represents a class of defect
that is well documented in the literature on machine-learning
reproducibility and against which the recommendations of [Wilson
et al. (2014)](references.md#wilson2014best) and [Sculley et al.
(2015)](references.md#sculley2015hidden) are collectively directed.
The value of the refactoring exercise lies not in identifying
unique errors but in establishing a structural mechanism — typed
configurations, unified training paths, centralised metric
computation — that prevents the recurrence of such defects.

## 9.3 Configuration management

Experimental hyperparameters are declared in YAML files under the
`configs/` directory. Each configuration is a self-contained
description of a single experiment: model name, data paths, training
parameters, augmentation settings, seed, and output directories.
Configurations are read through a typed schema defined in
`src/traffic_signs/config.py` using Pydantic, which validates the
content of each field against its declared type at the point of
loading rather than at the point of use.

The choice of Pydantic over plain YAML parsing or over an
ad-hoc dictionary-based interface is motivated by two
considerations. First, typed validation at load time means that a
malformed configuration fails immediately, with a clear error
message identifying the specific field and the expected type,
rather than failing deep inside the training loop with an opaque
traceback. Second, the schema itself functions as documentation of
the configuration space; a user confronting an unfamiliar
configuration file can consult `config.py` to understand which
fields are required, which are optional, and what values they
admit.

A secondary benefit of the typed configuration system is that it
enables configuration-hash stamping on checkpoints. At training
time, the configuration object is serialised to a canonical form
and hashed; the resulting hash is stored alongside the model
weights in the checkpoint file. At inference time, the predictor
verifies that the checkpoint hash matches the configuration hash
under which inference is being requested, and raises an explicit
error if they differ. This mechanism prevents the silent mismatch
between trained model and inference configuration that was
possible under the original notebook's implicit configuration
convention, in which the trained weights carried no information
about the conditions under which they had been produced.

## 9.4 Reproducibility mechanisms

Beyond the configuration system, three specific mechanisms
contribute to the numerical reproducibility of the reported
results.

**Seed control.** A single integer seed parameter in the
configuration file governs every pseudo-random operation in the
training pipeline: Python's `random` module, NumPy's global
generator, PyTorch's CPU and CUDA generators, and the
`PYTHONHASHSEED` environment variable that fixes dictionary
iteration order. The seed is applied in a single place, the
`utils/seed.py` module, which is called before any other
randomness-sensitive operation. The shipped configurations all use
the value $s = 42$; alternative seed values produce alternative
trajectories, and the planned multi-seed study of v0.3.0 (§ 11)
relies on this single point of control.

**Deterministic cuDNN configuration.** The cuDNN backend is
configured in deterministic mode, with its autotuning disabled.
This configuration trades a small reduction in throughput — empirically
between 10 % and 20 % on the architectures examined here — for
bit-exact reproducibility of the gradient computation across runs
on identical hardware. The configuration is applied by the same
`utils/seed.py` module, in its `set_seed` function, and can be
overridden by setting the environment variable
`TRAFFIC_SIGNS_DISABLE_CUDNN_DETERMINISTIC=1` for users who prefer
throughput to bit-exact reproducibility. The default is the
reproducible configuration.

**Checkpoint provenance.** Every checkpoint written during training
includes, in addition to the model and optimiser state dictionaries,
a provenance block that records the configuration hash, the random
seed, the epoch number, the validation metrics at that epoch, and
the version of the `traffic_signs` package under which the
checkpoint was produced. The provenance block is written in a
structured format that can be inspected from the checkpoint file
without loading the model weights, which permits rapid triage of
a collection of checkpoints without incurring the overhead of GPU
or CPU memory allocation.

The three mechanisms together are intended to close the
reproducibility gap that [Bouthillier et al.
(2021)](references.md#bouthillier2021accounting) characterises as
the dominant source of non-reproduced results in contemporary
benchmarks. Seed control addresses within-hardware reproducibility;
deterministic cuDNN configuration addresses determinism of the
gradient computation; checkpoint provenance addresses the silent
configuration drift that can otherwise accumulate across multiple
training runs.

## 9.5 Test strategy

The accompanying test suite, under `tests/`, contains fifteen
passing tests as of v0.2.0, organised into four categories.

**Unit tests for configuration loading.** These tests verify that
each shipped YAML configuration loads cleanly through the Pydantic
schema, that invalid configurations are rejected with appropriate
error messages, and that configuration defaults match the values
documented in § 6.6.

**Unit tests for model construction.** These tests verify that
each of the three architectures can be instantiated, accepts an
input tensor of the expected shape, produces an output tensor of
the expected shape, and has the documented parameter count.
Parameter-count tests are particularly valuable because they
quickly catch silent architectural drift: any change to a model
definition that affects the parameter count — intended or not —
fails the corresponding test.

**Data-pipeline tests.** These tests verify that the stratified
validation carve-out produces balanced sub-folds, that the
augmentation pipeline applies the expected transformations in the
expected order, and that the test set is not accessed by any
component of the training pipeline. The last of these is a
protocol-enforcement test: an attempt from within the training
code path to load or inspect the test partition is caught at the
point of data-loader instantiation.

**End-to-end smoke test.** A single integration test exercises the
full training pipeline against a synthetic GTSRB-shaped dataset —
43 classes, small images, a few dozen training samples per class —
for two epochs, and verifies that the resulting checkpoint, metrics,
and reports are produced. The synthetic dataset is constructed on
the fly by the test fixture and requires no download. The smoke
test runs in under twenty seconds on CPU.

The test suite is intentionally not exhaustive. A sufficient test
suite would require property-based testing of the augmentation
pipeline, statistical tests of the stratified split, and
hardware-level tests of cuDNN determinism that are beyond the
scope of a community-sourced open-source project. What the suite
does guarantee is that the core contracts of the package —
configuration validity, model shapes, data-partition discipline,
end-to-end training — are checked on every commit and cannot
silently regress.

## 9.6 Continuous integration

The `.github/workflows/ci.yml` pipeline runs four checks on every
commit and pull request to the `main` branch: Ruff linting for
code-style compliance, Ruff formatting verification, MyPy type
checking, and pytest against the full test suite. The pipeline
executes on three Python versions — 3.10, 3.11, and 3.12 — against
the Ubuntu 22.04 GitHub-hosted runner image. CPU-only PyTorch wheels
are used to avoid the overhead of GPU provisioning in the CI
environment.

The choice of these four checks, and specifically the choice to
include MyPy type checking alongside the more conventional
linting and testing, reflects the position that machine-learning
code bases in which configuration shape and data-tensor shape are
both variable are particularly susceptible to the class of bug in
which an incorrect type propagates through multiple functions
before failing at a surface far from the cause. Type checking
catches this class of bug at the point of introduction rather than
at the point of failure; the modest overhead of maintaining type
annotations is a small price for the corresponding reduction in
diagnostic effort.

A secondary consideration for the CI pipeline is its role as a
protocol-enforcement mechanism. Any pull request that modifies the
training loop, the augmentation pipeline, or the data-partition
logic is required to pass the corresponding unit and integration
tests before it can be merged; a pull request that regresses the
test-set-isolation property, for example, fails the CI pipeline
and is blocked from merging. The CI pipeline is therefore not
merely a quality mechanism but a mechanism for the enforcement of
the methodological commitments documented in § 4.2 across the
lifetime of the project.

The engineering choices documented in this chapter are, in
aggregate, the mechanism by which the numerical claims of
chapters 7 and 8 become claims rather than assertions. Each
choice has been made against a specific alternative and documented
with reference to the general machine-learning-engineering
literature. The reproducibility protocol that these choices
support is the subject of the next chapter.
