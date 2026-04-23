# Traffic-Sign Recognition on GTSRB — Technical Documentation

This directory contains the extended technical documentation accompanying
the `traffic-sign-recognition` repository. The main `README.md` at the
repository root provides a concise overview; the chapters below go into
methodological detail, results analysis, engineering rationale, and
related literature.

The documentation is written as a self-contained technical report. It
can be read linearly, consulted chapter by chapter, or exported as a
single PDF via `pandoc` (see `docs/STYLE.md` for instructions).

## Abstract

The German Traffic Sign Recognition Benchmark (GTSRB) remains a useful
testbed for evaluating small-to-medium convolutional neural network
architectures under realistic multi-class imbalance, intra-class
variability, and capture-condition noise. Three architectures of
increasing structural complexity are trained and evaluated on this
benchmark: a three-block baseline convolutional network with 630 k
parameters (`TrafficSignNet`), a variant augmented with a spatial
transformer front-end for geometric canonicalisation
(`TrafficSignNet-STN`, 1.3 M parameters), and a deeper five-block
network with 6.3 M parameters (`DeepTrafficNet`). Under matched
training protocols and seeds, the smallest model achieves the highest
test accuracy (98.75 %, MCC 0.987), outperforming both higher-capacity
variants by 0.7 to 0.8 percentage points. A gap of 1.1–1.8 pp between
validation and test accuracy is observed in all three models, with the
larger architectures generalising less well across this split despite
nominally higher validation performance. These results motivate a
subsequent release (v0.3.0) in which ensembling and targeted
regularisation techniques are applied to the existing models before any
architectural changes are considered. The full experimental protocol,
configuration files, and evaluation artefacts are released under the
MIT licence.

## Table of contents

| # | Chapter | Status |
|---|---|---|
| 00 | [Abstract](00_abstract.md) | _pending_ |
| 01 | [Introduction](01_introduction.md) | _complete_ |
| 02 | [Related Work](02_related_work.md) | _complete_ |
| 03 | [Dataset](03_dataset.md) | _complete_ |
| 04 | [Methodology](04_methodology.md) | _complete_ |
| 05 | [Architectures](05_architectures.md) | _complete_ |
| 06 | [Training Setup](06_training_setup.md) | _pending_ |
| 07 | [Results](07_results.md) | _pending_ |
| 08 | [Discussion](08_discussion.md) | _pending_ |
| 09 | [Engineering](09_engineering.md) | _pending_ |
| 10 | [Reproducibility](10_reproducibility.md) | _pending_ |
| 11 | [Future Work](11_future_work.md) | _pending_ |
| —  | [References](references.md) | _seeded_ |
| —  | [Style Guide](STYLE.md) | _internal_ |
| —  | [Outline](OUTLINE.md) | _internal_ |

## Conventions

- Mathematical notation follows standard conventions used in Goodfellow
  et al. (2016). Scalars are italic lower-case ($x$); vectors are bold
  lower-case ($\mathbf{x}$); matrices are bold upper-case ($\mathbf{W}$).
- Inline code refers to Python identifiers from the accompanying source
  tree, rooted at `src/traffic_signs/`.
- Figures reproducible from committed artefacts are linked rather than
  embedded where possible, to reduce repository size while preserving
  build reproducibility.
- Citations follow the author-year convention, e.g. *Stallkamp et al.
  (2012)*, with full bibliographic entries listed in `references.md`.

## Citation

If this documentation or the accompanying code informs further work,
please cite the repository and the underlying dataset:

> Ysop, D. (2026). *Traffic-sign recognition on GTSRB: A reproducible
> comparison of three convolutional architectures.* Version 0.2.0.
> https://github.com/DYsop/traffic-sign-recognition

Together with the original dataset reference:

> Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012). Man
> vs. computer: Benchmarking machine learning algorithms for traffic
> sign recognition. *Neural Networks*, 32, 323–332.

## Licence

Source code is released under the MIT licence; see `LICENSE` at the
repository root. The documentation text is released under the
Creative Commons Attribution 4.0 International licence (CC BY 4.0).
