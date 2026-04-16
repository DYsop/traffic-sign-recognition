# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — Unreleased

### Added
- Typed, YAML-backed configuration (`src/traffic_signs/config.py`).
- Deterministic seeding helper (`src/traffic_signs/utils/seed.py`).
- Centralised `rich`-based logging (`src/traffic_signs/utils/logging_setup.py`).
- Stratified train/val split that **never** uses the test set during training.
- Unified `Trainer` with early stopping, checkpointing and JSON history.
- `EvaluationResult` dataclass — identical metrics (test loss, test acc,
  top-5, MCC, κ) for every model, eliminating the notebook's apples-to-oranges
  comparison table.
- CLI entry points: `traffic-signs-train`, `traffic-signs-eval`,
  `traffic-signs-predict`.
- Full test suite (`pytest tests/` → 15 passing tests, end-to-end trainer smoke test).
- GitHub Actions CI: ruff + mypy + pytest on Python 3.10 / 3.11 / 3.12.
- Community files: `LICENSE` (MIT), `CONTRIBUTING.md`, `SECURITY.md`,
  `CHANGELOG.md`, `data/README.md`.

### Changed
- Keras baseline model ported to PyTorch to remove the dual-framework dependency.
- Checkpoint naming is deterministic (`checkpoints/<model>/best.pt`,
  `.../last.pt`) instead of auto-incrementing version numbers in CWD.
- `horizontal_flip` defaults to **OFF** — the notebook default was a silent
  labelling bug on directional signs ("no entry", arrows, …).

### Removed
- Runtime `input()` prompts for missing-library installs.
- Duplicate import blocks across model cells.
- Hard-coded Windows paths.
- `convert_to_percentage` heuristic that masked inconsistent metric units.

## [0.1.0] — Initial notebook
- Original three-model exploration in a single Jupyter notebook
  (TrafficSignNet / TrafficSignNet-STN / DeepTrafficNet).
