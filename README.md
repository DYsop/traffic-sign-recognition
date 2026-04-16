# Traffic-Sign Recognition on GTSRB

[![CI](https://github.com/<your-user>/traffic-sign-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-user>/traffic-sign-recognition/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-46aef7.svg)](https://github.com/astral-sh/ruff)

A reproducible PyTorch implementation that compares three convolutional
architectures on the **German Traffic Sign Recognition Benchmark (GTSRB)**:

| Model | Parameters | Idea |
|---|---:|---|
| `TrafficSignNet` | ~1.1 M | Three-block CNN baseline (Conv + BN + MaxPool + Dropout) |
| `TrafficSignNet-STN` | ~1.3 M | Baseline with a *Spatial Transformer Network* front-end that learns to crop/align the sign |
| `DeepTrafficNet` | ~6.3 M | Five-block CNN with a three-layer MLP head |

The project started as a single Jupyter notebook and was refactored into a
library + CLI + test suite so that third parties can reproduce the results
on their own machines. See [`CHANGELOG.md`](CHANGELOG.md) for the full list
of changes.

---

## Highlights

- **Clean separation** of data loading, model definition, training,
  evaluation and inference.
- **Reproducibility** — deterministic seeds, typed YAML configs, version-pinned
  dependencies, test suite that runs in < 20 s on CPU.
- **Correctness fixes** over the original notebook, including:
  - no more test-set-as-validation data leak,
  - `horizontal_flip` off by default (directional signs),
  - consistent metrics sourced from the same `EvaluationResult` object for
    every model.
- **Three entry points**: `traffic-signs-train`, `traffic-signs-eval`,
  `traffic-signs-predict`.
- **CI**: ruff + mypy + pytest across Python 3.10 / 3.11 / 3.12.

---

## Repository layout

```
traffic-sign-recognition/
├── configs/                     # YAML configs, one per model + default
├── data/                        # dataset goes here (gitignored)
│   └── README.md                # how to obtain GTSRB
├── checkpoints/                 # trained weights (gitignored)
├── reports/
│   ├── figures/                 # training curves, confusion matrices
│   └── metrics/                 # JSON metrics, CSV reports, comparison table
├── notebooks/
│   └── 00_showcase.ipynb        # presentation-only notebook
├── scripts/
│   ├── prepare_test_set.py      # reshape official GTSRB test folder
│   ├── train_all.py             # train + evaluate all three models
│   ├── bootstrap_git.sh         # init + push on Linux/macOS
│   └── bootstrap_git.ps1        # init + push on Windows
├── src/traffic_signs/
│   ├── config.py                # Pydantic schema for experiments
│   ├── cli.py                   # train / eval / predict entry points
│   ├── data/gtsrb.py            # stratified split, transforms, dataloaders
│   ├── models/
│   │   ├── traffic_sign_net.py
│   │   ├── stn.py
│   │   ├── deep_traffic_net.py
│   │   └── registry.py          # name → model mapping
│   ├── training/trainer.py      # one generic trainer for all models
│   ├── evaluation/
│   │   ├── metrics.py           # EvaluationResult dataclass
│   │   └── reports.py           # plots and CSV dumps
│   ├── inference/predict.py     # single-image predictor
│   └── utils/
│       ├── seed.py
│       ├── device.py
│       └── logging_setup.py
├── tests/                       # 15 tests — config, seed, data, model, trainer, eval
├── .github/workflows/ci.yml
├── pyproject.toml               # package metadata + ruff/mypy/pytest config
├── requirements.txt
├── requirements-dev.txt
├── environment.yml
└── README.md
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/<your-user>/traffic-sign-recognition.git
cd traffic-sign-recognition
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate traffic-signs
pip install -e .
```

### 2. Get the data

Download GTSRB and extract it under `data/raw/GTSRB/`. Full instructions in
[`data/README.md`](data/README.md). One-liner to fix the official flat test
folder:

```bash
python scripts/prepare_test_set.py \
    --test-dir data/raw/GTSRB/Final_Test/Images \
    --annotations data/raw/GTSRB/GT-final_test.csv
```

### 3. Train

One model:

```bash
traffic-signs-train --config configs/deep_traffic_net.yaml
```

All three:

```bash
python scripts/train_all.py
```

Outputs are written deterministically:

```
checkpoints/<model>/{best,last}.pt
checkpoints/<model>/class_names.json
reports/figures/<model>/{training_curves,confusion_matrix}.png
reports/metrics/<model>/{training_history,test_metrics}.json
reports/metrics/<model>/classification_report.csv
reports/metrics/comparison.csv
```

### 4. Evaluate

```bash
traffic-signs-eval \
    --config configs/deep_traffic_net.yaml \
    --checkpoint checkpoints/deep_traffic_net/best.pt
```

### 5. Predict on a single image

```bash
traffic-signs-predict \
    --checkpoint checkpoints/deep_traffic_net/best.pt \
    --class-names checkpoints/deep_traffic_net/class_names.json \
    --image path/to/sign.png
```

Emits JSON with the top-5 predictions.

---

## Results

> The comparison table below is regenerated automatically by
> `scripts/train_all.py` and written to `reports/metrics/comparison.csv`.
> Replace these rows with your own numbers after your first full training
> run; the results below are the shape of the table, not numbers from any
> specific run.

| Model | Test Accuracy | Top-5 Accuracy | Test Loss | MCC | Cohen's κ | Epochs trained |
|---|---:|---:|---:|---:|---:|---:|
| TrafficSignNet | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| TrafficSignNet-STN | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| DeepTrafficNet | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Artefacts:**

- `reports/figures/<model>/training_curves.png` — loss and accuracy per epoch
- `reports/figures/<model>/confusion_matrix.png` — normalised CM on the test set
- `reports/metrics/<model>/classification_report.csv` — per-class precision / recall / F1
- `notebooks/00_showcase.ipynb` — interactive walkthrough that loads all of the above

---

## Design decisions

### Why one framework

The original notebook used TensorFlow/Keras for the baseline and PyTorch for
the other two models. That is twice the dependency surface for no analytical
benefit. The Keras baseline was ported to PyTorch with an identical
architecture; the original is still reachable through `git log`.

### Why configs, not flags

Every reproducible experiment needs a single file that captures all the
knobs. `configs/*.yaml` is that file. The CLI only accepts a config path;
hyperparameters are never on the command line.

### Why `horizontal_flip=False` by default

GTSRB contains directional signs — "turn left only", "keep right", "no
entry", various arrow markers. A horizontal flip silently relabels them.
The notebook had flip on; this repo turns it off and documents why.

### Why the test set is not used during training

Using the test set as a validation signal is a textbook data leak. Here the
validation set is carved out of the training folder with a stratified split
(`data.val_split = 0.2`), and the test set is only touched by
`traffic_signs.evaluation`. That restores the integrity of the final test
numbers.

### Checkpoint naming

The notebook wrote `traffic_sign_model_v1.keras`, `…_v2.keras` etc. into
the current working directory — finding the right one after a few runs was
guesswork. This repo always writes
`checkpoints/<model>/best.pt` and `.../last.pt`. If you want versioning, use
git.

---

## Development

```bash
ruff check .
ruff format --check .
mypy
pytest
```

CI runs the same four commands. `pytest` includes an end-to-end trainer test
against a synthetic GTSRB-shaped dataset, so the full pipeline is exercised
without needing the 300 MB download.

---

## Citation

If you use this code, please cite GTSRB (see `data/README.md`) and feel free
to link to this repository.

## Licence

MIT — see [`LICENSE`](LICENSE).
