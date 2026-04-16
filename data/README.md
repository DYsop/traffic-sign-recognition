# Datasets

This repository **does not ship the training data**. The German Traffic Sign
Recognition Benchmark (GTSRB) is distributed by the
[Institut für Neuroinformatik, Ruhr-Universität Bochum](https://benchmark.ini.rub.de/gtsrb_dataset.html)
under its own licence. Please obtain the data yourself.

## Expected directory layout

After extraction, the repository expects the following structure:

```
data/
└── raw/
    └── GTSRB/
        ├── Final_Training/
        │   └── Images/
        │       ├── 00000/       # class 0
        │       │   ├── 00000_00000.ppm
        │       │   ├── 00000_00001.ppm
        │       │   └── ...
        │       ├── 00001/       # class 1
        │       └── ...  (43 classes total: 00000 … 00042)
        └── Final_Test/
            └── Images/
                ├── 00000/       # one subfolder per class
                └── ...
```

This is the standard `torchvision.datasets.ImageFolder` layout: each class
lives in its own subfolder whose name is the class id, zero-padded to five
digits.

## Getting the data

### Option A — Official download (recommended)

1. Visit the [GTSRB download page](https://benchmark.ini.rub.de/gtsrb_dataset.html).
2. Download `GTSRB_Final_Training_Images.zip` (≈ 263 MB).
3. Download `GTSRB_Final_Test_Images.zip` (≈ 88 MB) **and** the test annotations.
4. Extract both under `data/raw/GTSRB/`.

The official test set ships as a flat folder with one CSV that maps file
names to class labels. Convert it into the per-class layout above with:

```bash
python scripts/prepare_test_set.py \
    --test-dir data/raw/GTSRB/Final_Test/Images \
    --annotations data/raw/GTSRB/GT-final_test.csv
```

### Option B — Kaggle mirror

A well-known Kaggle mirror is available — terms of use vary; see the
dataset's page.

## Sub-folders

| Folder | Purpose | In git? |
|---|---|---|
| `data/raw/` | Pristine downloads. Never edited. | No (gitignored) |
| `data/interim/` | Scratch space for intermediate artefacts. | No (gitignored) |
| `data/processed/` | Cleaned, train-ready tensors if any. | No (gitignored) |

Only the `README.md` and empty `.gitkeep` markers are committed; the images
themselves are excluded via `.gitignore`.

## Citation

If you use GTSRB for a publication, please cite:

> Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012).
> *Man vs. computer: Benchmarking machine learning algorithms for traffic
> sign recognition.* Neural Networks, 32, 323–332.
