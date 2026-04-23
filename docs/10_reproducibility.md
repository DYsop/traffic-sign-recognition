# 10. Reproducibility

The claims advanced in chapters 7 and 8 rest on a specific training
and evaluation procedure, a specific software environment, and a
specific data source. An independent reader in possession of the
same data source, a compatible software environment, and the
configuration files distributed with the repository should be able
to reproduce the numerical results reported in Table 7.1 within a
narrow tolerance. The present chapter documents the procedure
under which this reproduction can be performed.

The reproducibility protocol described here concerns numerical
reproduction — the regeneration of the specific accuracy,
cross-entropy loss, Matthews correlation coefficient, and confusion
matrix values reported in chapter 7. It does not address
methodological reproduction, in which the same scientific claims
are validated by an independently designed experiment; the latter
lies beyond the scope of the repository release and would form
part of a broader programme of replication work in the sense of
[Pineau et al. (2021)](references.md#pineau2021improving).

The chapter is organised into six sections. The first documents
the software environment required. The second describes the data
acquisition procedure. The third specifies the command sequence
required to reproduce the reported training and evaluation runs.
The fourth enumerates the expected outputs. The fifth documents
the hardware tolerances within which numerical reproduction is
expected to hold. The sixth addresses known failure modes and
their diagnosis.

## 10.1 Software environment

The repository targets Python 3.10, 3.11, and 3.12 on Ubuntu 22.04
as the reference environment and is verified against all three
versions in the continuous-integration pipeline (§ 9.6). It is
known to work on Windows 11 and macOS 14 with the same Python
versions, with the caveats documented in § 10.5 and § 10.6.

Two dependency-specification files are provided at the repository
root. The file `requirements.txt` lists the runtime dependencies
with loose lower-bound pins, suitable for installation into a
generic Python environment. The file `requirements-dev.txt` adds
the dependencies required for running the test suite and the
development tooling (Ruff, MyPy, pytest). An alternative
`environment.yml` is provided for users of the Conda package
manager; it specifies the same core dependencies in the Conda
dependency-specification format.

The canonical installation procedure, starting from a freshly
cloned repository and assuming Python 3.11 is the active
interpreter, is the following:

```bash
git clone https://github.com/DYsop/traffic-sign-recognition.git
cd traffic-sign-recognition
python -m venv .venv
source .venv/bin/activate              # Linux / macOS
# .venv\Scripts\Activate.ps1            # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

The `.[dev]` extras install both runtime and development
dependencies. The `-e` flag installs the package in editable mode,
such that changes to the source code are reflected without
reinstallation. The installation completes in approximately two
minutes on a reasonable internet connection; the two components
of this duration are the PyTorch wheel (approximately 770 MB on
Linux, similar on Windows and macOS) and the remaining
dependencies (approximately 40 MB in aggregate).

Verification of the installation is performed by running the full
test suite:

```bash
pytest tests/ -q
```

A successful verification reports fifteen passing tests and no
warnings from Ruff or MyPy if those are run separately. A failing
test at this stage indicates an environment mismatch rather than a
repository bug and warrants diagnosis before proceeding.

## 10.2 Data acquisition

The GTSRB dataset is distributed by its curators under a
non-commercial research licence and is not redistributed with the
repository. Users are expected to download the dataset separately.
The canonical acquisition procedure is documented in
`data/README.md` and is reproduced in summary here.

The dataset is available from the official
Institut für Neuroinformatik (INI) distribution at Ruhr-University
Bochum. Two archive files are required: the *Training Images*
archive (approximately 263 MB), which contains the 39 209 training
images organised into forty-three class-named subdirectories, and
the *Test Images* archive (approximately 88 MB), which contains
the 12 630 test images in a flat directory together with a CSV
file providing the class annotations. After download, both
archives are to be extracted into the `data/raw/` subdirectory of
the repository root, producing the directory structure
`data/raw/GTSRB/Final_Training/` and `data/raw/GTSRB/Final_Test/`
respectively.

The test-image directory requires one post-processing step to
align with the `torchvision.datasets.ImageFolder` interface used
by the data loader. The flat test directory is to be rearranged
into class-named subdirectories matching the training layout. A
helper script `scripts/prepare_test_set.py` performs this
rearrangement:

```bash
python scripts/prepare_test_set.py \
    --source data/raw/GTSRB/Final_Test \
    --destination data/processed/test
```

The script reads the CSV annotation file, creates the 43
class-named subdirectories under `data/processed/test/`, and
copies (not moves) each image into the subdirectory corresponding
to its class label. The resulting `data/processed/test/` directory
is the test partition used by the evaluation pipeline; the original
`data/raw/GTSRB/Final_Test/` directory is preserved and may be
deleted by the user if disk space is a concern.

## 10.3 Training and evaluation

The full training and evaluation procedure for all three
architectures is performed by a single command:

```bash
python scripts/train_all.py
```

The script reads the three configuration files
`configs/traffic_sign_net.yaml`,
`configs/traffic_sign_net_stn.yaml`, and
`configs/deep_traffic_net.yaml` in sequence, trains each
architecture under the hyperparameters documented in § 6.6, saves
the resulting checkpoints to
`checkpoints/<model_name>/{best,last}.pt`, evaluates the
best-on-validation checkpoint on the test partition, and writes
the resulting metrics to
`reports/metrics/<model_name>/{test_metrics.json,
training_history.json, classification_report.csv}`. The training
curves and confusion matrices are written to
`reports/figures/<model_name>/{training_curves.png,
confusion_matrix.png}`.

Alternatively, each architecture may be trained individually by
invoking the command-line interface directly:

```bash
traffic-signs train --config configs/traffic_sign_net.yaml
traffic-signs eval  --config configs/traffic_sign_net.yaml
```

The `traffic-signs` executable is installed as a console script at
package installation time and is available on the shell `PATH` as
long as the virtual environment is active. The `train` and `eval`
subcommands correspond to the two phases of the experiment and can
be executed independently; `eval` requires that a `best.pt`
checkpoint exist at the location implied by the configuration.

Total wall-clock time for the full three-model training and
evaluation is approximately 48 minutes on the reference CPU
configuration documented in § 10.5. On a CUDA-capable GPU of
comparable class — for example an NVIDIA T4 or V100 — the same
procedure completes in approximately 6 to 10 minutes. The GPU
speed-up is modest because the architectures are small by
contemporary standards and because the batch size of 64 does not
saturate GPU throughput; the speed-up would be substantially
larger at higher batch sizes or on deeper architectures.

## 10.4 Expected outputs

The `train_all.py` script, run to completion, writes files to two
directories: `checkpoints/` and `reports/`. The specific contents
are enumerated below, against which a user can verify that the
reproduction has completed successfully.

Under `checkpoints/`, one subdirectory per model is created,
containing two files each:

- `best.pt` — the model and optimiser state at the epoch of lowest
  validation loss, together with the provenance block described
  in § 9.4
- `last.pt` — the model and optimiser state at the end of the
  20-epoch training budget

Under `reports/metrics/`, one subdirectory per model is created,
containing three files each:

- `training_history.json` — per-epoch training and validation
  metrics for the full 20-epoch trajectory
- `test_metrics.json` — headline metrics on the test partition
  (accuracy, top-5 accuracy, loss, MCC, κ) and the full
  confusion matrix
- `classification_report.csv` — per-class precision, recall, F1,
  and support

A consolidated `reports/metrics/comparison.json` and corresponding
`comparison.csv` collate the headline metrics across the three
models in a form amenable to direct tabular inspection.

Under `reports/figures/`, one subdirectory per model is created,
containing two files each:

- `training_curves.png` — four-panel figure showing training and
  validation accuracy and loss per epoch
- `confusion_matrix.png` — normalised confusion matrix on the test
  set

The values committed to the `reports/` directory in the v0.2.0
release of the repository are the reference values against which
reproduction may be compared. A reproduction is considered
successful if the headline accuracy values in
`test_metrics.json` fall within the tolerance documented in § 10.5
of the corresponding reference values.

## 10.5 Hardware tolerances

Numerical reproducibility across different hardware is not
guaranteed even under strict seed control, because of
implementation differences in floating-point arithmetic between
CPU, CUDA, and Apple Silicon back-ends of PyTorch. The observed
differences are of the order of 0.01 pp to 0.1 pp in test
accuracy. Table 10.1 documents the reference hardware configuration
and the tolerances under which reproduction is expected to hold.

**Table 10.1.** Reference hardware and expected numerical tolerance.

| Configuration | Reference | Expected tolerance (test accuracy) |
|---|---|---|
| Intel/AMD x86-64 CPU with PyTorch CPU wheel | reference | bit-exact within the reference |
| NVIDIA CUDA GPU (Turing/Ampere/Ada) with PyTorch CUDA wheel | — | ±0.1 pp |
| Apple Silicon (M1–M3) with PyTorch MPS wheel | — | ±0.2 pp |
| Other CPU architectures (ARM, POWER) with PyTorch CPU wheel | — | ±0.1 pp |

The reference values reported in the repository were produced on
an Intel x86-64 CPU with PyTorch 2.3 installed from the standard
wheel index. Reproduction on the same hardware class with the same
Python version, the same PyTorch version, and the seeded
configurations should yield bit-identical numerical output.
Reproduction on alternative hardware classes is expected to produce
output within the tolerances stated in Table 10.1.

The tolerance values in the table are empirical estimates drawn
from preliminary tests rather than rigorous bounds. A reproduction
that produces test accuracy outside the stated tolerance indicates
either a silent environment mismatch or an implementation-level
non-determinism source that the deterministic cuDNN configuration
of § 9.4 has not caught. Known non-determinism sources are
enumerated in § 10.6.

An additional consideration concerns the training-time metric. The
validation accuracy trajectory — the sequence of per-epoch
validation accuracies that appears in `training_history.json` —
may vary more substantially between hardware classes than the
final test accuracy does, because the trajectory is sensitive to
the specific order in which floating-point operations are performed
during gradient descent. A reproduction in which the final test
accuracy matches the reference but the per-epoch trajectory does
not is not a failed reproduction; it is a reproduction under
floating-point accumulation differences that do not affect the
endpoint.

## 10.6 Known failure modes and diagnosis

Three classes of failure have been observed in reproduction
attempts to date. Each is documented below with its diagnosis and
recommended remedy.

**CUDA compatibility on recent consumer GPUs.** The NVIDIA
RTX 5090 laptop GPU, which incorporates the Blackwell architecture
with compute capability sm_120, was not supported by the stable
PyTorch release as of the v0.2.0 release of this repository. A
user attempting to run the training pipeline on such hardware will
encounter a `RuntimeError` at the point of first CUDA tensor
allocation, reporting an unsupported device. The remedy is to
install a pre-release PyTorch nightly build that supports
Blackwell, or to fall back to CPU execution by setting `device:
cpu` in the active configuration file. The CPU fallback is
approximately five times slower than the GPU path on comparable
architectures but produces bit-identical numerical output.

**Windows `num_workers` non-determinism.** On Windows, the
`torch.utils.data.DataLoader` with `num_workers > 0` exhibits
non-deterministic behaviour in the order of worker-process
initialisation, which propagates into the order of data samples
within an epoch. This is an acknowledged limitation of the Python
multiprocessing module on Windows. The shipped configurations
therefore set `num_workers: 0` on all platforms, which serialises
data loading at a modest throughput cost but guarantees identical
sample ordering across runs. Users on Linux or macOS who wish to
trade reproducibility for throughput may set `num_workers: 4` in
their local configuration override.

**Stale checkpoints after configuration changes.** If a user
modifies a configuration file and re-runs training, and a
previously produced `best.pt` checkpoint exists at the expected
output location, the configuration-hash verification described in
§ 9.4 will detect the mismatch and refuse to overwrite the
checkpoint silently. The error message identifies the specific
configuration field that has changed. The remedy is to delete the
stale checkpoint explicitly; the refusal to overwrite silently is
protocol-enforcement behaviour, not a bug.

Beyond the three failure modes documented above, a reproduction
that produces results outside the tolerance of Table 10.1 should
be investigated by examining the checkpoint provenance block
(§ 9.4) to verify that the checkpoint was produced under the
expected configuration hash, and by re-running the unit tests
(§ 9.5) to verify that the software environment matches the
specification.

The concerns of this chapter are the operational surface of the
reproducibility commitments made in § 4.6 and realised through the
engineering choices of chapter 9. The broader question of how the
findings of this work extend to unseen data distributions,
additional random seeds, and alternative architectural choices is
the subject of the final chapter. The empirical programme sketched
there — including the multi-seed verification that [Bouthillier et
al. (2021)](references.md#bouthillier2021accounting) argues is
necessary to establish the statistical force of any
benchmark-level claim — is the mechanism by which the single-run
findings of this document are to be placed on a firmer footing in
subsequent releases.
