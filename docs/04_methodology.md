# 4. Methodology

The experimental apparatus described in the preceding chapter — the
dataset, its partition, and its task-specific properties — is now
given a formal methodological framing. The present chapter states,
at the level of abstraction appropriate to protocol description, how
the three architectures introduced in chapter 5 are to be compared
without the comparison being confounded by extraneous sources of
variance. Concrete hyperparameter choices are deferred to chapter
6; architectural definitions are deferred to chapter 5; the present
treatment concerns the experimental design itself.

The chapter is organised into six sections. The first restates the
research questions of § 1.3 in a form directly testable against the
quantities that will be reported in chapter 7. The second formalises
the three-way data partition and articulates the case, already
introduced in § 3.5, against any exposure of the test set to the
training pipeline. The third defines the five evaluation metrics
adopted in this work and justifies their joint reporting. The
fourth documents the seed protocol and the determinism guarantees
associated with it. The fifth specifies the hardware and software
environment under which the reported numbers were produced. The
sixth articulates the limitations of the single-run reporting
adopted in the present release and connects them to the multi-seed
protocol planned for the follow-up release.

## 4.1 Operationalised research questions

The research questions posed in § 1.3 admit of the following
testable formulations against the quantities reported in chapter 7.

**Q1. Accuracy ordering.** Let $\mathcal{M} = \{M_1, M_2, M_3\}$
denote the three architectures described in chapter 5, with
$|\theta_1| < |\theta_2| < |\theta_3|$ ordering them by parameter
count. Let $\mathrm{acc}_{\mathrm{test}}(M_i)$ denote the test-set
accuracy of architecture $M_i$ under the protocol specified in the
remainder of this chapter. Q1 asks whether
$\mathrm{acc}_{\mathrm{test}}$ is monotonically increasing in $i$.
The null hypothesis is monotonic increase, motivated by the
informal expectation that additional capacity, under matched
training, ought to improve generalisation. Evidence against the
null is any observed ordering inconsistent with it.

**Q2. Validation–test consistency.** Let
$\mathrm{acc}_{\mathrm{val}}(M_i)$ denote the peak validation
accuracy of architecture $M_i$ across the training trajectory, and
$\mathrm{acc}_{\mathrm{test}}(M_i)$ the test accuracy of the
corresponding best-on-validation checkpoint. Q2 asks whether the
ranking induced by $\mathrm{acc}_{\mathrm{val}}$ agrees with the
ranking induced by $\mathrm{acc}_{\mathrm{test}}$. Disagreement
constitutes evidence of validation–test divergence, whose
interpretation is the subject of chapter 8.

**Q3. Error structure.** Let $\mathrm{err}_c(M_i)$ denote the
per-class error rate of architecture $M_i$ on class $c$, for
$c \in \{0, 1, \ldots, 42\}$. Q3 asks whether the sets
$\{c : \mathrm{err}_c(M_i) > \tau\}$, for a fixed threshold $\tau$
and across $i$, exhibit substantial overlap or disjointness.
Substantial overlap is consistent with the interpretation that all
three architectures fail on the same classes for dataset-intrinsic
reasons; substantial disjointness is consistent with the
interpretation that the architectures capture partially
complementary representations of the input distribution. The
practical implication of the latter is that ensembling of the three
models, of the kind scheduled for v0.3.0 (§ 11), is expected to
yield non-trivial gains.

These three questions share the property that their answers are
jointly determined by the experimental protocol specified below.
Changing the data partition, the random seeds, the evaluation
metrics, or the model-selection rule would change the quantities
being measured, and hence the answers. The protocol is accordingly
fixed in advance and held invariant across the three architectures.

## 4.2 Data splits

The three-way partition adopted in this work is constructed as
follows.

The **test set** $\mathcal{D}_{\mathrm{test}}$ is the official GTSRB
test partition, containing 12 630 images with their associated
class labels. It is defined by the dataset originators and is
preserved verbatim by this work. No modifications, augmentations,
or re-partitionings are applied.

The **validation set** $\mathcal{D}_{\mathrm{val}}$ and the
**training set** $\mathcal{D}_{\mathrm{train}}$ are obtained by a
stratified 20 / 80 split of the official GTSRB training partition,
as described in § 3.5. The resulting cardinalities are
approximately $|\mathcal{D}_{\mathrm{val}}| = 7\,842$ and
$|\mathcal{D}_{\mathrm{train}}| = 31\,367$. The stratification
preserves the per-class count ratios of the source partition in
both sub-folds. The assignment is deterministic under the
experiment seed and is therefore reproducible exactly.

The three sets are disjoint by construction: no image appears in
more than one of them. The partition is held invariant across all
three architectures, such that comparisons between architectures
are not confounded by differences in the data they observe.

A commitment to the case against test-set exposure during training
is stated explicitly. The **test set is touched by no component of
the training pipeline**. It is not used for early stopping, for
learning-rate scheduling, for model selection, for hyperparameter
tuning, for architecture selection, or for any other purpose that
would make observations on it part of the effective training
signal. The rationale for this commitment is documented by
[Pineau et al. (2021)](references.md#pineau2021improving), who
report that the most common cause of non-reproducibility in
machine-learning submissions is the use of test data in a role for
which validation data is appropriate. The commitment is enforced by
convention rather than by technical mechanism; § 9 describes the
organisational separation between training and evaluation code paths
that makes the convention easy to respect in practice.

The validation set is used for three purposes and three only. First,
for monitoring of training dynamics epoch by epoch, such that loss
and accuracy trajectories can be inspected during training and
reported in § 7.2. Second, for triggering of early stopping under
the criterion that the running minimum of validation loss has not
improved for a prescribed number of consecutive epochs. Third, for
selection of the checkpoint — the best-on-validation rather than
the final — that is subsequently evaluated on the test set. No
further use of the validation set is made.

## 4.3 Metrics

Five metrics are reported in chapter 7 alongside one another. Joint
reporting is necessary because each metric admits of a different
failure mode, and a classifier that optimises one without regard to
the others is a classifier whose behaviour is incompletely
characterised. The formal definitions follow.

Let $N = |\mathcal{D}_{\mathrm{test}}|$ denote the number of test
samples, $K = 43$ the number of classes, $y_i \in \{0, \ldots, K-1\}$
the ground-truth label of test sample $i$, and
$\hat{y}_i \in \{0, \ldots, K-1\}$ the predicted label.

**Accuracy.** The fraction of test samples that are correctly
classified:

$$
\mathrm{accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]
\tag{4.1}
$$

where $\mathbb{1}[\cdot]$ is the indicator function. Accuracy is
the most commonly reported metric and the most intuitive to
interpret, but it is also the metric most sensitive to class
imbalance: a classifier that achieves zero accuracy on the eight
least-populated classes of GTSRB (together approximately 4 % of
the training data) can nonetheless report aggregate accuracy above
95 %. For the moderately imbalanced GTSRB distribution documented
in § 3.2, this insensitivity is a material caveat rather than a
theoretical one.

**Top-5 accuracy.** The fraction of test samples for which the
ground-truth label is among the five highest-probability predicted
classes:

$$
\mathrm{accuracy}_{\mathrm{top}\text{-}5} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[y_i \in \mathrm{top}_5(f_\theta(\mathbf{x}_i))]
\tag{4.2}
$$

where $f_\theta(\mathbf{x}_i) \in \mathbb{R}^K$ denotes the
logit-or-probability vector produced by the classifier on input
$\mathbf{x}_i$ and $\mathrm{top}_5(\cdot)$ returns the set of
indices of its five largest components. Top-5 accuracy is a more
permissive metric than top-1 accuracy and is informative about the
distribution of errors: a classifier with high accuracy and
correspondingly high top-5 accuracy is one whose errors are
near-misses; one with high top-5 but markedly lower top-1 accuracy
is one whose errors are confined to a small number of confusable
pairs. The second pattern is expected on GTSRB in virtue of the
inter-class similarity among numeric speed-limit classes documented
in § 3.6.

**Macro-averaged F1 score.** The unweighted mean of per-class F1
scores. For class $c$, let $\mathrm{prec}_c$ and $\mathrm{rec}_c$
denote the precision and recall of the classifier on that class:

$$
\mathrm{prec}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c}, \qquad \mathrm{rec}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c}
\tag{4.3}
$$

where $\mathrm{TP}_c$, $\mathrm{FP}_c$, and $\mathrm{FN}_c$ denote
the counts of true positives, false positives, and false negatives
for class $c$ respectively. The per-class F1 is the harmonic mean

$$
F_{1,c} = \frac{2 \cdot \mathrm{prec}_c \cdot \mathrm{rec}_c}{\mathrm{prec}_c + \mathrm{rec}_c}
\tag{4.4}
$$

and the macro-averaged F1 is

$$
F_1^{\mathrm{macro}} = \frac{1}{K}\sum_{c=0}^{K-1} F_{1,c}.
\tag{4.5}
$$

The significance of macro-averaging rather than micro-averaging is
that it weights every class equally, irrespective of its sample
count. Consequently, $F_1^{\mathrm{macro}}$ penalises a classifier
that performs poorly on rare classes, which aggregate accuracy
does not.

**Matthews correlation coefficient.** For the multi-class case, the
MCC is computed from the confusion matrix $\mathbf{C}$ with
$C_{jk}$ denoting the count of samples with ground-truth class $j$
predicted as class $k$:

$$
\mathrm{MCC} = \frac{\sum_{jk}C_{kk}C_{jj} - C_{jk}C_{kj}}{\sqrt{\left(\sum_{k}(\sum_{j}C_{jk})(\sum_{j'\neq j}\sum_{k'}C_{j'k'})\right)\left(\sum_{k}(\sum_{j}C_{kj})(\sum_{j'\neq j}\sum_{k'}C_{k'j'})\right)}}
\tag{4.6}
$$

This multi-class generalisation is due to Gorodkin and is summarised
alongside its properties by [Chicco and Jurman
(2020)](references.md#chicco2020advantages), who argue that MCC is a
more reliable statistical summary than accuracy or F1 when the
class distribution is imbalanced. The coefficient takes values in
$[-1, +1]$: $+1$ indicates perfect prediction, $0$ indicates the
level of chance, and $-1$ indicates perfect disagreement. Because
the coefficient is bounded both below and above, it does not
inflate artificially under extreme class imbalance, a property that
accuracy lacks.

**Cohen's kappa.** Also computed from the confusion matrix,
Cohen's $\kappa$ measures the agreement between predicted and
ground-truth labels adjusted for the agreement expected by chance:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
\tag{4.7}
$$

where $p_o$ is the observed agreement (equal to accuracy for the
multi-class case) and $p_e$ is the agreement expected under
independence of predictions and ground truth. Like MCC, $\kappa$
is bounded in $[-1, +1]$ and takes value $0$ at chance level; it is
bounded above by $1$ at perfect agreement. The computation of
$p_e$ from the confusion matrix accounts for the marginal
distributions of predictions and ground truth, which makes
$\kappa$ sensitive to systematic bias toward majority classes in a
way that accuracy is not.

**Table 4.1.** Summary of the five metrics reported in chapter 7,
their formal ranges, their sensitivity to class imbalance, and their
principal interpretation.

| Metric | Range | Imbalance sensitivity | Principal interpretation |
|---|:-:|:-:|---|
| Accuracy | $[0, 1]$ | low | fraction correctly classified |
| Top-5 accuracy | $[0, 1]$ | low | fraction of near-misses |
| Macro F1 | $[0, 1]$ | high | class-balanced precision/recall |
| MCC | $[-1, +1]$ | high | chance-adjusted multi-class correlation |
| Cohen's κ | $[-1, +1]$ | high | chance-adjusted agreement |

The joint reporting of these five metrics is motivated by three
considerations. The first, from [Chicco and Jurman
(2020)](references.md#chicco2020advantages) and earlier work by
[Powers (2011)](references.md#powers2011evaluation), is that
accuracy is a brittle summary on imbalanced data and that MCC or
$\kappa$ should be reported alongside it. The second is that
differences between classifiers that are visible in one metric but
not in another constitute diagnostic evidence about the structure of
those differences: for example, two classifiers with similar
accuracy but markedly different macro F1 differ in their treatment
of rare classes. The third is that the chosen metrics are
inter-computable from the confusion matrix alone, so their joint
reporting adds no computational cost and no measurement overhead.

## 4.4 Seeds and determinism

Every component of the training pipeline that draws from a
pseudo-random source is seeded with the integer value specified in
the experimental configuration. For the reported baseline results,
that value is $s = 42$. The components so seeded include:

- The Python built-in `random` module.
- The NumPy pseudo-random generator.
- The PyTorch CPU pseudo-random generator.
- The PyTorch CUDA pseudo-random generator (per-device).
- The `PYTHONHASHSEED` environment variable, which fixes the
  iteration order of sets and dictionaries in otherwise hash-ordered
  operations.

In addition, the cuDNN backend is configured in deterministic mode,
with its autotuning disabled. This configuration trades a small
reduction in throughput — empirically between 10 % and 20 % on the
architectures examined here — for bit-exact reproducibility of the
gradient computation across runs on identical hardware. For the
configuration of this project, bit-exact reproducibility is a
defensible investment: the runtimes involved are modest, and the
alternative — results that differ by small amounts between runs for
no reason related to the experiments' content — undermines the
reproducibility claim this report makes elsewhere.

The determinism guarantee is, however, conditional on hardware. A
given seed, under the same software stack, produces identical
numerical output on a given GPU. Across GPUs of different
architectures — and, at the time of writing, even across driver
versions on the same GPU — floating-point drift at the level of the
final decimal places is to be expected. This drift does not affect
the qualitative conclusions of this report but does mean that the
reported numbers should be taken as specific to the hardware
configuration documented in § 4.5.

## 4.5 Environment

The results reported in this version of the document were produced
on a single workstation with the following software and hardware
characteristics.

**Software.** Python 3.12, PyTorch 2.x with CPU-only backend,
torchvision in the corresponding version, NumPy 2.x, Pandas 2.x,
scikit-learn 1.x, Matplotlib 3.x, and Seaborn 0.13.x. The exact
versions are pinned in `requirements.txt` at the repository root.
The operating system was Windows 11, though the code base is
portable and has been tested under Ubuntu 22.04 in the continuous
integration pipeline described in § 9.

**Hardware.** The training runs were performed on CPU. Although the
workstation was equipped with an NVIDIA RTX 5090 Laptop GPU — a
Blackwell-architecture device with 24 GB of video memory — the
release of PyTorch available at the time did not yet ship stable
kernels for the `sm_120` compute capability of this GPU. The
experimental runs therefore fell back to CPU execution, with the
documented runtime consequences summarised in § 10. The workaround
switches exposed in `src/traffic_signs/utils/seed.py` remain
available for users on hardware where the CPU fallback is not
acceptable; these are documented in § 10 and are expected to become
unnecessary with forthcoming PyTorch releases.

The separation between software and hardware in this account is
deliberate. The software stack, specified by the requirements file
and the Python version, is under the user's control and can be
reconstructed exactly on any platform that supports the dependencies.
The hardware stack is not, and the reported numbers are therefore
valid only up to the floating-point tolerance noted in § 4.4.

## 4.6 Limitations of single-run reporting

The baseline results reported in chapter 7 derive from a single run
per architecture, at the fixed seed $s = 42$. This reporting
convention is the simplest that an experimental protocol can adopt,
and it is widely used in the deep-learning literature; it is,
however, known to underrepresent the variance that seed choice alone
induces in reported metrics. [Bouthillier et al.
(2021)](references.md#bouthillier2021accounting) quantifies this
variance for a collection of representative benchmarks and finds
that it is, for small-to-moderate effect sizes, of the same order
of magnitude as the differences typically reported in the literature
to substantiate claims of architectural improvement. [Dror et al.
(2018)](references.md#dror2018hitchhiker) argues accordingly for
the routine inclusion of appropriate statistical significance tests
in machine-learning comparisons.

The present work adopts single-seed reporting for the baseline
release under two justifications. The first is that the effect
sizes observed across the three architectures (§ 7.1) are
sufficiently large that they are unlikely to be reversed by
seed-level variance under the threshold established empirically by
[Bouthillier et al.
(2021)](references.md#bouthillier2021accounting). The second is
that single-seed reporting establishes a reproducible reference
point against which subsequent multi-seed runs can be calibrated.
Multi-seed reporting in advance of a reproducible single-seed
reference would produce statistical claims without a verifiable
ground truth; this report prefers the inverse sequencing.

The follow-up release described in § 11 is planned to include
multi-seed reporting across at least three seed values, such that
mean and standard deviation of the reported metrics can be
substantiated for each architecture. Until that release, the
numerical claims of chapter 7 should be read as specific to the
seed configured in the current distribution rather than as
population estimates of expected performance. The qualitative
conclusions — most importantly, the ordering observed across the
three architectures — are discussed in chapter 8 with explicit
attention to whether they can be sustained under the
seed-variability bound from
[Bouthillier et al. (2021)](references.md#bouthillier2021accounting).
