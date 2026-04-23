# 11. Future Work

The empirical findings of chapters 7 and 8, in conjunction with
the engineering infrastructure documented in chapter 9, define a
programme of subsequent work whose broad outlines can be stated
with precision and whose individual components have defensible
expectations of success. The present chapter describes two planned
releases — v0.3.0 and v0.4.0 — and enumerates the open research
questions that extend beyond their scope.

The chapter is organised into four sections. The first describes
the v0.3.0 release, which applies ensemble averaging to the three
baseline architectures without introducing any new training. The
second describes the v0.4.0 release, which introduces a new
architecture (`TrafficSignNetV2`) and a targeted set of training
techniques under a pre-registered ablation protocol. The third
enumerates the open limitations and research questions that
remain beyond the scope of v0.4.0. The fourth offers a concluding
summary of the release programme.

## 11.1 Version 0.3.0: ensemble of the three baseline models

The most immediate application of the empirical findings of this
report is the one that requires the least engineering effort:
ensemble averaging of the three architectures whose individual
behaviour has already been reported in chapter 7. An ensemble of
$M$ classifiers operating on input $x$ produces its output
probability for class $c$ as the arithmetic mean of the per-classifier
probabilities:

$$
\hat{p}_{\mathrm{ens}}(c \mid x) = \frac{1}{M}\sum_{m=1}^{M} \hat{p}_m(c \mid x)
$$

**Equation (11.1)**

The classification decision is the class $c$ that maximises the
ensemble probability. This is the simplest ensemble formulation,
variously described as *soft voting* or *probability averaging*; a
richer family of ensemble methods exists ([Dietterich
(2000)](references.md#dietterich2000ensemble) provides a taxonomic
review), but the arithmetic mean is the appropriate baseline
because it introduces no additional parameters and requires no
additional training.

### 11.1.1 Scientific motivation

The expected benefit of ensemble averaging depends on a property
of the individual classifiers known variously as *error diversity*
or *decorrelated errors*: the ensemble outperforms its components
in direct proportion to the extent that the individual classifiers
make errors on different inputs. If two classifiers fail
concordantly — if they are wrong on the same test samples — then
averaging their probabilities does not move the error distribution.
If they fail divergently, then the averaging suppresses each
classifier's errors by the probability mass contributed by the
correctly classifying peer. The quantitative relationship was
formalised by [Hansen and Salamon
(1990)](references.md#hansen1990neural) for the special case of
majority-voting ensembles of neural classifiers: under the
assumption that individual errors are independent and that each
classifier's individual error rate is below 0.5, an $M$-classifier
ensemble can arbitrarily reduce error by increasing $M$. Modern
deep-network ensembles rarely achieve the error independence
assumed by that analysis, but the qualitative implication —
diverse errors compound into ensemble improvement — remains
operative.

The divergent-error analysis in § 7.6 provided direct evidence
that the three architectures of this report satisfy the necessary
precondition for ensemble benefit. The concordant-error set at the
95 % threshold contained exactly one class (class 30, beware of
ice/snow), against which no ensemble benefit is expected. The
divergent-error set contained four classes — 0, 19, 21, and the
concordant class 30 itself — on three of which (classes 0, 19,
21) exactly one architecture underperformed while the other two
classified reliably. This pattern is the diagnostic signature of
an ensemble-favourable error distribution: the majority vote or
probability average will, on samples from those classes, resolve
the disagreement in the direction of the two correctly classifying
architectures.

A quantitative upper bound on the expected ensemble benefit can
be derived by assuming that every divergent-class error currently
observed — 29 from `DeepTrafficNet` on class 21, 29 from
`TrafficSignNet-STN` on class 3, and the remaining 13 to 20 from
various architectures on classes 0, 19, 22, 39 — is fully resolved
by ensembling. Under this maximally optimistic assumption, the
ensemble would recover approximately 150 additional correct
predictions out of 12 630, equivalent to approximately 1.2 pp of
test accuracy. A realistic expectation is that approximately half
of this bound is realisable, which would place the expected
ensemble test accuracy in the range of 99.2 % to 99.5 %. The
interval is wide because it depends on the specific pattern of
error decorrelation, which cannot be predicted from the
per-architecture results without the ensemble actually being
constructed.

### 11.1.2 Engineering plan

The v0.3.0 release requires no new training. The three checkpoints
produced for v0.2.0 are used directly. A new module
`src/traffic_signs/inference/ensemble.py` is introduced, exposing
an `EnsemblePredictor` class that accepts a list of model checkpoints
and an aggregation strategy (`mean_prob`, `max_prob`, or
`majority_vote`) and produces per-sample ensemble predictions.

A single additional evaluation script, `scripts/eval_ensemble.py`,
applies the `EnsemblePredictor` to the test partition and produces
a test-set metrics report structurally identical to the one
produced for individual models in v0.2.0. The reported metrics
include, in addition to the standard five (accuracy, top-5
accuracy, cross-entropy, MCC, κ), a per-class ensemble-vs-best-
individual comparison that quantifies the benefit of ensembling on
each class.

The v0.3.0 release adds two tests to the existing suite (a unit
test for the three aggregation strategies and an integration test
for the ensemble evaluator) and is estimated at approximately 200
lines of new code and 50 lines of tests, corresponding to an
implementation time of one to two days.

## 11.2 Version 0.4.0: `TrafficSignNetV2`

The v0.4.0 release introduces a new architecture and applies to it
a targeted set of training techniques that the v0.2.0 release
deliberately excluded under the principle of introducing one change
at a time. The intention is to produce a single architecture whose
test accuracy substantially exceeds that of the current best
(`TrafficSignNet` at 98.75 %) without relying on ensemble
averaging, such that the ensemble strategy of v0.3.0 and the
architectural improvements of v0.4.0 can be composed in a
subsequent release.

### 11.2.1 Architectural plan

`TrafficSignNetV2` is a moderate extension of the baseline
`TrafficSignNet`, retaining the three-block convolutional feature
extractor but with three specific modifications. First, each
convolutional block incorporates a residual connection from block
input to block output, following the convention of the residual
networks of He et al., in order to stabilise the training
trajectory at increased depth. Second, the classifier head is
widened from 512 hidden units to 768 units while retaining the
single-layer structure, reflecting the observation in § 5.1.2 that
the classifier head contributes the dominant fraction of
parameters and accommodates capacity expansion more
parameter-efficiently than additional convolutional blocks.
Third, global average pooling is applied after the final
convolutional block in place of the flattening operation, which
reduces the flattened feature dimension from 4 608 to 128 and
correspondingly reduces the parameter count of the first linear
layer by a factor of approximately 36. The aggregate effect is an
architecture with approximately the same parameter budget as the
existing `TrafficSignNet` but with the parameters redistributed
more favourably between convolutional and dense components.

### 11.2.2 Training techniques under ablation

Five specific training techniques are applied to `TrafficSignNetV2`
under a pre-registered ablation protocol. The protocol registers
the hypothesised effect of each technique before the experiments
are conducted, such that the post-hoc analysis can distinguish
between confirmed and disconfirmed hypotheses without selection
bias.

**Class-weighted cross-entropy.** Per-sample loss is scaled by a
class-specific weight inversely proportional to the square root of
class frequency, following the convention for imbalanced
classification. *Registered hypothesis:* test accuracy on the
eight least-populated classes (§ 3.2) improves by at least 1 pp
relative to uniform weighting; overall test accuracy is unchanged
or improves by at most 0.3 pp.

**Label smoothing.** The one-hot target distribution is replaced
by a softened variant with smoothing parameter $\varepsilon = 0.1$,
following the recommendation of [Szegedy et al.
(2016)](references.md#szegedy2016rethinking). *Registered
hypothesis:* classifier calibration improves (measured by expected
calibration error against a 20-bin reliability diagram); test
accuracy is unchanged or improves by at most 0.2 pp; training
cross-entropy is substantially higher than under unsoftened targets
by construction.

**One-cycle learning-rate schedule.** The schedule of [Smith and
Topin (2019)](references.md#smith2019super) is applied in place
of the `ReduceLROnPlateau` schedule of v0.2.0, with a maximum
learning rate of $1 \times 10^{-2}$ (ten times the current
stationary rate) and a cosine-annealing decay over the epoch
budget. *Registered hypothesis:* convergence to within 0.5 pp of
final accuracy is achieved in 10 epochs (half the current budget);
final accuracy is at most 0.2 pp different from under the baseline
schedule.

**Test-time augmentation.** The same augmentation pipeline used
during training is applied at test time, with the final prediction
computed as the average of predictions over five augmented versions
of each test image. *Registered hypothesis:* test accuracy improves
by at least 0.3 pp, primarily via resolution of the geometric
confusions identified in Table 7.3.

**Mixed-precision training.** Float16 arithmetic is used for
forward and backward passes, with float32 accumulation, following
the established practice surveyed by [Howard and Gugger
(2020)](references.md#howard2020fastai). *Registered hypothesis:*
wall-clock training time is reduced by at least 30 % on CUDA
hardware without loss of final accuracy beyond 0.1 pp.

### 11.2.3 Multi-seed evaluation

The v0.4.0 release additionally addresses the single-seed
limitation identified in § 4.6 and § 8.6. Each configuration
evaluated under v0.4.0 is run under three seeds ($s \in \{42, 43,
44\}$), with the reported metrics comprising the per-seed mean and
standard deviation. The choice of three seeds rather than the
statistically preferable larger number reflects the computational
budget of a community-maintained open-source project; three seeds
is sufficient to distinguish seed-level variance from genuine
configuration effects at the 0.3 pp magnitude estimated by
Bouthillier et al. and quoted in § 8.6, but is insufficient for
statistically rigorous claims about differences smaller than this
threshold.

A consequence of the multi-seed protocol is that accuracy
comparisons between v0.2.0 and v0.4.0 are no longer
single-sample comparisons. The three-seed mean of v0.4.0 is
reported against the single-seed point estimate of v0.2.0, with
the interpretation that the v0.4.0 number is a lower-variance
estimate of the same underlying quantity. The v0.3.0 ensemble
numbers, which depend on the three single-seed v0.2.0 checkpoints,
are correspondingly excluded from the multi-seed comparison.

## 11.3 Open limitations

Beyond the scope of v0.3.0 and v0.4.0, three classes of open
question remain and are acknowledged here without prescription.

**Out-of-distribution robustness.** The GTSRB benchmark is a
closed-world classification task: every test sample belongs to one
of the 43 training classes, and the classifier is evaluated on the
assumption that this closed-world property holds. Real-world
traffic-sign recognition operates under open-world conditions: the
classifier must detect signs of types not seen during training,
signs occluded by foreign objects, signs photographed under
weather or illumination conditions absent from the training
distribution, and the possibility of no sign being present at all.
A credible assessment of the architectures examined in this report
against any of these operational requirements is beyond the scope
of the benchmark and therefore beyond the scope of the present
work. The v0.5.0 release is intended to extend the evaluation to
at least one public out-of-distribution test set, likely the
Mapillary Traffic Sign Dataset, but this extension is not yet
scheduled.

**Cross-country generalisation.** GTSRB contains German traffic
signs exclusively, which conform to a subset of the Vienna
Convention on Road Signs and Signals and differ in visual detail
from the signage standards of other countries. The extent to which
a classifier trained on GTSRB generalises to the signage standards
of, for example, the United Kingdom, the United States, or Japan is
an open empirical question. The direct application of a
GTSRB-trained classifier to the images of a differently
standardised corpus is likely to fail catastrophically; the extent
to which the learned features transfer under light fine-tuning is
less clear. No component of the present work addresses this
question.

**Adversarial robustness.** The brittleness of contemporary
convolutional classifiers to adversarial perturbations is a
well-documented phenomenon, and the three architectures examined
in this report are presumed vulnerable in the same way. The
operational consequences of adversarial vulnerability are
particularly salient for traffic-sign recognition, where a targeted
perturbation on a physical sign could in principle cause
misclassification by an autonomous driving system. No component of
the present work addresses adversarial robustness; the v0.2.0
release is therefore unsuitable for deployment in any
safety-critical application without substantial additional
evaluation.

## 11.4 Summary

The empirical findings of this report support a defined programme
of subsequent releases. The v0.3.0 release applies ensemble
averaging to the three existing checkpoints, which is expected on
the basis of the divergent-error analysis of § 7.6 to improve test
accuracy by approximately 0.5 to 1.0 pp without any additional
training. The v0.4.0 release introduces `TrafficSignNetV2` with a
residual-connection modification, a global-average-pool classifier
front-end, and five pre-registered training techniques — class
weighting, label smoothing, one-cycle learning-rate scheduling,
test-time augmentation, and mixed-precision arithmetic — each with
an explicit registered hypothesis against which its post-hoc
effect is to be measured. A multi-seed evaluation protocol
addresses the single-seed limitation of v0.2.0 sufficiently to
distinguish genuine configuration effects from seed-level variance
at the estimated noise floor of approximately 0.3 pp.

Three classes of question remain beyond the scope of v0.4.0:
out-of-distribution robustness, cross-country generalisation, and
adversarial robustness. Each is acknowledged as a real operational
concern without claim to being addressed by the present work. The
deployment posture of the current release is that of a research
artefact suitable for benchmarking and pedagogical use, not of a
production system; the distance between these postures is
measurable, and the work required to close it is substantial and
largely orthogonal to the empirical programme sketched above.
