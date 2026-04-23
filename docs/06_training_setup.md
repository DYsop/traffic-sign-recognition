# 6. Training Setup

The architectures described in chapter 5 are fit to the training
data by gradient-based optimisation under a protocol specified in
the present chapter. The protocol is structured to minimise the
number of choices that differ between the three architectures: a
common optimiser family, a common learning-rate schedule, a common
augmentation pipeline, a common batch size, and a common epoch
budget are adopted wherever feasible. Where deviations are necessary
— as in the choice of optimiser for the deeper variant — they are
documented with their rationale. A consolidated hyperparameter
summary appears in § 6.6.

The chapter is organised into six sections. The first covers the
optimiser choice and learning-rate schedule. The second specifies
the loss function and argues against the inclusion of label
smoothing in the baseline release. The third documents the
augmentation pipeline, including the task-specific exclusion of
horizontal flipping introduced in § 3.6. The fourth fixes the
batch size, epoch budget, and early-stopping criterion. The fifth
describes the checkpointing policy. The sixth assembles the
consolidated hyperparameter table and enumerates the deviations
between architectures.

## 6.1 Optimiser and learning-rate schedule

All three architectures are trained by first-order stochastic
gradient descent with adaptive per-parameter learning rates. The
specific optimiser is Adam ([Kingma and Ba,
2015](references.md#kingma2015adam)) for the two shallower
architectures and AdamW ([Loshchilov and Hutter,
2019](references.md#loshchilov2019decoupled)) for `DeepTrafficNet`.
The two optimisers differ only in the treatment of weight decay:
Adam applies an L2-penalty-equivalent update that is entangled with
the per-parameter learning-rate normalisation, while AdamW applies
a multiplicative weight-decay step after the gradient update, which
preserves the intended regularising effect independently of the
adaptive normalisation.

The choice of adaptive optimisation over stochastic gradient
descent with momentum deserves specific justification. [Wilson et
al. (2017)](references.md#wilson2017marginal) has shown, in
controlled comparisons across image classification benchmarks, that
the generalisation advantage sometimes attributed to adaptive
methods is in large part attributable to differences in
learning-rate tuning effort rather than to the adaptive update rule
per se. Under well-tuned SGD with momentum, adaptive optimisers
offer little or no generalisation benefit, and occasionally a
small one is observed in the reverse direction. The present work
is nonetheless conducted with Adam and AdamW for engineering
rather than statistical reasons: the adaptive methods transfer
learning-rate settings more reliably across the three architectures,
which removes a per-architecture tuning exercise that would otherwise
confound the comparison. The implication is that the reported results
should be read as a comparison of architectures under a standardised
optimisation protocol, not as a claim about optimiser preference.

The initial learning rate is $1 \times 10^{-3}$ for all three
architectures. This value is the default recommended in the
original Adam paper and has been found empirically to be a
reasonable starting point for convolutional classifiers on
moderate-scale image datasets. The learning rate is subsequently
reduced by a fixed factor of 0.5 whenever the validation loss has
failed to improve for three consecutive epochs, as implemented by
the `ReduceLROnPlateau` scheduler of PyTorch. The patience threshold
of three epochs is conservative; it allows the scheduler to
distinguish epoch-level noise in the validation signal from genuine
stagnation, at the cost of admitting a small number of unproductive
epochs at each plateau. A more aggressive schedule — for example,
the one-cycle policy of [Smith and Topin
(2019)](references.md#smith2019super) — is considered for the
follow-up release described in § 11, but has been deferred from
v0.2.0 on the principle of introducing one change at a time.

Weight decay is applied at a rate of $1 \times 10^{-4}$ for
`DeepTrafficNet` (the AdamW-trained architecture) and disabled
for the two Adam-trained architectures. This asymmetric choice
reflects the empirical observation that the deeper variant, with
its larger parameter count, exhibits a tendency toward unbounded
weight magnitude growth during extended training; decoupled weight
decay counteracts this tendency without affecting the
learning-rate dynamics. The shallower architectures, whose
parameter count is smaller by roughly a factor of three, do not
exhibit the same tendency in the epoch budget under consideration
and are therefore left without explicit weight decay.

## 6.2 Loss function

The loss function is the standard categorical cross-entropy between
predicted class probabilities and one-hot encoded ground-truth
labels. For a batch of $B$ samples with logits
$\mathbf{z}_i \in \mathbb{R}^{K}$ and ground-truth labels
$y_i \in \{0, \ldots, K-1\}$, the mean cross-entropy is

**Equation (6.1):**

$$
\mathcal{L}_{\mathrm{CE}}(\theta) = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(z_{i, y_i})}{\sum_{k=0}^{K-1} \exp(z_{i, k})}
$$

No label smoothing is applied in the baseline release. The
technique, introduced by [Szegedy et al.
(2016)](references.md#szegedy2016rethinking), replaces the
one-hot target distribution with a softened variant in which a
small mass $\epsilon$ is redistributed uniformly across the
non-target classes. Label smoothing has been observed empirically
to improve calibration and, in some settings, generalisation; it
is a defensible addition to the training pipeline for GTSRB. The
decision to omit it from v0.2.0 reflects the protocol principle
of introducing one technique at a time, such that the contribution
of any single change to the final test-set accuracy can be
attributed unambiguously. Label smoothing is among the techniques
scheduled for evaluation in the v0.4.0 release (§ 11).

Class weighting within the cross-entropy loss is similarly absent
from the baseline release. Given the moderate class imbalance of
GTSRB documented in § 3.2, a class-weighted loss — in which the
per-sample loss is multiplied by a class-specific factor inversely
proportional to class frequency — would in principle improve
performance on the least populated classes. The same reasoning
applies: class weighting is deferred to the follow-up release,
where its isolated contribution can be measured. For the baseline
run, the uniform per-sample weighting of the standard cross-entropy
is retained.

## 6.3 Augmentation pipeline

The augmentation pipeline is applied to training-set images and is
disabled on validation and test sets. The broader rationale for
augmentation — as a means of imposing invariances on the learned
representation without enlarging the label space — is surveyed by
[Shorten and Khoshgoftaar
(2019)](references.md#shorten2019survey). The pipeline adopted here
has been chosen to reflect the real-world variability documented in
§ 3.3 (illumination changes, moderate rotation, scale variation,
modest perspective distortion) without introducing transformations
that would violate the label-preservation requirement described in
§ 3.6. The full specification follows.

**Resize to 48 × 48.** The native resolution of the GTSRB images
varies across the corpus, from approximately 15 × 15 pixels at the
low end to approximately 250 × 250 pixels at the high end. All
images are resampled to a common 48 × 48 resolution prior to any
subsequent transformations. The resampling is performed by bilinear
interpolation in torchvision's default configuration. The resolution
choice balances two constraints: a resolution below approximately
32 × 32 loses discriminative detail for the numeric speed-limit
classes discussed in § 3.6, and a resolution substantially above
48 × 48 increases the computational cost of training without
corresponding accuracy gains on this dataset.

**Colour jitter.** Brightness, contrast, and saturation are each
perturbed by a uniformly distributed factor in the range $[1-0.2,
1+0.2]$, and hue is perturbed by an additive offset in the range
$[-0.1, +0.1]$. The implementation is
`torchvision.transforms.ColorJitter` in its standard formulation.
The parameters are chosen conservatively: the jitter reproduces the
photometric variability documented in § 3.3 without pushing the
augmented image outside the empirical range of the training
distribution.

**Random affine transformation.** The image is rotated by an angle
drawn uniformly from $[-15°, +15°]$, translated along each axis by
a fraction of the image dimension drawn uniformly from $[-0.1,
+0.1]$, and sheared by an angle drawn uniformly from $[-10°,
+10°]$. The ranges are intentionally modest: larger rotations
would relabel certain directional signs (for example, a rotation
exceeding approximately 30° on the no-entry class produces an image
that is no longer canonically recognisable), and larger translations
would risk moving the sign outside the image frame. The
implementation is `torchvision.transforms.RandomAffine`.

**Random perspective distortion.** A projective transformation
simulating mild perspective change is applied with probability
0.5, with a distortion scale of 0.2. The implementation is
`torchvision.transforms.RandomPerspective`. Perspective variation
of this magnitude is consistent with the camera-pose variability
documented in § 3.3 and is plausibly the source of the geometric
distortion that the spatial transformer network of § 5.2 is
designed to canonicalise.

**Conversion to tensor and normalisation.** The augmented image is
converted to a PyTorch tensor with values in the range $[0, 1]$ and
normalised by per-channel means and standard deviations of $(0.5,
0.5, 0.5)$ each. This is the mid-range normalisation adopted by the
original notebook code and preserved for consistency with the
reported numbers; a more precise alternative would use the
dataset-wide channel means and standard deviations, but the
empirical difference in final accuracy is negligible on this
benchmark.

### 6.3.1 The case against horizontal flipping

The augmentation pipeline explicitly omits horizontal flipping. The
justification, introduced in § 3.6 and made concrete here, is that
GTSRB contains multiple classes whose semantic meaning is
orientation-dependent. A non-exhaustive enumeration includes:

- Class 33 (turn right only) and class 34 (turn left only) are
  visual mirror images of one another.
- Class 17 (no entry) has a visually symmetric appearance but its
  prohibitive meaning is oriented.
- Class 19 (dangerous curve to the left) and class 20 (dangerous
  curve to the right) are visual mirror images.
- Class 36 (mandatory direction: straight or right) and class 37
  (mandatory direction: straight or left) are visual mirror images.
- Class 38 (keep right) and class 39 (keep left) are visual mirror
  images.

For these classes, a horizontal flip applied during augmentation
produces an image that, if interpreted according to the semantics
of the original label, is no longer labelled correctly. The
consequence of applying a flip unconditionally is that the
classifier is trained against label noise whose magnitude scales
with the frequency of the affected classes. An audit of the
training partition indicates that approximately 17 % of training
samples belong to classes for which horizontal flip produces a
different ground-truth class; applying flip with probability 0.5
would therefore introduce approximately 8.5 % label noise into the
training signal.

The correct alternative — class-aware flipping that applies the
flip only to classes for which the transformation is
label-preserving, such as the circular warning-sign category — is
feasible but complicates the implementation without delivering
measurable benefit on this benchmark. Simply omitting horizontal
flipping altogether is the chosen design. The omission is an
intentional deviation from the default augmentation pipeline of the
original notebook code, which applied `RandomHorizontalFlip` without
class awareness; the resulting label corruption is enumerated
further in § 9 as one of the defects addressed during refactoring.

Figure 6.1 shows representative augmentation samples produced by the
pipeline described in this section, illustrating the range of
photometric and geometric variability introduced during training.

**Figure 6.1.** Five augmented versions of a single GTSRB training
image (class 14, stop sign), produced by the full augmentation
pipeline. The variability spans photometric perturbation (brightness
and contrast), small rotation and translation, and mild perspective
distortion. Horizontal flipping is absent, as motivated above.

## 6.4 Batch size, epoch budget, and early stopping

The batch size is 64 for all three architectures. This value fits
comfortably within the memory envelope of both GPU and CPU
execution and admits sufficient per-step gradient noise for the
adaptive optimisers to function effectively without additional
stabilisation. Larger batch sizes (128 or 256) were considered but
not adopted: the accuracy difference observed in preliminary
experiments was within the single-seed variance envelope, and the
computational cost of larger batches is disproportionate in the
CPU-execution configuration documented in § 4.5.

The epoch budget is 20 for all three architectures. The choice was
made empirically: all three training trajectories exhibit
convergence of the validation metrics well before epoch 20, and an
extension of the budget to 40 epochs did not yield measurable test
accuracy gains in preliminary experiments. The 20-epoch ceiling is
consequently a conservative upper bound rather than a tightly tuned
parameter.

Early stopping is implemented as a soft criterion rather than a
hard one: training proceeds for the full 20-epoch budget, but the
checkpoint selected for test-set evaluation is the best-on-validation
checkpoint encountered during training. The separation between the
training trajectory and the checkpoint selected reflects the
observation, documented by [Prechelt
(1998)](references.md#prechelt1998early), that a strict early-stopping
rule — in which training is terminated as soon as a patience
criterion is met — can prematurely terminate training when the
validation signal exhibits brief plateaus. Allowing the training
trajectory to run to the full budget while selecting the
best-on-validation checkpoint combines the robustness of full-budget
training with the generalisation benefit of early stopping.

## 6.5 Checkpointing

Two checkpoints are maintained per architecture during training.
The first, `best.pt`, is the checkpoint corresponding to the lowest
validation loss observed to date. It is overwritten whenever the
current epoch produces a lower validation loss than any previous
epoch. The second, `last.pt`, is the checkpoint corresponding to
the final training epoch. It is overwritten at the end of every
epoch.

Evaluation on the test set uses the `best.pt` checkpoint
exclusively. The `last.pt` checkpoint is retained for diagnostic
purposes — specifically, for inspection of the terminal training
state and for potential warm-start of a subsequent training run —
but is not used in the reported results. This separation of
selection criterion from trajectory retention mirrors the soft
early-stopping discussion of § 6.4.

Each checkpoint includes the model state dictionary, the optimiser
state dictionary, the epoch number, the validation metrics at the
corresponding epoch, and the configuration hash of the experiment.
The inclusion of the configuration hash is a protocol safeguard:
any attempt to load a checkpoint with a configuration that differs
materially from the one under which the checkpoint was produced
raises an explicit error rather than silently proceeding. The
safeguard is implemented in `src/traffic_signs/training/trainer.py`
and is documented further in § 9.

## 6.6 Consolidated hyperparameter summary

The hyperparameter choices documented in the preceding sections are
assembled in Table 6.1. Values that are held constant across the
three architectures are stated once; values that deviate between
architectures are enumerated per architecture.

**Table 6.1.** Consolidated training hyperparameters.

| Hyperparameter | `TrafficSignNet` | `TrafficSignNet-STN` | `DeepTrafficNet` |
|---|---|---|---|
| Optimiser | Adam | Adam | AdamW |
| Initial learning rate | $1 \times 10^{-3}$ | $1 \times 10^{-3}$ | $1 \times 10^{-3}$ |
| Weight decay | 0 | 0 | $1 \times 10^{-4}$ |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |
| Scheduler factor | 0.5 | 0.5 | 0.5 |
| Scheduler patience | 3 epochs | 3 epochs | 3 epochs |
| Loss | cross-entropy | cross-entropy | cross-entropy |
| Label smoothing | disabled | disabled | disabled |
| Class weighting | disabled | disabled | disabled |
| Batch size | 64 | 64 | 64 |
| Epochs | 20 | 20 | 20 |
| Early stopping | soft (best-on-val) | soft (best-on-val) | soft (best-on-val) |
| Input resolution | 48 × 48 | 48 × 48 | 48 × 48 |
| Colour jitter | $\pm 0.2$ BCS, $\pm 0.1$ hue | same | same |
| Random affine | $\pm 15°$, $\pm 10\%$ trans | same | same |
| Random perspective | $p = 0.5$, scale $0.2$ | same | same |
| Horizontal flip | disabled | disabled | disabled |
| Normalisation | mean 0.5, std 0.5 | same | same |
| Seed | 42 | 42 | 42 |

The deviations between architectures reduce to two: the choice of
Adam versus AdamW and the corresponding weight-decay setting. Both
deviations are tied to the parameter-count differential between the
shallower architectures and the deeper variant and are motivated in
§ 6.1.

The aggregate effect of the protocol specified in this chapter is
that the three architectures of chapter 5 are trained under
conditions that differ only where justified by architectural
considerations. The results reported in chapter 7, and their
interpretation in chapter 8, are accordingly specific to this
protocol rather than to the architectures in isolation — a scope
constraint that should be borne in mind when comparing the numbers
reported here against results from the broader literature on
GTSRB.
