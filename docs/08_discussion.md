# 8. Discussion

The empirical observations reported in chapter 7 are interpreted in
the present chapter. Four substantive claims are developed: that the
validation–test gap observed across the three architectures is
attributable to validation-specific overfitting rather than to
optimisation noise; that the ordering of test-set accuracy by
architecture is inconsistent with a monotone capacity-generalisation
relationship at the present scale; that the per-class error
structure exposes architecture-specific representational biases
rather than dataset-intrinsic difficulty alone; and that the
residual gap to the performance ceiling can plausibly be closed by
ensembling the three architectures rather than by further
architectural search. Each claim is qualified with respect to the
single-seed protocol under which it was obtained.

The chapter is organised in six sections. The first addresses the
validation–test gap directly. The second considers two competing
explanations for the non-monotone accuracy ordering. The third
interprets the behaviour of the spatial transformer network in
light of the observations. The fourth examines the per-class error
patterns. The fifth relates the findings to the broader literature
on overparameterisation and implicit regularisation. The sixth
enumerates the principal caveats under which the preceding
arguments should be read.

## 8.1 The validation–test gap

Three gap values were reported in Table 7.2: a gap of +1.17 pp for
`TrafficSignNet`, +1.83 pp for `DeepTrafficNet`, and +1.84 pp for
`TrafficSignNet-STN`. The direction is consistent across
architectures — validation performance exceeds test performance in
all three cases — but the magnitude correlates with parameter
count. The smallest architecture exhibits the smallest gap; the two
larger architectures exhibit gaps roughly 50 % wider and nearly
identical to each other.

Three explanations can be entertained for a gap of this pattern.
The first is statistical: the gap reflects random variation in a
finite test set of 12 630 samples. The second is structural: the
gap reflects a genuine difference between the validation
distribution and the test distribution, independent of architecture.
The third is architectural: the gap reflects a
capacity-dependent susceptibility to overfitting on
validation-specific features. These three are not mutually
exclusive, but they admit of distinct empirical signatures.

The statistical explanation is the easiest to dismiss. A gap of
1.17 pp on 12 630 test samples corresponds to 148 additional
incorrect predictions, against an expected sampling noise of
approximately $\sqrt{12\,630 \cdot p(1-p)} \approx 13$ samples at
$p = 0.99$. The observed gap is therefore approximately eleven
standard deviations of sampling noise — well beyond the regime in
which finite-sample variation could plausibly account for it. The
same calculation applied to the two larger architectures places
their gaps at approximately eighteen standard deviations of
sampling noise. Statistical noise is not a tenable explanation.

The structural explanation, which attributes the gap to a
distribution shift between validation and test sets, is more
difficult to dismiss — but it is also inconsistent with the
capacity dependence of the gap magnitude. A distribution shift
that operates on the input distribution alone, without interaction
with the classifier, should produce gaps of comparable magnitude
across the three architectures. The observed pattern, in which the
smallest architecture exhibits a gap approximately 60 % as large
as the two larger architectures, is not predicted by pure
distribution shift.

The architectural explanation therefore remains as the residual
hypothesis, and the one to which the remainder of this chapter
will attach. Under this explanation, the validation–test gap
arises because the larger architectures have sufficient capacity
to fit idiosyncratic features of the validation set — features
specific to the particular stratified sub-fold constructed in
§ 3.5 — which do not transfer to the test set. The smaller
architecture, with reduced capacity, is forced to rely on features
that transfer. This is an application of the overparameterisation
framework of [Nakkiran et al.
(2021)](references.md#nakkiran2021deep) and
[Belkin et al. (2019)](references.md#belkin2019reconciling) to the
specific regime in which the validation set is used for
checkpoint selection; its implications are developed further in
§ 8.2.

A limitation of this interpretation, directly acknowledged: it
rests on the proposition that the validation set contains
non-generalising features that are exploitable by higher-capacity
architectures. This proposition is empirically plausible on the
basis of the observed gap pattern but would require multi-seed
verification to be established with statistical force. The
verification is scheduled for the v0.3.0 release (§ 11); the
argument in the present document is to be read as a hypothesis
consistent with the available single-seed data.

## 8.2 Why additional capacity does not generalise

The headline ordering of test-set accuracies reported in Table 7.1
places `TrafficSignNet` (2.7 M parameters in the configuration
documented in § 5.1) above `DeepTrafficNet` (2.0 M parameters,
slightly fewer than the baseline) and `TrafficSignNet-STN` (1.3 M
parameters). The question this ordering invites — *why does
additional capacity not translate into improved generalisation?* —
is one of the central questions in the contemporary deep-learning
literature. Two competing explanatory frameworks are considered
here, and their applicability to the present observations is
assessed in turn.

**The augmentation-specific overfitting framework.** Under this
framework, the relevant quantity is not the parameter count in the
abstract but the ratio of effective model capacity to the
effective size of the training distribution. The augmentation
pipeline documented in § 6.3 enlarges the effective training
distribution considerably — each training image is seen hundreds of
times over the 20-epoch budget, each time under a different
combination of colour jitter, affine transformation, and
perspective distortion. A classifier with limited capacity is
compelled to learn invariances to these augmentations; a classifier
with surplus capacity can, in principle, learn not the invariances
but their sample-specific instantiations in the validation set. The
latter behaviour is precisely what the observed validation–test
gap pattern suggests.

**The double-descent framework.** Under this framework, following
[Nakkiran et al. (2021)](references.md#nakkiran2021deep) and the
earlier formulation of [Belkin et al.
(2019)](references.md#belkin2019reconciling), test-set accuracy
exhibits a non-monotone dependence on model capacity: there exists
an intermediate capacity regime in which additional parameters
produce *worse* generalisation before the interpolation threshold
is reached and generalisation recovers. The three architectures
examined here, all of which achieve validation accuracy above 99.7
%, may be positioned near or within this intermediate regime, with
the observed accuracy ordering reflecting the non-monotone
relationship.

The two frameworks are not strictly exclusive. The
augmentation-specific overfitting mechanism operates at the level
of the data pipeline and requires only that capacity exceed
task-intrinsic complexity; the double-descent mechanism operates
at the level of the loss landscape geometry and requires that the
model be in the intermediate-capacity regime. Both would predict a
non-monotone accuracy ordering of the kind observed here. Without
the multi-seed data required to distinguish them empirically, the
present work does not attempt to adjudicate between them. What it
does establish is the empirical claim that on GTSRB at the 98-99 %
accuracy regime, additional capacity does not guarantee improved
generalisation, and the implicit assumption that it does — which
remains common in applied machine-learning practice — should be
viewed with corresponding caution.

A related argument, due to [Zhang et al.
(2021)](references.md#zhang2021understanding), is that the
generalisation behaviour of overparameterised networks cannot be
predicted from the conventional hypothesis class complexity
measures that govern classical learning-theoretic guarantees; the
empirical generalisation gap depends on implicit biases of the
optimisation procedure and on properties of the data distribution
that are not captured by parameter counts alone. The present
observations, in which three architectures of comparable capacity
exhibit materially different generalisation, are consistent with
this broader claim.

## 8.3 The spatial transformer in context

The spatial-transformer-augmented variant, which was motivated in
§ 2.3 and § 5.2.4 as an architecture with an explicit
claim on the geometric source of intra-class variability,
underperforms both the unaugmented baseline and the deeper variant
on the test set. At 97.94 % accuracy, it is the weakest of the
three architectures examined here. Two considerations are relevant
to the interpretation of this result.

First, the geometric canonicalisation implemented by the spatial
transformer is not the only mechanism by which the classifier can
achieve invariance to pose, scale, and rotation. The augmentation
pipeline documented in § 6.3, which applies random affine
transformations and perspective distortions to training samples
independently of any architectural choice, provides a parallel
mechanism: a classifier trained on such augmented samples is
implicitly encouraged to learn invariant representations. The
question whether an explicit architectural invariance is necessary
when an implicit data-driven one is already imposed is therefore
not answered a priori; it depends on whether the architectural
invariance is sufficiently data-efficient to outperform the
data-driven one on the available training budget.

Second, the spatial transformer is not a neutral addition. It
introduces a localisation sub-network whose training signal flows
through the grid-sampling operation, and its own capacity adds to
the capacity budget of the architecture. The per-class analysis of
§ 7.6 suggests that the spatial-transformer variant fails on class
0 (speed limit 20) — a class on which the two non-STN architectures
both achieve 100 % accuracy — with a per-class accuracy of only
86.67 %. A plausible interpretation is that the learned spatial
transformations, having been optimised on the validation sample of
class 0, do not transfer to the test sample distribution of the
same class. This pattern is architecturally specific; the two
non-STN classifiers, which see the images unperturbed by the
spatial transformer, do not exhibit it.

The broader observation that a technique motivated by a specific
inductive bias can underperform when the bias interacts
detrimentally with the effective training distribution is not a
new one in the deep-learning literature. It recurs in the discussion
of implicit regularisation by [Neyshabur et al.
(2017)](references.md#neyshabur2017implicit), who argue that the
generalisation benefit of a given architectural choice depends on
its interaction with the optimisation trajectory. The present
observation that the spatial transformer's inductive bias does not
translate into a generalisation advantage on GTSRB is consistent
with this broader argument.

## 8.4 Per-class error patterns

The empirical result with the greatest diagnostic value, developed
in § 7.5 and § 7.6, concerns the structure of the residual errors
rather than their aggregate count. Two observations from those
sections admit of direct architectural interpretation.

**The concordant-error singleton.** That the concordant-error set
at the 95 % threshold contains exactly one class — class 30, beware
of ice/snow — is a non-trivial structural feature of the error
distribution. Its interpretation is straightforward. Class 30 is a
warning sign whose visual content is intrinsically difficult: the
ice/snow icon is small relative to the overall sign geometry, is
rendered at low contrast against the typically overcast backgrounds
of winter road conditions that dominate the class's training data,
and is visually confusable with several other triangular warning
signs bearing geometrically similar symbols. The failure of all
three architectures on this class is therefore attributable to a
dataset-intrinsic difficulty that no classifier operating on the
48 × 48 input resolution is well positioned to overcome. It is a
limit of the data, not of any classifier. This limit is specific
to the present benchmark; alternative approaches — higher input
resolution, classifier-specific oversampling of the affected class
during training, or, pragmatically, upstream filtering of inputs
for the relevant context — lie outside the scope of this work.

**The divergent errors on classes 0, 19, 21.** The divergent-error
analysis of § 7.6 reveals a qualitatively different pattern. Each
of these classes is failed by *exactly one* architecture while
being classified reliably by the other two. The specific
correspondences, transcribed from Table 7.5, are: class 0 (speed
limit 20) is failed by `TrafficSignNet-STN` only (86.67 %, against
100 % for the two non-STN architectures); class 19 (dangerous curve
to the left) is failed by `DeepTrafficNet` (83.33 %, against 100 %
and 98.33 %); and class 21 (double curve) is failed dramatically
by `DeepTrafficNet` (67.78 %, against 100 % for both shallower
architectures). The structural features of these confusions,
inspected in Table 7.3, show that `DeepTrafficNet`'s failure on
class 21 takes the form of consistent misclassification as class
11 (right-of-way at next intersection), a class that shares with
double-curve the property of being a warning-sign-like triangular
shape but is visually distinguishable by colour and internal
symbol.

The interpretation that accounts for these patterns is one of
architecture-specific representational bias. The two shallower
architectures, possessing a more limited feature hierarchy, rely
on features that happen to distinguish class 21 from class 11 —
perhaps the specific curvature of the double-curve symbol or its
rotational asymmetry. The deeper variant, having acquired a more
abstract feature representation, has discarded or de-emphasised
those specific features in favour of a more general warning-sign
representation that fails to distinguish the two. This is the
double-edged property of deep representations: they generalise
well on average but can be systematically mis-calibrated on
specific inputs. A related framing, due to [Zhang et al.
(2021)](references.md#zhang2021understanding), is that deep
networks interpolate training data in a class of function spaces
that is not constrained by the hypothesis classes of classical
learning theory; the resulting representations are high-performing
on aggregate but fragile on specific inputs where the aggregate
smoothness does not reflect the underlying class structure.

## 8.5 Implications for the overall research questions

The three research questions articulated in § 1.3 and operationalised
in § 4.1 can now be addressed directly on the basis of the
observations reported in chapter 7.

**Q1. Accuracy ordering.** The null hypothesis was that accuracy
would be monotonically increasing in parameter count. This
hypothesis is rejected by the observations: the smallest architecture
by parameter count at the committed configuration
(`TrafficSignNet-STN` at 1.3 M) and the largest
(`TrafficSignNet` at 2.7 M, following the configuration
reconciliation in § 5.1.2) are separated by 0.81 pp in test
accuracy, but in the direction opposite to the capacity prediction.
The more general claim — that increased parameter count on GTSRB
at the present performance regime yields monotone accuracy
improvement — is unsupported by these data.

**Q2. Validation–test consistency.** The validation accuracies of
the three architectures are within 0.14 pp of one another (99.78 %
to 99.92 %), whereas the test accuracies span 0.81 pp (97.94 % to
98.75 %). The ranking induced by validation accuracy is
`TrafficSignNet > DeepTrafficNet > TrafficSignNet-STN` (by small
margins); the ranking induced by test accuracy is nominally the
same, but the magnitudes of the differences differ substantially
between validation and test. Validation accuracy is therefore a
consistent but noisily calibrated predictor of test accuracy at
this performance regime, with particular unreliability in the
upper 0.3 pp band where all three validation accuracies fall.

**Q3. Error structure.** The per-class error patterns documented
in § 7.5 and § 7.6 admit of the clear answer that errors are *not*
uniformly distributed across the architectures. One class (class
30) fails concordantly across all three; four classes fail
divergently, with one architecture markedly underperforming the
other two. The error structure is therefore a mixture of
dataset-intrinsic difficulty (class 30) and
architecture-specific representational bias (classes 0, 19, 21).
The implication for subsequent work is that ensembling of the
three architectures — an operation that exploits divergent errors
as a source of complementary information — is a principled next
step with measurable expected improvement.

## 8.6 Caveats

The interpretive claims developed in the preceding sections rest
on a single training run per architecture, at seed $s = 42$. The
magnitude of random-seed-induced variation in contemporary
deep-learning benchmarks has been quantified by [Bouthillier et
al. (2021)](references.md#bouthillier2021accounting), who report
that it is, for small-to-moderate effect sizes, of the same order
of magnitude as differences typically used in the literature to
substantiate architectural improvements. The effect sizes observed
here — 0.81 pp of accuracy between best and worst architectures,
0.67 pp between the leader and the third-placed — fall within the
regime in which seed-level variance cannot be ignored.

The response to this concern, already articulated in § 4.6, is
that the qualitative conclusions of this chapter are expected to
be robust to seed variance even if specific numerical gaps shift by
a fraction of a percentage point. The claim that additional
capacity does not monotonically improve generalisation is a
structural claim, not a claim about precise accuracy differentials;
the divergent-error analysis of classes 0, 19, and 21 reveals
architecture-specific failure modes whose magnitude (13 to 32 pp in
per-class accuracy) substantially exceeds any plausible seed-level
variation; and the concordant-error finding on class 30 is
supported by per-class accuracies in the 75-90 % range, well below
the validation-level saturation at which seed noise dominates.

The more delicate claims — the specific ordering between
`DeepTrafficNet` and `TrafficSignNet-STN`, which are separated by
0.14 pp of accuracy, or the specific magnitude of the
validation–test gap differentials — should be treated with
greater caution. These are within the range where multi-seed
verification could plausibly reverse them. The v0.3.0 release
(§ 11) includes this verification in its specification.

A second and more specific caveat concerns the interpretation of
`DeepTrafficNet`'s failure on class 21. The failure is dramatic
(67.78 % accuracy) and is idiosyncratic to this architecture, but
it derives from a single test-set evaluation; whether the failure
is reproducible or instead a particular artefact of this training
trajectory cannot be determined without multi-seed data. The
evidence favours reproducibility — the failure is consistent with
the broader pattern of architecture-specific representational bias
documented in this chapter — but the empirical argument is not yet
closed.

The interpretations developed in this chapter should therefore be
understood as consistent with the available single-seed data and
with broader theoretical considerations, but as hypotheses subject
to revision rather than as established claims. The specific
empirical programme of the subsequent releases, detailed in
chapter 11, is the mechanism by which these hypotheses are to be
tested.
