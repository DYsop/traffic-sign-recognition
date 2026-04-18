# 2. Related Work

The problem of traffic sign recognition sits at the intersection of
several established research threads. To situate the present
contribution precisely, this chapter traces five of those threads.
The first charts the evolution of reported performance on the
German Traffic Sign Recognition Benchmark (GTSRB) from its
introduction in 2011 through the present, with attention to the
architectural idioms that have successively occupied the state of
the art. The second examines shallow convolutional baselines of
the kind that remain relevant in low-resource deployment contexts
and that motivate the three-block architecture of § 5.1. The third
treats spatial transformer networks — the mechanism that
distinguishes the second of the three architectures examined here
from the baseline — with emphasis on their subsequent reception.
The fourth surveys the regularisation strategies that have shaped
how small-scale image classifiers are trained, and which enter the
present work as specific hyperparameter choices. The fifth, in
some ways the most important for the framing of this report,
addresses the reproducibility literature that has accumulated over
the past decade and the practices it recommends.

A brief tabular summary of benchmark-relevant prior work appears at
the end of the chapter.

## 2.1 The GTSRB benchmark: performance evolution 2012–2024

The benchmark was introduced as a competition at the 2011
International Joint Conference on Neural Networks, with the
associated dataset and baseline analyses subsequently published by
[Stallkamp et al. (2012)](references.md#stallkamp2012man). The
canonical reference reports three baseline classifiers — a
linear discriminant analysis, a random forest, and a multi-layer
perceptron operating on hand-designed features — alongside a
convolutional submission that achieved 98.52 % accuracy. Human
performance, estimated from a panel of test subjects, was reported
at 98.84 %, establishing the upper envelope against which
subsequent work has been evaluated.

The winning competition submission, described in more detail in
[Cireşan et al. (2012)](references.md#cirecan2012multi), introduced
a multi-column deep neural network that combined the outputs of
multiple convolutional columns trained on differently preprocessed
versions of the input. The reported test accuracy of 99.46 %
exceeded the human estimate and established the first meaningful
margin above it. Two aspects of that submission warrant attention
for the present work. First, the ensemble aspect of the
architecture — averaging predictions across columns with
independently initialised weights — anticipates the ensembling
strategy proposed for the forthcoming v0.3.0 release and discussed
in chapter 11. Second, the preprocessing diversity among the
columns foreshadows the subsequent recognition, established
through the data-augmentation literature, that training-time
variability in the input pipeline serves a similar function to
architectural variability in the prediction pipeline.

Subsequent work on the benchmark has broadly fallen into two
categories. The first consists of progressively deeper convolutional
architectures, typically transplanted with minor modifications from
architectures originally developed for ImageNet. Networks inspired
by the VGG family of [Simonyan and Zisserman
(2015)](references.md#simonyan2015very), the inception modules of
[Szegedy et al. (2015)](references.md#szegedy2015going), and the
residual connections of [He et al. (2016)](references.md#he2016deep)
have all been reported to reach accuracies in the 99.3 % to 99.8 %
range, though the exact ordering depends heavily on the training
protocol and augmentation pipeline adopted. The second category
consists of architectures that incorporate explicit geometric
priors, of which the spatial transformer network of [Jaderberg et
al. (2015)](references.md#jaderberg2015spatial) is the most
influential example and is discussed at greater length in § 2.3.

The aggregate pattern across this body of work is one of sustained
improvement until approximately 99.8 % test accuracy, followed by
an effective saturation. The remaining gap to 100 % is dominated
by images that the benchmark's own annotation conventions label
ambiguously, by artefacts of the capture process that leave some
images functionally uninterpretable, and by a small residue of
genuinely hard cases where the inter-class margin in pixel space
is thinner than the intra-class variability. The implication,
which colours the interpretation of the present results in chapter
8, is that differences between architectures of a tenth of a
percentage point at the 99.7 % regime are unlikely to reflect
meaningful differences in generalisation capacity.

## 2.2 Shallow convolutional baselines

The architectural idiom on which all subsequent work is built
originates in the seminal contribution of [LeCun et al.
(1998)](references.md#lecun1998gradient), which established the
LeNet-5 architecture for handwritten digit recognition. Three
structural choices from that work survive in the baseline
architecture of § 5.1: the alternation of convolution and spatial
subsampling, the use of local receptive fields to exploit
translation equivariance, and the terminal dense classifier. The
subsequent generalisation of these choices to colour imagery of
natural scenes by [Krizhevsky et al.
(2012)](references.md#krizhevsky2012imagenet) was accompanied by
quantitative changes — larger receptive fields, deeper stacks,
wider channel counts — that did not alter the underlying idiom.

For the purposes of traffic sign classification, architectures at
the shallow end of the spectrum remain empirically competitive.
The reason is one of capacity matching: the forty-three-class label
set is of modest cardinality, the training set contains
approximately thirty-nine thousand examples, and the effective
intra-class variability is bounded by the regulatory canonicalisation
of sign appearance. A three-block convolutional classifier with
between half a million and two million parameters can accommodate
this capacity budget without substantial overfitting, provided that
regularisation in the form of dropout ([Srivastava et al.,
2014](references.md#srivastava2014dropout)) and batch normalisation
([Ioffe and Szegedy, 2015](references.md#ioffe2015batch)) is
applied judiciously. The baseline architecture of § 5.1 is
constructed along exactly these lines.

A related argument, which becomes important in the interpretation of
the empirical results in chapter 8, is that overparameterisation
relative to the intrinsic complexity of the task is not neutral.
The classical bias–variance framing — under which additional
parameters trade bias for variance — has been qualified, though
not replaced, by the observation that modern architectures in the
overparameterised regime exhibit double-descent behaviour
([Belkin et al., 2019](references.md#belkin2019reconciling);
[Nakkiran et al., 2021](references.md#nakkiran2021deep)). The
practical consequence for the three-architecture comparison
developed here is that differences in test accuracy cannot be
predicted from parameter counts alone, and that empirical
comparison remains necessary.

## 2.3 Spatial transformer networks

The spatial transformer network, introduced by [Jaderberg et al.
(2015)](references.md#jaderberg2015spatial), addresses a specific
limitation of convolutional architectures: although convolutions are
equivariant to translation by construction, they are not
equivariant to the broader affine group of transformations that
includes rotation, scaling, shearing, and — for perspective
transformations — non-linear warpings. A spatial transformer is a
differentiable module that predicts, from the input image itself,
a set of affine or thin-plate-spline parameters and applies the
corresponding transformation before handing the warped result to
subsequent layers. The module is trained end-to-end with the rest
of the network and requires no explicit supervision of the
transformation.

The original contribution demonstrated improvements on three
benchmarks: a distorted variant of MNIST, a synthetic
street-number recognition task, and the CUB-200-2011 fine-grained
bird classification benchmark. On the distorted-MNIST variant,
where the learned transformation could plausibly canonicalise the
input, the improvements were substantial. On CUB-200-2011, where
the relevant geometric variation is of a different character, the
improvements were more modest. This conditional effectiveness has
persisted in subsequent applications: spatial transformers help
substantially when the dominant source of intra-class variability
is geometric, and marginally or not at all when it is not.

For traffic sign recognition, the relevant geometric variation
consists of perspective distortion arising from non-frontal camera
pose, scale variation due to range, and planar rotation induced by
camera tilt and sign installation. Whether the intrinsic
geometric content of GTSRB is of the kind a spatial transformer
can productively canonicalise is an empirical question rather than
a deductive one. The comparison in chapter 7 between the baseline
architecture and its spatial-transformer-augmented variant, both
trained under identical protocols, is the instrument by which this
question is addressed in the present work.

A secondary consideration concerns the localisation subnetwork
within the spatial transformer. The prediction of transformation
parameters requires a small auxiliary convolutional network whose
training signal is indirect: gradients flow through the grid
sampling operation and back into the localisation parameters. The
stability of this training signal depends on initialisation choices
that the original paper discusses only briefly. In the present
work the localisation head is initialised to the identity
transformation, following the recommendation of [Jaderberg et al.
(2015)](references.md#jaderberg2015spatial), with the effect that
the spatial transformer begins training as a no-op and acquires
non-trivial transformations only to the extent that they reduce
the classification loss.

## 2.4 Regularisation strategies for small-scale classification

Four regularisation techniques appear in the training configuration
of § 6 and together account for much of the generalisation
performance achieved on GTSRB. Each has an established literature
which is briefly traced here.

**Dropout.** Introduced by [Srivastava et al.
(2014)](references.md#srivastava2014dropout), dropout randomly
zeroes a fraction of activations during training and rescales the
remaining activations so that the expected value is preserved. The
mechanism can be understood as approximate ensembling over an
exponential number of thinned subnetworks, though the variance
reduction it achieves empirically exceeds what the naive ensemble
interpretation would predict. In the present work, dropout is
applied both in the convolutional feature extractor, with rates
increasing with depth, and in the dense classifier, following
conventional practice.

**Batch normalisation.** [Ioffe and Szegedy
(2015)](references.md#ioffe2015batch) introduced batch
normalisation as a remedy for what the authors termed "internal
covariate shift"; subsequent analysis has complicated this
interpretation without displacing the empirical utility of the
technique. The effect of batch normalisation on training dynamics
is pronounced: learning rates can be set an order of magnitude
higher without divergence, and the training loss typically becomes
markedly smoother. In the present work, batch normalisation is
applied immediately after every convolution in all three
architectures, and immediately after the first dense layer of the
classifier in the baseline architecture.

**Weight decay and adaptive optimisers.** The decoupled
weight-decay formulation of [Loshchilov and Hutter
(2019)](references.md#loshchilov2019decoupled) replaces the L2
penalty implicit in classical SGD with momentum by a separate
multiplicative decay applied after the gradient step. For adaptive
optimisers such as Adam ([Kingma and Ba,
2015](references.md#kingma2015adam)), this distinction is
non-trivial: the L2-penalty formulation interacts with the
per-parameter learning-rate normalisation in ways that the
decoupled formulation does not. The broader question of whether
adaptive optimisers are indeed preferable to well-tuned SGD has
been investigated by [Wilson et al.
(2017)](references.md#wilson2017marginal), whose controlled
comparisons across several image classification benchmarks find
that the generalisation advantage sometimes attributed to adaptive
methods is largely attributable to differences in learning-rate
tuning rather than to the adaptive update rule itself. The
practical implication for the present work is that the choice of
Adam or AdamW over SGD in § 6 is defended on engineering grounds —
ease of hyperparameter transfer across architectures — rather than
on a claim of superior generalisation. In the present work, the
deeper of the three architectures is trained with decoupled
weight decay (via the AdamW optimiser), while the shallower
architectures use plain Adam without weight decay; the choice is
documented in § 6 and justified by the differing susceptibilities
to overfitting.

**Data augmentation.** A broad survey is provided by [Shorten and
Khoshgoftaar (2019)](references.md#shorten2019survey). For the
traffic sign case, two pipeline choices warrant specific attention.
The first is the use of random affine transformations — small
rotations, translations, and scalings — which simulate the
geometric variability of on-road capture without introducing
label-corrupting changes. The second, which is discussed at
length in § 6 and § 9, concerns the explicit exclusion of
horizontal flipping from the augmentation pipeline. Flipping is
standard practice for many image-classification benchmarks, but
the presence of directional signs in GTSRB — arrow markings, the
"no entry" sign, asymmetric priority signs — means that a flipped
example is incorrectly labelled. The exclusion is a narrow but
non-trivial methodological choice whose motivation is
task-specific.

A broader pattern visible across these four techniques is that
their interactions with architecture are not additive. The
combination of batch normalisation and dropout, for example, has
been observed empirically to produce results that are not
predictable from either technique in isolation, and the ordering
of the two modules within a block affects behaviour in ways that
the original papers do not address. The three architectures in
this report therefore represent specific, locally optimised
configurations rather than instances of a single parameterisable
family.

## 2.5 Reproducibility in applied deep learning

Over the past decade, the deep-learning literature has accumulated
a parallel literature on its own reproducibility problems. Three
strands of this literature inform the experimental protocol of
§ 4 and the engineering discussion of § 9.

The first concerns protocol ambiguity. [Pineau et al.
(2021)](references.md#pineau2021improving) reports on the NeurIPS
2019 reproducibility program, in which accepted submissions were
subjected to independent reimplementation. The central finding
was that numerical reproducibility was achieved in a minority of
cases, with the most frequent cause of failure being underspecified
training protocols — details of data preprocessing, hyperparameter
sweeps, and selection criteria that were not reported in the
original submission. The conclusion that follows, and that is
adopted in this work, is that complete specification of the
training protocol is a precondition for meaningful comparison
across models.

The second concerns variance due to random seeds. [Bouthillier et
al. (2021)](references.md#bouthillier2021accounting) quantifies the
variability in reported metrics arising from seed choice alone, and
finds that it is of the same order of magnitude as the differences
typically reported in the literature to substantiate architectural
improvements. The implication is that single-seed comparisons
between architectures are, in the absence of unusually large
effect sizes, statistically uninformative. This work adopts the
standard of documenting a single seed for the baseline release
(v0.2.0) while explicitly scheduling multi-seed comparison for the
follow-up release described in § 11; the reasoning for the
separation is that the three-architecture comparison conducted
here is of sufficient effect size that the seed-variability lower
bound does not swamp the signal.

The third concerns engineering practices. [Wilson et al.
(2014)](references.md#wilson2014best) articulates a set of
recommendations for scientific computing — version control,
automated testing, reproducible environments — that predate the
deep-learning boom but apply directly to it. [Sculley et al.
(2015)](references.md#sculley2015hidden) characterises the specific
form that technical debt takes in machine-learning systems, with
particular attention to dependency entanglement and configuration
drift. The engineering rationale documented in chapter 9 draws on
both references. The general argument, and one of the explicit
motivations for this project, is that the gap between
research-prototype code and reproducible artefact is often wide,
and that the closure of that gap is itself a methodologically
non-trivial activity.

A fourth and more recent strand, which the present work engages
with obliquely, concerns the statistical treatment of the benchmark
comparisons themselves. [Dror et al.
(2018)](references.md#dror2018hitchhiker) argues for the routine
use of appropriate statistical significance tests in
machine-learning comparisons, in analogy with long-established
practice in natural language processing. [Gundersen and Kjensmo
(2018)](references.md#gundersen2018state) surveys the
reproducibility landscape more broadly and proposes a graded
definition of reproducibility levels that is consistent with the
protocol adopted here. Statistical significance testing is
deferred in the present work to the multi-seed follow-up, where
the underlying distributions can be characterised more directly
than is possible from single-seed runs.

## 2.6 Summary and positioning

The five strands traced in this chapter converge on the specific
research questions articulated in § 1.3. The benchmark-evolution
literature establishes that the 98–99 % accuracy regime is where
the GTSRB benchmark has operated for more than a decade, and that
differences within this regime cannot be interpreted naively as
architectural superiority. The literature on shallow convolutional
baselines and on spatial transformers supplies the specific
architectural alternatives that the present comparison contrasts.
The regularisation literature provides the technical context for
the hyperparameter choices documented in § 6, including the
task-specific exclusion of horizontal flipping. The reproducibility
literature supplies the protocol standards against which the
present work measures itself and motivates the separation of the
baseline release from the planned multi-seed follow-up.

Against this background, the present work does not introduce a new
architecture, a new dataset, or a new training technique. Its
contribution is methodological: a controlled three-way comparison
conducted under a reproducible protocol, reported with the
statistical caveats that the literature warrants, and accompanied
by an open-source software artefact that permits independent
verification of the numerical claims. The effect of this framing
is to situate the work not as a claimed improvement on the state
of the art, but as an instrument for answering a specific and
tractable methodological question.

**Table 2.1.** Selected prior work on GTSRB and related benchmarks,
ordered by publication year.

| Reference | Architecture family | Reported test accuracy | Year |
|---|---|---:|---:|
| [LeCun et al.](references.md#lecun1998gradient) | LeNet-5 (MNIST, not GTSRB) | 99.05 % (MNIST) | 1998 |
| [Krizhevsky et al.](references.md#krizhevsky2012imagenet) | AlexNet (ImageNet, not GTSRB) | 84.7 % top-5 (ImageNet) | 2012 |
| [Stallkamp et al.](references.md#stallkamp2012man) | Convolutional baseline | 98.52 % | 2012 |
| [Cireşan et al.](references.md#cirecan2012multi) | Multi-column DNN | 99.46 % | 2012 |
| [Simonyan and Zisserman](references.md#simonyan2015very) | VGG (ImageNet-transplanted) | 99.3 %–99.6 % (reported range) | 2015 |
| [Jaderberg et al.](references.md#jaderberg2015spatial) | Spatial transformer (various) | variable by benchmark | 2015 |
| [He et al.](references.md#he2016deep) | ResNet (ImageNet-transplanted) | 99.5 %–99.8 % (reported range) | 2016 |

Accuracy ranges are reported as cited in the secondary literature;
primary references sometimes report only the best of several runs,
which is one of the reasons the methodological observations of § 2.5
are relevant.
