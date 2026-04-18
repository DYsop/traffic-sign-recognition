# 1. Introduction

Traffic sign recognition occupies a well-defined niche in applied
computer vision. On the one hand, the task is narrow enough to be
treated as a closed-world multi-class classification problem: a
finite, culturally fixed vocabulary of signs is to be mapped to
discrete labels under the implicit assumption that every input belongs
to one of those classes. On the other hand, the input distribution
encountered in practice is substantially harder than the classroom
framing suggests. Signs appear at arbitrary distances, under varying
illumination, partial occlusion, motion blur, and with the full
spectrum of weather-induced degradation. It is this combination — a
closed label set against an open-world input distribution — that makes
traffic sign recognition a productive testbed for the empirical
evaluation of convolutional architectures.

The present work revisits the problem not with the intent of claiming a
new state of the art, but with the intent of asking a more specific
question: at the 98–99 % accuracy regime that the German Traffic Sign
Recognition Benchmark now routinely admits, what is the empirical
relationship between model capacity and generalisation performance on
held-out data? This question, and the reproducible experimental
apparatus assembled to answer it, constitute the contribution of this
report.

## 1.1 Motivation

Perception is the first of the three traditional subsystems of an
autonomous driving stack, preceding planning and control. Within
perception, classification of static infrastructure — traffic lights,
lane markings, and traffic signs — stands apart from general object
detection in that the set of possible labels is bounded a priori by
the relevant national or regional traffic code. For the German case
addressed here, that code enumerates forty-three sign classes whose
visual identity has been standardised by decades of regulatory and
engineering practice.

The attractiveness of the classification subtask for methodological
study stems from three properties. First, the ground truth is
unambiguous: a sign either belongs to a given class or it does not,
and the regulatory basis of each class supplies a stable canonical
appearance against which deviations can be measured. Second, the
classes are visually rich yet finite, permitting the evaluation of
architectures whose parameter counts range from hundreds of thousands
to tens of millions without the label-space explosion that
complicates analogous studies on ImageNet-scale benchmarks. Third,
the cost of error is operationally consequential: misclassifying a
thirty-kilometre-per-hour limit sign as a fifty-kilometre-per-hour
limit sign is not a benign failure mode in any realistic deployment,
and this consequentiality focuses attention on the tail of the
error distribution rather than on aggregate accuracy alone.

The pedagogical and methodological utility of the problem has long
been recognised. The earliest substantial treatment of convolutional
recognition applied to digit recognition, itself a closed-world
classification task with visual similarity among confusable pairs
([LeCun et al., 1998](references.md#lecun1998gradient)), established
the architectural vocabulary — convolution, local receptive fields,
spatial pooling — that was subsequently transplanted, with
modifications, to the traffic sign setting. The transplant proved
fruitful: a decade later, purpose-built convolutional classifiers had
closed most of the gap to human performance on the relevant
benchmark, a result first documented at scale by [Cireşan et al.
(2012)](references.md#cirecan2012multi).

What that closure did not resolve, however, is a question that has
become more pointed with the subsequent proliferation of
overparameterised architectures: at the upper end of the benchmark,
where reported accuracies cluster between 98 % and 99.8 %, is the
observed ordering of models a stable property of the task, or an
artefact of the specific validation set used to select among them?
The present report addresses this question in a controlled setting,
with the deliberate choice of three architectures that span an order
of magnitude in parameter count.

## 1.2 The GTSRB benchmark

The German Traffic Sign Recognition Benchmark, introduced as a
competition at the 2011 International Joint Conference on Neural
Networks and subsequently described in the canonical reference of
[Stallkamp et al. (2012)](references.md#stallkamp2012man), provides
the empirical substrate for this work. The benchmark consists of
colour images of forty-three classes of German traffic signs, drawn
from video sequences captured during real driving. A full dataset
description is deferred to chapter 3; for the purposes of this
introduction, three of its properties warrant mention.

The first is longevity. The benchmark has remained in active use
since 2012, which means that its behaviour under a wide variety of
architectural families — from shallow convolutional classifiers
through inception-style networks to transformer-based models — is
reasonably well characterised. This longitudinal coverage makes it
useful as an instrument even in the absence of any claim to
contemporary novelty.

The second is moderate scale. With roughly thirty-nine thousand
training images and twelve thousand six hundred and thirty test
images, the benchmark occupies an intermediate regime: large enough
that naive memorisation is not a plausible strategy, yet small enough
that full training runs can be conducted on a single consumer-grade
graphics processing unit in a matter of minutes. This property is
instrumentally important for the present work, which depends on the
ability to train and evaluate three architectures under matched
protocols repeatedly.

The third is pronounced class imbalance. The number of training
images per class ranges from approximately two hundred to
approximately two thousand, a ratio of roughly ten to one between
the most and least populated classes. This imbalance is not
extreme, but it is sufficient to make the difference between
accuracy and class-balanced metrics empirically visible, and
accordingly motivates the use of the Matthews correlation
coefficient and Cohen's kappa alongside raw accuracy in the
evaluation that follows.

The aggregate effect of these three properties is that the benchmark
functions as a well-characterised, computationally tractable, and
metrically informative testbed. It is neither a toy problem nor a
contemporary frontier, but a stable instrument against which specific
methodological questions can be posed and answered.

## 1.3 Research questions

The central question of this work is whether, under matched training
protocols, additional model capacity produces measurable gains in
test-set generalisation at the 98–99 % accuracy regime. This
question is operationalised through three concrete subquestions:

*Q1. Accuracy ordering.* Given three architectures that differ by
approximately one order of magnitude in parameter count, is the
ranking induced by held-out test accuracy monotonic in capacity?

*Q2. Validation–test consistency.* Does the architecture that
maximises validation accuracy also maximise test accuracy, or do the
two metrics diverge in a systematic way that depends on model
capacity?

*Q3. Error structure.* Do higher-capacity architectures fail on the
same classes as the baseline, or on different classes? The answer
bears on whether capacity produces uniform improvement or shifts the
error surface from one region of input space to another.

These questions are deliberately modest. None of them asks for a new
architectural innovation; none claims to test a frontier technique.
Their value derives from the controlled setting in which they are
answered. The three architectures are held to identical data
splits, identical random seeds, identical augmentation pipelines
where feasible, and identical evaluation protocols. The resulting
numerical answers, reported in chapters 7 and 8, are therefore
comparable in a sense that results drawn from the literature — where
authors report under whatever protocols best serve their claims — are
typically not.

## 1.4 Contributions

The contributions of this report fall into three categories.

**First, a controlled empirical comparison.** Three architectures
spanning an order of magnitude in parameter count — a three-block
baseline convolutional network, a variant augmented with a spatial
transformer front-end, and a deeper five-block network — are trained
and evaluated under a matched protocol and reported with a common
set of metrics. The finding, anticipated in chapter 7 and discussed
in chapter 8, is that the smallest of the three architectures
achieves the highest test-set accuracy, a result which runs counter
to the intuition that would order the three models by capacity.

**Second, a reproducible software artefact.** The training, evaluation,
and inference apparatus is released as a Python package under an
open-source licence, configured through typed configuration files
rather than ad-hoc scripts. The package is accompanied by a
non-trivial test suite, continuous integration across three Python
versions, and deterministic seed management. The engineering choices
underlying this artefact are documented in chapter 9, with particular
attention to three defects uncovered in the original research-prototype
code base during the refactoring process.

**Third, a methodological observation about validation–test
divergence.** The three architectures examined here produce
near-identical validation accuracies, but their test accuracies
differ by approximately one percentage point. This divergence is
consistent across the evaluation, and its magnitude is comparable to
the differences typically reported in the literature to substantiate
claims of architectural superiority. The observation, developed in
chapter 8, does not contradict those reported claims, but it does
suggest that single-seed validation-set rankings at the 99 %
saturation regime should be interpreted with greater caution than is
customary. This observation motivates the planned follow-up work,
described in chapter 11, which employs multi-seed reporting and
ensemble methods as more stable instruments.

## 1.5 Document structure

The remainder of the report is organised as follows. Chapter 2
situates the contribution within the prior literature on traffic
sign recognition, spatial transformer networks, and reproducibility
in applied deep learning. Chapter 3 characterises the GTSRB dataset
in detail, with emphasis on the properties that influence the
experimental design. Chapter 4 formalises the experimental
protocol, including the definition of the data splits and the
rationale for the metrics adopted. Chapter 5 describes the three
architectures with sufficient precision for independent reproduction.
Chapter 6 documents the training setup, including the hyperparameter
choices that differ between architectures and the augmentation
pipeline that is held constant across them. Chapter 7 reports the
numerical and visual results without interpretation. Chapter 8
interprets those results, with particular focus on the
validation–test divergence that motivates the research questions of
§ 1.3. Chapter 9 documents the engineering rationale of the
accompanying software artefact and the defects uncovered during
refactoring. Chapter 10 provides a reproducibility protocol.
Chapter 11 delineates the planned extensions. Full bibliographic
information is collected in `references.md`.

The report has been written with two readers in mind. The first is
the practitioner who wishes to understand, in technical detail, what
was done and why. For this reader, chapters 3 through 8 constitute
the technical core; chapters 9 and 10 supply the engineering and
reproducibility context. The second is the researcher or hiring
manager who wishes to evaluate the quality of the work in aggregate.
For this reader, chapter 7 and § 9.2 may be read in isolation and are
self-contained with respect to the claims they substantiate. In
either case, the intermediate treatments of batch normalisation
([Ioffe and Szegedy, 2015](references.md#ioffe2015batch), cited in
chapter 5), dropout ([Srivastava et al.,
2014](references.md#srivastava2014dropout), cited in chapter 6),
and the general methodology of modern deep learning
([Goodfellow et al., 2016](references.md#goodfellow2016deep),
cited throughout) are assumed as background rather than explained
in the text. The same assumption applies to the convolutional
architectural idiom itself, whose historical treatment by
[Krizhevsky et al. (2012)](references.md#krizhevsky2012imagenet) is
taken as sufficient context for what follows.
