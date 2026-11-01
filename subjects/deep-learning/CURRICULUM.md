# Deep Learning: Architectural Intuition and Engineering Judgment

**Personalised curriculum for Cheng He**  
**Level:** Intermediate machine learning → deep learning  
**Core duration:** 12 weeks, approximately 4 hours per week  
**Format:** Conceptual explanations, visual reasoning, selective mathematics and small experiments  
**Version:** 1 November 2026

## 1. Purpose

Develop a durable understanding of how neural networks work, why their structures suit particular problems, and how to adapt that understanding to unfamiliar applications.

The central question throughout the course is:

> What problem does this architectural choice solve, what assumption does it introduce, and when does that assumption help or hurt?

The course prioritises independent reasoning over memorising architectures or completing coding tutorials. It is a syllabus, not the lesson content itself.

## 2. Starting point and scope

The curriculum assumes practical experience with XGBoost, scikit-learn, predictive modelling and gene-expression analysis, supported by engineering training, research experience and technical leadership.

It assumes familiarity with supervised learning, feature engineering, training and validation splits, cross-validation, standard evaluation metrics and basic overfitting. These topics are revisited only when deep learning introduces a meaningful difference.

Experience with neural-network internals is not assumed. Tensors, computation graphs, gradients and representation learning receive a compact but substantive foundation. Mathematical refreshers are introduced where needed rather than as a separate prerequisite course.

Applications are predominantly outside biology: photography and computer vision, audio, industrial inspection, demand forecasting, document search and recommendation. Biological examples may occasionally provide a bridge to familiar territory.

## 3. Learning outcomes

By the end of the core course, the learner should be able to:

1. Trace inputs, intermediate representations, predictions and gradients through a small network.
2. Explain the roles of locality, parameter sharing, recurrence, gating, residual paths, bottlenecks and attention.
3. Distinguish architecture, training objective, optimisation method and inference procedure.
4. Explain why CNNs, ResNets, LSTMs, transformers and encoder–decoder designs suit particular tasks, including their limitations.
5. Compare plausible architectures using data structure, sample size, computational cost and deployment requirements.
6. Reason about representations learned through supervised, self-supervised and transfer learning.
7. Diagnose training behaviour and propose controlled experiments or ablations.
8. Read an unfamiliar architecture diagram or model proposal and assess its design and supporting evidence.

## 4. Course design

### Principles before model names

Architectural families and reusable building blocks are treated separately. CNNs, ResNets and LSTMs are not mutually exclusive categories: residual connections, for example, can appear in multiple network families. The course repeatedly examines how components combine.

### Consistent learning sequence

Each architecture module follows six steps:

1. Identify the motivating problem and a simpler baseline.
2. Inspect the structure through a visual walkthrough.
3. Trace the mechanism with selected equations and tensor shapes.
4. Identify the assumptions that make it fit an application.
5. Compare limitations and credible alternatives.
6. Test the explanation through a small experiment or ablation.

### Typical weekly allocation

| Activity | Approximate time |
| --- | ---: |
| Conceptual explanation and visual walkthrough | 90 minutes |
| Worked examples and selective mathematics | 45 minutes |
| Guided experiment or code inspection | 60 minutes |
| Retrieval, comparison and design discussion | 45 minutes |
| **Total** | **4 hours** |

Coding occupies approximately 25% of the course. PyTorch is the experimental language. The practical target is to read a forward pass, follow tensor shapes, understand a training loop and modify a component. Large engineering projects are not required.

## 5. Core syllabus at a glance

| Week | Module | Main emphasis |
| --- | --- | --- |
| 1 | From conventional ML to differentiable models | Composition, learned features, tensors and gradients |
| 2 | Making networks trainable | Gradient flow, optimisation and stabilising components |
| 3 | Representation learning and geometry | Embeddings, similarity and objective-dependent features |
| 4–5 | CNNs and ResNets in depth | Locality, sharing, receptive fields and residual design |
| 6 | Sequence modelling and memory | RNNs, LSTMs, GRUs and temporal convolutions |
| 7–8 | Attention and transformers in depth | Learned interactions, position, masking and transformer blocks |
| 9 | Structured outputs and architectural composition | Encoder–decoder designs, U-Net and multiscale information |
| 10 | Learning from unlabelled data | Reconstruction, contrastive learning and masked prediction |
| 11 | Reusing and adapting pretrained models | Transfer, fine-tuning, LoRA and multimodal alignment |
| 12 | Investigating and selecting architectures | Ablations, model critique and a final design exercise |

## 6. Module specifications

### Module 1 — From conventional ML to differentiable models

**Duration:** Week 1

**Topics:** Learned versus engineered features; multilayer perceptrons; affine transformations and nonlinearities; depth and width; tensor dimensions; computation graphs; gradients; backpropagation and automatic differentiation.

**Key questions:** What does adding a layer change? Why are nonlinearities necessary? How does changing an early parameter affect the final loss?

**Guided activity:** Trace one small network through a forward pass and parameter update; inspect activations and gradients.

**Completion evidence:** Explain the computation and learning process in a small network, including the shapes of its intermediate quantities.

### Module 2 — Making networks trainable

**Duration:** Week 2

**Topics:** Initialisation; activation scale; vanishing and exploding gradients; SGD, momentum and Adam; learning rates and schedules; batch and layer normalisation; residual connections; weight decay and dropout in the context of neural-network training.

**Key questions:** Why might a deeper network train worse? Which observations suggest an optimisation problem? Which intervention should be tested first?

**Guided activity:** Investigate a poorly training network by changing one factor at a time and inspecting loss, activations or gradients.

**Completion evidence:** Produce a diagnostic hypothesis and a controlled experiment, distinguishing trainability from generalisation.

### Module 3 — Representation learning and geometry

**Duration:** Week 3

**Topics:** Hidden representations; distributed features; embedding spaces; dot products and cosine similarity; scale and normalisation; bottlenecks; task-dependent geometry; limits of interpreting low-dimensional visualisations.

**Key questions:** What makes a representation useful? Why can two models organise the same inputs differently? When is similarity meaningful?

**Guided activity:** Compare raw features and learned embeddings for an image or document retrieval task.

**Completion evidence:** Explain how an objective and representation affect which inputs appear similar, without treating an embedding plot as definitive evidence.

### Module 4 — CNNs and ResNets in depth

**Duration:** Weeks 4–5

**Topics:** 1D and 2D convolutions; kernels and channels; parameter sharing; stride, padding, pooling and dilation; receptive fields; hierarchical features; equivariance and invariance; global pooling; residual blocks, identity paths and projection shortcuts.

**Comparisons:** Dense networks versus CNNs; plain CNNs versus residual CNNs; local versus global information.

**Applications:** Photography, industrial defect detection, audio and sensor signals.

**Guided activities:** Inspect feature maps and receptive fields; compare architectures under a controlled training setup; test sensitivity to translation and broader context.

**Completion evidence:** Justify convolutional and residual design choices and identify tasks where their assumptions may be restrictive. Recognise that residual connections support trainability without guaranteeing better accuracy.

### Module 5 — Sequence modelling and memory

**Duration:** Week 6

**Topics:** Recurrent state; unrolling through time; backpropagation through time; basic RNN limitations; LSTM and GRU gates; bidirectionality; temporal convolutional networks; causal versus noncausal processing.

**Comparisons:** RNN versus LSTM/GRU; recurrent versus temporal convolutional processing; streaming versus whole-sequence tasks.

**Applications:** Sensor monitoring, speech and demand forecasting.

**Guided activity:** Compare sequence models on a small task with a controllable dependency length.

**Completion evidence:** Trace how history is retained or discarded and distinguish long-range modelling capability from a guarantee of learning long-range dependencies.

### Module 6 — Attention and transformers in depth

**Duration:** Weeks 7–8

**Topics:** Queries, keys and values; scaled dot-product attention; multihead attention; positional information; causal and padding masks; feed-forward blocks; residual paths and normalisation; encoder, decoder and encoder–decoder variants; vision patches; computational and memory costs.

**Comparisons:** Attention versus recurrence and convolution; causal prediction versus bidirectional encoding; text tokens versus image patches.

**Applications:** Document understanding, translation, image analysis and sequence prediction.

**Guided activities:** Trace a small attention operation numerically; inspect a transformer block and its tensor shapes; change a mask or positional component and predict the consequence.

**Completion evidence:** Explain information flow through a transformer and justify its use against alternatives. Distinguish an attention visualisation from evidence of causal explanation.

### Module 7 — Structured outputs and architectural composition

**Duration:** Week 9

**Topics:** Encoders and decoders; bottlenecks; downsampling and upsampling; U-Net; multiscale information; skip connections; additive versus concatenative paths; resolution and alignment.

**Applications:** Scene segmentation, image restoration, denoising and sequence-to-sequence prediction.

**Guided activity:** Inspect a small encoder–decoder and compare reconstruction or segmentation with and without selected skip connections.

**Completion evidence:** Design an information path appropriate to an output's structure, and distinguish the typical roles of U-Net skips and residual additions.

### Module 8 — Learning from unlabelled data

**Duration:** Week 10

**Topics:** Autoencoders; reconstruction objectives; masked prediction; contrastive learning; positive and negative pairs; augmentation choices; representation collapse; evaluation of learned features.

**Applications:** Image and audio representations, document retrieval and recommendation.

**Guided activity:** Compare representations learned with different objectives or augmentation choices, using a simple downstream evaluation.

**Completion evidence:** Propose a self-supervised task and explain what useful information it may preserve, what it may discard and how to test transfer.

### Module 9 — Reusing and adapting pretrained models

**Duration:** Week 11

**Topics:** Frozen encoders; linear probes; partial and full fine-tuning; adapters and LoRA; catastrophic forgetting; data and compute requirements; multimodal alignment and fusion; conceptual distinctions between continued pretraining, supervised fine-tuning and preference optimisation.

**Applications:** Image classification with limited labels, document search and image–text retrieval.

**Guided activity:** Compare a frozen representation with a limited fine-tuning strategy, or inspect a prepared comparison where local compute is insufficient.

**Completion evidence:** Select an adaptation strategy and justify its expected benefit, resource requirements and evaluation plan. Separate weight adaptation from providing external information at inference time.

### Module 10 — Investigating and selecting architectures

**Duration:** Week 12

**Topics:** Architecture selection; baselines; sample efficiency; shortcut learning; robustness; distribution shift; ablations; reproducibility; parameter count versus computation; latency, memory and throughput; critical reading of model claims.

**Guided activity:** Review an unfamiliar architecture proposal or paper, then complete the final design exercise.

**Completion evidence:** Defend a modelling choice, identify competing explanations for its performance and specify evidence that would change the recommendation.

## 7. Mathematical depth

| Area | Expected depth |
| --- | --- |
| Linear algebra and tensors | Follow matrix operations, dot products, broadcasting and shape changes in key architectures |
| Calculus | Work through backpropagation in one tiny network; understand the chain rule and gradient flow |
| Probability and objectives | Interpret cross-entropy, reconstruction, contrastive and masked-prediction objectives |
| Optimisation | Explain update rules and training behaviour at a practical level |
| Complexity | Estimate parameter counts and reason about major compute and memory costs |

Formal convergence proofs, detailed generalisation bounds and lengthy derivations are outside the core scope. Prior mathematical training determines the pace of refreshers, not whether the concepts are omitted.

## 8. Assessment and progression

Assessments are short and based on reasoning rather than memorisation:

- **Explain:** Describe a component's purpose and mechanism in plain language.
- **Trace:** Follow information, gradients or tensor shapes through a small example.
- **Predict:** State what should happen before altering a component.
- **Investigate:** Design an ablation that distinguishes plausible explanations.
- **Transfer:** Apply the same principle to a different domain.

Progression checkpoints:

| Point | Expected capability |
| --- | --- |
| End of Week 3 | Trace learning and reason about representations |
| End of Week 6 | Compare convolutional, residual and recurrent designs |
| End of Week 9 | Read transformer and encoder–decoder diagrams and explain their information paths |
| End of Week 12 | Propose, critique and test an architecture for an unfamiliar task |

### Final design exercise

Choose one non-biological application, such as industrial defect segmentation, equipment-failure forecasting or document retrieval. Produce a short modelling proposal covering:

1. Input and output structure, available data and operational requirements.
2. A credible simple baseline.
3. A proposed architecture and the assumptions behind its components.
4. A training objective and any pretraining or adaptation strategy.
5. Validation and error-analysis plans.
6. At least two informative ablations or comparisons.
7. Compute, memory and latency tradeoffs.
8. Failure modes and evidence that would trigger a different approach.

A full implementation is optional. Success is the quality of the reasoning and proposed evidence.

## 9. Optional electives

Each elective is approximately 1–2 additional weeks; deeper projects can extend it.

| Elective | Content | Purpose |
| --- | --- | --- |
| Graph neural networks | Message passing, aggregation, graph attention, permutation properties and oversmoothing | Extend architectural reasoning to relational data |
| Generative model families | VAEs, GANs, autoregressive generation and diffusion | Compare generative objectives, mechanisms and tradeoffs |
| Recommendation and ranking | Two-tower models, learned retrieval, ranking objectives and feedback bias | Apply embeddings and architecture choices to user–item problems |
| Advanced time series | Temporal convolutions, recurrent and transformer approaches, covariates and forecast horizons | Develop deeper sequence-modelling judgment |
| Efficient inference | Quantisation, distillation, batching, caching and profiling | Connect architectural choices to deployment constraints |

## 10. Instructional guidance

- Keep CNNs, ResNets, LSTMs and transformers as the architectural anchors.
- Reuse a small number of examples so comparisons remain interpretable.
- Use non-biological applications by default.
- Introduce equations when they clarify a mechanism; accompany them with visual or numerical reasoning.
- Prioritise inspecting and modifying compact examples over building large pipelines.
- Include credible alternatives and failure cases in every architecture discussion.
- Distinguish established mechanisms, empirical observations and simplified intuitions.
- Adapt the pace to demonstrated understanding, using checkpoints rather than credentials alone.

**Completion standard:** Independently explain a network's structure, assess its fit to a problem, and design experiments that test the explanation.
