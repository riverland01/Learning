# Module 1 editorial revision — 2 November 2026

## Objective and scope

Make the existing deep-learning Module 1 understandable without requiring the learner to reconstruct the model, notation or missing reasoning. Preserve its approved outcomes, mathematical depth, worked example, stable lesson paths and offline HTML format. No simulation, quiz or curriculum change is part of this revision. Authored material is not evidence of learner mastery.

## Observations and counter-rules

The learner supplied the examples below in conversation. The central failure is an explanation that makes the reader infer connections that the author should supply. The categories overlap; they are editorial checks, not independent diagnoses.

| Passage or concern | Issue | Rule that counters it |
| --- | --- | --- |
| Contrast has “the same predicted effect” at every edge density | Model and assumptions unstated; interaction left implicit; could suggest equal coefficients | Show the additive model, define its distinct coefficients, and explain that an interaction means one feature's effect depends on another |
| “Early stages build measurements; later stages combine them” | Metaphor hides computation; execution order confused with fitting order | State layer roles directly; distinguish sequential evaluation from joint parameter learning |
| “The analogy has a limit” and no “unique internal explanation” | Analogy needs immediate repair; abstract caveat lacks a clear referent | Remove a burdensome analogy; explain concretely that learned features need not map to human-named concepts |
| “Start with the stage…” | Abrupt transition; unidentified component | Explain why the next operation is being examined and connect it to the familiar linear predictor |
| “Put the d input values… into… x” | Arrangement before meaning | Define one observation and its features before introducing its vector and orientation |
| “m intermediate measurements” and rows of weights | Terminology drift; matrix structure precedes unit computation | Explain one hidden unit's calculation, connect it to a learned feature, then explain one row per unit |
| W1 and W2 roles and number of layers unclear | Missing overall architecture; parameter matrices confused with layers | Introduce one hidden layer and one output layer first; identify the parameters each uses and the layer-count convention |
| “The restriction of the original linear regression model” | Unclear backward reference | Restate the exact restriction where it matters, then show the affine-collapse calculation |
| “Input-dependent participation”; boundary “can move” | Abstract shorthand and ungrounded geometry | State zero versus positive ReLU output; define the separating equation and what shifting or rotating means |
| Tensor paragraph appears abruptly | Definition without purpose or continuity | Motivate batches and shape checking, then define tensor through concrete row/column meanings |
| “Libraries… rank… different from matrix rank” | Vague attribution and irrelevant, unexplained distinction | Omit the rank detour here; introduce software terminology only when it helps a current interpretation or action |
| Single vector `(2,)` follows a column-vector derivation | Silent convention change | Distinguish mathematical column-vector orientation from software array shapes when used; explain transposes for row-batched data |
| “Let B be its size… each example… X” | Ambiguous pronoun, synonym switch, undefined symbol | Define B as observation count and X as the batch matrix; consistently use observation; define entries and axes |

The operational rules now live in [AGENTS.md](../AGENTS.md), under “Editorial rules from learner feedback.” Responses to future flagged passages should propose the revision first, then identify the issues and connect them to recurring patterns.

## Enforcement options and decision

| Option | Benefits | Costs and limitations |
| --- | --- | --- |
| Rules in AGENTS.md with author review only | Discoverable, canonical, little coordination cost | Authors can overlook the same implicit assumptions during self-review |
| Separate editorial agents required for every edit | Another reader can identify missing context | Repeated context loading and review overhead for minor corrections; agent agreement is not technical verification |
| Canonical rules plus independent review for new lessons and substantial rewrites | Combines explicit author obligations with a fresh review where structural problems are likely | Requires a bounded review and integration step for substantial work; unavailable review must be disclosed |

Choose the third option. Every substantive change receives an author editorial pass. New lessons and substantial rewrites receive a separate read-only editorial agent review when tools are available. Minor targeted changes do not need another agent unless they reveal a broader problem. Findings must be resolved and affected passages checked before completion. Without required independent review, content remains drafted. Calculations, references and rendering still require their own checks.

No separate agent instruction file is needed: it would repeat the canonical contract and create a maintenance risk. The review task points directly to AGENTS.md, the curriculum, profiles and the complete proposed lesson. The existing quality-standard wording was also changed so it does not implicitly require an analogy.

## Work and acceptance tracking

| Task | Owner and scope | Dependencies | State / evidence |
| --- | --- | --- | --- |
| T1: identify issues and compare enforcement | Root; separate policy reviewer, read-only | Learner examples and current AGENTS.md | Completed; mapping and decision above; reviewer identified the contradictory analogy default |
| T2: implement editorial contract | Root; AGENTS.md | T1 | Completed; eight operational rules and scaled review obligations |
| T3: revise complete lesson | Lesson editor; Markdown only | Original lesson, profiles, curriculum, learner feedback | Completed; entire lesson revised, original worked example retained |
| T4: independent editorial review | Separate reviewer; read-only | T2 and T3 | Completed; two findings resolved and rechecked |
| T5: synchronise HTML and verify | Root; HTML, calculations, links, rendering and progress | T3 and T4 | Completed; results below |
| T6: commit and push | Root; selected tracked files | Accepted T1–T5, personal identity confirmed by learner | Final publication step; Git history and the completion message record the result |

| Criterion | Responsible tasks | Status / evidence |
| --- | --- | --- |
| C1: all reported patterns mapped to actionable rules | T1–T2 | Met; mapping and AGENTS.md |
| C2: whole lesson revised without losing outcomes or depth | T3–T4 | Met; full editorial review against Module 1 curriculum; forward, backward and update derivations retained |
| C3: computations, sources and editions consistent | T5 | Met; independently recalculated values, six references revisited, HTML generated from revised Markdown |
| C4: offline reading, navigation and responsive layout checked | T5 | Met; desktop and phone browser checks, local links and shared CSS checked |
| C5: review and authoring recorded without changing learner status | T4–T5 | Met; this report and local PROGRESS.md |
| C6: requested changes committed and pushed | T6 | Assessed after this artifact is committed; remote verification is reported in the completion message |

## Review findings and verification

The independent reviewer read the complete revised lesson, curriculum, subject profile and updated rules, including the later gradient, update and automatic-differentiation sections. Two remaining issues were corrected: array gradients are now explained before their first use, and the change from column-vector biases to software shape `(m,)` is explained before the batch equation. The reviewer rechecked both corrected passages and confirmed resolution. No further material findings were reported within that review scope.

The rewrite makes the overall architecture explicit, connects hidden features to feature engineering, explains interactions and activation boundaries concretely, and motivates tensors through batching. Later sections also received clearer transitions and terminology. The original curriculum, numerical example and depth remain intact. The Markdown and HTML use the same revised text, with the existing offline theme, navigation and accessible static diagrams retained.

Verification performed:

- Recomputed all nine parameter gradients using central finite differences; largest absolute discrepancy was approximately `2.47e-10`. Prediction/loss before the update: `3.5 / 1.125`; after: `1.5575 / 0.097903125`. Rechecked parameter counts, conditional ReLU response and the broadcasting counterexample.
- Revisited all six existing reference pages on 2 November 2026, including the version-specific PyTorch Linear documentation and autograd tutorial. The source access date is updated in both editions.
- Checked 47 HTML links, including local files, reference anchors and navigation targets; no duplicate IDs or missing local targets. Local curriculum/profile/progress links continue to require the Git-ignored local files, as documented in README.md.
- HTML contains 189 MathML expressions with no browser-reported MathML errors. The embedded shared CSS matches the canonical theme. No script or external stylesheet is needed to read the lesson.
- Opened the file in an offline headless Edge browser at `1440×1000` and `390×844`. Checked no page-wide horizontal overflow, one visible diagram appropriate to each layout, eight lesson sections, sidebar navigation, course-contents navigation and return. Print-media navigation hiding passed; no page JavaScript errors occurred. Visually inspected the desktop reading view and phone diagram.
- The first phone-layout check caught missing equation scroll containers after conversion. Added the wrappers and repeated the affected browser checks successfully.

No executable PyTorch lesson or learner assessment was generated or run. The pseudocode remains explicitly explanatory. Verification establishes the stated editorial and technical checks, not that the learner now finds every passage clear or has mastered the material. The local learner status remains `not-attempted`.
