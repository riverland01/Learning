# Applied mathematics editorial revision — 3 November 2026

The learner prefers concrete engineering mathematics with purposeful equations and intermediate reasoning, rather than either theoretical abstraction or long verbal substitutes for calculations. AGENTS.md rule 10 now encodes that editorial standard. The Git-ignored subject profile records the learner's reported background and preference; the approved curriculum is unchanged.

## Scope and review

Module 1's backward pass now derives the effect of changing one output weight before collecting array gradients. It states what is held fixed, shows the exact prediction change, connects it to an approximate loss change, and explains the chain-rule multiplication. The two factors equal to 1.5 are explicitly distinguished. Hidden activation, ReLU and hidden-weight derivatives follow individual computations before matrix notation. The hypothetical perturbation is distinguished from the actual optimiser update.

Module 2 now introduces a concrete scalar connection before the layer-wide gradient recurrence, explains the weighted summation of contributions and adds a scalar residual derivative before its matrix form. The new preactivation symbol is explicitly identified. Existing outcomes, numerical update, source references, lesson paths and scope are preserved. No simulations, quizzes or training experiments were added.

One writer owned Module 1 Markdown; the root owned Module 2, the canonical editorial rule, local profile/progress and HTML integration. A separate read-only editorial reviewer inspected both revised mathematical explanations and rule 10. It found no material issues and suggested naming Module 2's new preactivation explicitly; that clarification was incorporated. All dispatched work was collected before publication.

## Completion evidence

- Recomputed the new weight perturbation: prediction increase 0.015; first-order loss increase 0.0225; actual loss increase 0.0226125. The original parameter gradients and update were retained and checked by the lesson writer. Checked Module 2's scalar derivatives 6 and 3.6 independently.
- Regenerated both HTML articles from Markdown, preserving navigation, existing diagrams and embedded theme. Complete converted article text matches the reading editions. All local file targets and internal anchors resolve with no duplicate IDs. Module 1 has 250 MathML expressions and 36 links; Module 2 has 216 expressions and 52 links.
- Offline headless Edge checks at 1440×1000 and 390×844 passed for both modules: no page-wide overflow or MathML error elements; section navigation and course contents/return navigation work. Visually inspected the phone backward-pass reading view. These are targeted rendering checks, not a new full print-pagination audit.
- Local PROGRESS.md records the revision while preserving learner status. The subject profile and progress remain Git-ignored as required. No external factual claims or sources were added; the changes derive existing elementary calculations rather than requiring a new literature investigation.

Editorial and numerical verification do not establish learner mastery. Git history and the completion message record the commit/push result.
