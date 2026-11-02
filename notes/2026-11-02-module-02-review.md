# Module 2 authoring and verification — 2 November 2026

## Scope and ownership

Author the approved deep-learning Module 2, “Making networks trainable,” as one core lesson with an editable Markdown source and a self-contained offline HTML reading edition. Follow the existing shared theme and curriculum; update course availability and local progress. No simulation, quiz, training experiment or learner assessment was requested or generated. The root LEARNER_PROFILE.md is absent; the curriculum's starting-level statement and the local subject preferences supplied learner context.

The root agent owned the complete explanatory sequence, source integration, revisions, rendering and publication. Orchestrate was used for bounded research and independent editorial review. Both agents were read-only and remained leaves; their completed findings were collected before finalisation.

| Task | Owner | Dependency | Completion evidence |
| --- | --- | --- | --- |
| T1: verify consequential mechanisms and sources | Separate research agent; root checks | Curriculum, profiles and Module 1 conventions | Primary-source findings for initialisation, optimisers, normalisation, residuals and regularisation integrated; 12 references in lesson |
| T2: write integrated lesson | Root | Orientation and T1 evidence | Complete lesson connects gradient flow to interventions and a controlled diagnostic comparison |
| T3: review complete explanation independently | Separate editorial reviewer | Complete T2 draft and repository editorial rules | Four concrete findings resolved; reviewer rechecked affected passages with no unresolved findings |
| T4: verify editions, calculations and reading experience | Root | Revised T2/T3 | Checks below passed |
| T5: record and publish | Root | Accepted T1–T4 | Local PROGRESS.md updated without changing learner status; Git commit and remote state reported in completion message |

## Editorial findings and resolution

The reviewer read the complete lesson against Module 2 outcomes, subject presentation preferences, the editorial contract and relevant Module 1 conventions. The author also checked transitions, symbol introductions, distinctions and technical consequences throughout the draft.

1. The initialisation table assumed fan-in 64, whereas Module 1 used two input features. The worked model now explicitly receives 64 precomputed image features and has six hidden layers of width 64, making the first-layer assumption explicit.
2. A two-observation BatchNorm example obscured batch dependence. The replacement compares `[1, 3, 5]` with `[1, 3, 9]`, deriving why the unchanged first observation's normalised value changes from approximately −1.225 to −0.981.
3. The matrix update ratio used vector/spectral 2-norm notation while describing the Frobenius norm. It now uses the Frobenius subscript and defines the summed-square calculation.
4. Permission-related authoring language was removed from the lesson. The diagnostic example is presented directly as hypothetical teaching material.

The reviewer independently rechecked these changes and the added SGD, schedule and clipping references. No unresolved findings remained within that review scope. The review does not establish learner comprehension or mastery.

## Acceptance and verification

| Criterion | Status | Evidence and limits |
| --- | --- | --- |
| Approved scope and level | Met | Initialisation, activation scale, vanishing/exploding gradients, SGD/momentum/Adam, rates/schedules, BatchNorm/LayerNorm, residuals, weight decay and dropout; controlled diagnostic reasoning distinguishes fitting from generalisation |
| Technical and source discipline | Met | Separate source investigation and root source checks; 12 primary papers/documentation pages linked beside relevant claims through reference anchors; hypothetical observations explicitly labelled |
| Numerical consistency | Met | Recomputed scalar gradient products; He standard deviation and scale recurrence; SGD and bias-corrected Adam first steps; both BatchNorm examples; weight-decay step; dropout expectation; residual cancellation. Quadratic stability condition checked algebraically |
| Source and reading editions agree | Met | Pandoc MathML conversion; section-by-section rendered text equivalence across all 11 sections, including equations/tables; no missing or duplicate anchor IDs; 37 local HTML links checked |
| Offline HTML and navigation | Met | Canonical CSS embedded, 174 MathML expressions, five tables; no scripts, remote assets or external stylesheet required. Offline headless Edge at desktop 1440×1000 and phone 390×844; no page-wide overflow, MathML errors or page errors; equation scroll wrappers, section navigation, reference target, course contents and return link passed |
| Visual and print checks | Met | Inspected desktop and phone screenshots of the normalisation section; equations legible and reading layout follows existing theme. Print-media navigation hiding checked; no full printed-page pagination audit |
| Progress and library navigation | Met | Course index now offers Module 2 and reports two available modules; README links to the lesson; local content marked verified, learner remains not-attempted |

Browser launch was blocked inside the sandbox and succeeded with an approved outside-sandbox run. A verification assertion initially treated an empty reference span as a visible element; it was corrected to inspect its containing reference item. All final browser checks passed. Display equations were also lifted out of paragraph wrappers to preserve valid block nesting before final rendering checks.

No empirical neural-network training was conducted. Diagnostic results are conditional interpretations, not reported experiments. No PyTorch runtime behaviour is claimed. Source investigation, editorial agreement, numerical recalculation and browser checks provide different evidence and do not substitute for learner feedback.
