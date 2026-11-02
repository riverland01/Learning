# Modules 1 and 2 — editorial follow-up, 2 November 2026

## Scope and accepted feedback

Update the canonical writing style guide in AGENTS.md and review both existing deep-learning modules against the latest learner feedback. Preserve curriculum scope, full derivations, stable reading paths, offline presentation and learner status. No simulations, quizzes, new modules or empirical training runs are included.

The latest observations add three concrete checks to the previous editorial contract:

| Observation | Issue | Editorial response |
| --- | --- | --- |
| Why are the ReLU weights equal to one? | Illustrative parameters can be mistaken for general assumptions | Name the chosen weights and bias, explain why those numbers were used, and connect the numerical expression to the general equation |
| How do we choose hidden width m? | Defining a configurable quantity does not explain who chooses it or how | Distinguish hyperparameters from learned weights, justify the teaching value and give proportionate selection guidance and trade-offs |
| What do “storage convention” and “role” mean? | Abstract assurances conceal which array changes and what operation stays the same | Name the biases, show column versus one-axis shapes and explain the per-unit bias addition for every observation |

The style guide remains inside AGENTS.md. No duplicate guide or agent configuration is needed. Rules 3 and 6 were strengthened, rule 9 added, and the author-review checks extended. The existing writer/researcher/editorial responsibilities and opt-in rules remain in force.

## Orchestration and acceptance

| Task | Owner / write scope | Dependency | State |
| --- | --- | --- | --- |
| T1: update editorial contract | Editorial agent / AGENTS.md only | Learner feedback and existing rules | Completed; root inspected diff |
| T2: revise Module 1 | Lesson writer / Module 1 Markdown only | Profiles, curriculum, existing lesson and feedback | Completed; independently reviewed |
| T3: review and revise Module 2 | Root / Module 2 Markdown | Profiles, curriculum, existing lesson and updated rules | Completed; independently reviewed |
| T4: independent editorial review | Separate reviewer / read-only both full lessons | T1–T3 integrated drafts | Completed; three findings resolved and rechecked |
| T5: synchronise and verify | Root / HTML editions, calculations, links and local progress | Accepted T2–T4 | Completed; checks below passed |
| T6: commit and push | Root / selected tracked changes | Accepted T1–T5 | Final publication step; completion message records result |

The Module 1 writer and root each own one complete lesson's narrative. Editorial review is independent of both lesson writers. Research checks are bounded to the revised claims and existing source context; a separate research agent is unnecessary for this foundational clarification task. All agents remain leaves and have disjoint write scopes.

| Criterion | Responsible tasks | Status / evidence |
| --- | --- | --- |
| C1: actionable rules for all three new patterns | T1 | Met; AGENTS.md rules 3, 6 and 9 |
| C2: both full lessons reviewed, needed revisions integrated | T2–T4 | Met; independent full-lesson review and affected-passage recheck |
| C3: numerical and source checks support changed content | T5 | Met; arithmetic checks and targeted primary-source verification |
| C4: HTML/Markdown agree; offline desktop and phone reading work | T5 | Met; text/equation correspondence, links and four browser configurations |
| C5: progress records checks without inferring mastery | T5 | Met; local PROGRESS.md updated; learner statuses unchanged |
| C6: selected changes committed and pushed | T6 | Assessed after commit; reported in completion message |

## Revisions and findings

Module 1 now identifies width as a hyperparameter at first use, explains the chosen two-unit example, gives an explicitly illustrative validation-based width comparison, states the ReLU example's weights and bias, and explains bias-array shapes through a numerical broadcasting example.

Module 2 now distinguishes model parameters, hyperparameters and optimiser running averages; explains example widths and starting scales; grounds momentum and Adam notation in their actual calculations; identifies the chosen learning-rate, retention, normalisation and dropout values; and gives scoped guidance for selecting rates, regularisation and clipping settings. It also replaces vague normalisation “sharing” wording with the exact axes and reuse of scale/offset parameters. The worked diagnosis retains its hypothetical status and six-layer comparison.

## Independent findings and resolution

The reviewer read both complete revised lessons, updated AGENTS.md, curriculum and subject profile. Module 1 had no additional material findings. Three Module 2 findings were resolved:

1. The normalisation equation now identifies its output before use and explains that its symbol is an intermediate activation rather than a training target.
2. The projection shortcut now defines its learned matrix and input/output dimensions before using it in the residual addition.
3. The comparison between accumulated and averaged momentum now states that its exact scale relationship assumes the same gradient sequence, fixed retention factor and zero initial accumulation. Separate training runs can generate different gradients.

The independent reviewer inspected all three revised passages and confirmed resolution. Review covered the updated prose and equations; numerical execution, source retrieval and browser checks were performed separately by the root.

## Verification results and limits

- Module 1: 8 sections, 201 MathML expressions, 3 tables and 36 HTML links. Module 2: 11 sections, 195 MathML expressions, 5 tables and 52 links. All local targets, reference anchors and stable section anchors resolve; no duplicate IDs. Git-ignored local curriculum/profile/progress links still require the local files, as documented in README.md.
- Both HTML editions were regenerated from the revised Markdown. Checked complete article text and every equation's source annotation against the converted source. Reused canonical embedded CSS and existing navigation/diagrams. Consolidated repeated Module 1 citations into one numbered entry per source; the source has seven reference definitions and the HTML now has seven reference items.
- Independently recomputed Module 1's nine finite-difference gradients (largest absolute discrepancy approximately `2.47e-10`), update, parameter counts, ReLU interaction and bias/broadcasting examples. Updated prediction/loss remain `1.5575 / 0.097903125`.
- Recomputed Module 2's scalar gradient products, He scale and six-layer recurrence, SGD/Adam example, accumulated-versus-averaged momentum relationship, both normalisation examples, decay update, dropout expectation, residual cancellation and clipping cases above/below threshold. These are arithmetic checks, not trained-model results.
- Targeted source checks covered the changed guidance and implementation distinctions: Stanford CS231n, official scikit-learn MLP guidance and version-specific PyTorch 2.9 Adam, SGD, BatchNorm and Dropout documentation. Added the scikit-learn reference to Module 1 and the Adam documentation reference to Module 2. Illustrative search ranges remain explicitly author-selected, not attributed to a source prescription. Existing unrelated reference claims were not subjected to a new full literature review.
- Offline headless Edge checks passed at desktop `1440×1000` and phone `390×844` for both lessons: no page-wide overflow, MathML error elements or page errors; display equations have scroll containers; section/reference links, course contents/return navigation and print navigation hiding passed. Inspected the desktop width-selection view and phone normalisation view. The first phone check exposed a long inline decay calculation; it was moved into a scrollable display equation and affected checks repeated successfully.
- Updated local content-progress notes while leaving both learner statuses unchanged. No curriculum or presentation preference was changed. No simulation, quiz, empirical training run or executable PyTorch lesson was generated; no learner mastery is inferred from the revision or review. Print navigation was checked, not full printed-page pagination.

Git history and the completion message record the publication result, avoiding a self-referential commit hash inside the committed report.
