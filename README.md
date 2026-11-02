# Personal Learning Library

Personalized learning materials organized by subject. Follow [AGENTS.md](AGENTS.md) for repository structure, authoring standards, and progress tracking.

## Subjects

| Subject | Curriculum | Current contents |
| --- | --- | --- |
| Deep learning | [Architectural Intuition and Engineering Judgment](subjects/deep-learning/CURRICULUM.md) | [HTML course contents](subjects/deep-learning/index.html) · [Read Module 1](subjects/deep-learning/lessons/01-from-conventional-ml-to-differentiable-models.html) · [Read Module 2](subjects/deep-learning/lessons/02-making-networks-trainable.html) · [Progress](subjects/deep-learning/PROGRESS.md) |

## Organization

Each subject lives in `subjects/<subject-slug>/`, with its syllabus in `CURRICULUM.md`. Add `lessons/`, `simulations/`, `assessments/`, and `assets/` only when needed. Simulations and quizzes require explicit opt-in under AGENTS.md.

Add `LEARNER_PROFILE.md` at the repository root and subject-specific `SUBJECT_PROFILE.md` files when preferences are supplied. Track content status and learner status separately in a subject's `PROGRESS.md` when lesson authoring begins.

Subject curricula, subject profiles, and progress files are kept locally and ignored by Git. Links to those files work when the local copies are present; a fresh clone does not include them.
