# Claude Code Commands

Custom slash commands for this project. Invoke in Claude Code as `/command-name <args>`.

## Commands

- `/prepare-phase <phase>` — Check prerequisites only. No file edits.
- `/implement-phase <phase>` — Run prerequisite check then implement exactly the requested phase.
- `/fix-bug <description>` — Localize and fix a bug with regression coverage.
- `/review-branch` — Review current branch for bugs, plan compliance, missing tests, and secrets.
- `/review-rag` — Review RAG retrieval and generation quality.
- `/refactor-safely <target>` — Refactor a target without changing behavior.
- `/review-pr-comments <PR number>` — Fetch all review comments on a PR, check each against the plan, apply valid ones with inline replies, and reject out-of-scope ones with a reasoned explanation.

## How `/review-pr-comments` works

1. Fetches all inline review thread comments and review body comments from the PR.
2. For each comment, reads the relevant phase spec in `project-1-rag-qa-plan.md`.
3. Classifies the comment as **APPLY**, **APPLY-WITH-MODIFICATION**, or **REJECT**.
4. For APPLY/APPLY-WITH-MODIFICATION: applies the fix, commits, and replies **inline** to the thread with `@reviewer` tagging.
5. For REJECT: replies inline with the specific plan section that makes the change out of scope.
6. Pushes all commits and resolves applied threads.
7. Reports final PR merge readiness.
