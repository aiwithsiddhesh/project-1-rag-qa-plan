# /review-branch

Review current branch changes before merging or moving to the next phase.

## Procedure
1. Read `CLAUDE.md`.
2. Read `project-1-rag-qa-plan.md`.
3. Run `git diff main...HEAD` and `git log main...HEAD --oneline` to see the full changeset.
4. Review findings first, ordered by severity.
5. Check plan compliance — do changes stay within the current phase scope?
6. Check whether implementation includes relevant tests.
7. Check whether docs were updated when behavior, setup, API, or architecture changed.
8. Check for secret leaks, `.env` values, generated vectorstores, generated reports, and large local docs.
9. Check for unintended real LLM/API/network calls in default tests.
10. For complex changes, spawn specialist subagents from `.claude/agents/`:
    - `rag-architect.md` for changes in `src/retriever.py`, `src/reranker.py`, `src/pipeline.py`.
    - `test-strategist.md` for changes in `tests/`.
    - `api-reviewer.md` for changes in `api/`.

## Output Required
- Findings first with file:line references.
- Open questions or assumptions.
- Brief change summary only after findings.
- If no findings, state that explicitly and list residual risks or testing gaps.
