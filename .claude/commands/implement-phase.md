# /implement-phase

Implement exactly one phase from `project-1-rag-qa-plan.md`.

## Arguments
`$ARGUMENTS` — phase number or name, e.g. `Phase 2`.

## Procedure
1. Read `CLAUDE.md`.
2. Read `project-1-rag-qa-plan.md`.
3. Run the prerequisite check skill `.claude/skills/phase-prerequisite-check/SKILL.md`.
4. If the result is `BLOCKED`, stop and report blockers. Do not write implementation code.
5. If the result is `READY`, use the skill `.claude/skills/implement-rag-milestone/SKILL.md`.
6. Implement only the requested phase. Do not implement later phases.
7. Add or update relevant tests in the same pass.
8. Update relevant docs in the same pass when setup, commands, behavior, API, or architecture changes.
9. Run `make test` (or `pytest tests -m "not slow"` if make is unavailable).
10. Report changed files, tests run, docs updated, and any blockers.

## Phase Boundary Rules
- Do not implement later phases.
- Do not add optional improvements outside the requested phase scope.
- Do not install packages unless explicitly approved.
- Do not add secrets or real `.env` values.
