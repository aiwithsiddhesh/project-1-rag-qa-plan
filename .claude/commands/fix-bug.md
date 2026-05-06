# /fix-bug

Fix a specific bug with root-cause analysis and regression coverage.

## Arguments
`$ARGUMENTS` — the observed failure or incorrect behavior.

## Procedure
1. Read `CLAUDE.md`.
2. Read `project-1-rag-qa-plan.md`.
3. Inspect relevant code and tests.
4. Use the skill `.claude/skills/test-rag-system/SKILL.md`.
5. Reproduce or localize the bug where practical.
6. Add a regression test where practical.
7. Fix the root cause without changing intended behavior outside the bug scope.
8. Update docs if behavior, errors, setup, API contract, or usage changed.
9. Run relevant tests.
10. Report before/after behavior, changed files, and verification results.

## Constraints
- Keep scope limited to the bug.
- Do not perform unrelated refactors.
- Do not introduce real LLM/API/network calls into default tests.
