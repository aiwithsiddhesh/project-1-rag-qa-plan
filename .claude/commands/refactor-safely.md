# /refactor-safely

Refactor a target without changing behavior.

## Arguments
`$ARGUMENTS` — module, package, function, or behavior to refactor, e.g. `src/pipeline.py`.

## Procedure
1. Read `CLAUDE.md`.
2. Read `project-1-rag-qa-plan.md`.
3. Use the skill `.claude/skills/test-rag-system/SKILL.md`.
4. Inspect existing tests for the target first.
5. Add characterization tests if the target is under-tested and behavior could drift.
6. Refactor only the requested target.
7. Preserve public behavior and interfaces unless explicitly requested.
8. Update docs only if public usage or behavior changes.
9. Run relevant tests.
10. Report changed files and verification results.

## Constraints
- Do not mix refactor with feature work.
- Do not make unrelated cleanup edits.
- Do not change public interfaces without explicit instruction.
