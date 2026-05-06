# /prepare-phase

Check whether the requested project phase is ready to implement. Read-only — no file edits.

## Arguments
`$ARGUMENTS` — phase number or name, e.g. `Phase 1` or `Phase 2 - Ingestion Layer`.

## Procedure
1. Read `CLAUDE.md`.
2. Read `project-1-rag-qa-plan.md`.
3. Use the skill `.claude/skills/phase-prerequisite-check/SKILL.md`.
4. Locate the requested phase exactly in the plan.
5. Identify required previous phases and required repo state.
6. Inspect the repo without editing files.
7. Report one of:
   - `READY: <phase> can be implemented`
   - `BLOCKED: <phase> is missing prerequisites`

## Output Required
- Requested phase and goal.
- Required previous phases.
- Existing repo state.
- Missing prerequisites (if any).
- Exact implementation scope.
- Recommended next command.

## Constraints
- Do not edit, create, delete, or overwrite any files.
- Do not install packages.
- Do not implement source code.
- Do not run destructive commands.
