---
name: implement-rag-milestone
description: Implement one exact project phase from project-1-rag-qa-plan.md after prerequisite checks pass. Use for phase-scoped RAG system implementation with relevant tests, docs, and verification while avoiding later-phase work.
---

# Implement RAG Milestone Skill

Use this skill to implement one exact phase from `project-1-rag-qa-plan.md` after prerequisite checks pass.

## Goal
Implement only the requested project phase with relevant tests and docs.

## Procedure
1. Confirm the phase prerequisite check returned `READY`.
2. Read the requested phase from `project-1-rag-qa-plan.md`.
3. Define the exact phase scope before editing any files.
4. Implement only files required by that phase.
5. Add or update relevant tests in the same pass.
6. Update relevant docs in the same pass if setup, commands, public behavior, API behavior, or architecture changed.
7. Run `pytest tests -m "not slow"` (or `make test` if available).
8. Report changed files, docs updated, tests run, verification result, and blockers.

## Scope Rules
- Do not implement later phases.
- Do not add optional improvements not requested by the phase.
- Do not install packages unless explicitly approved.
- Do not add secrets or real `.env` values.
- Do not commit generated vectorstores, generated reports, or large local docs.

## Testing Rules
- Use unit tests with mocks for external services.
- Mock LLMs and paid APIs in default tests.
- Add integration or slow eval tests only when the phase calls for them.

## Documentation Rules
- Docs are part of implementation when behavior or usage changes.
- Do not claim planned features are complete before they are implemented.
