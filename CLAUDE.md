# CLAUDE.md

## Project Mission
Build the Document Q&A RAG system described in `project-1-rag-qa-plan.md` as a senior-level AI engineering portfolio project. The plan file is the source of truth for architecture, phases, test strategy, and interview-focused design decisions.

## Operating Rules
- Read `project-1-rag-qa-plan.md` before implementing, reviewing, refactoring, or documenting project behavior.
- Implement phase-by-phase only. Do not jump ahead to later phases.
- Before implementing any phase, use the skill `.claude/skills/phase-prerequisite-check/SKILL.md`.
- If prerequisites are blocked, stop and report blockers instead of writing implementation code.
- Keep changes scoped to the requested command or phase.
- Do not modify unrelated files.
- Preserve user changes. Never revert work you did not make unless explicitly requested.

## Engineering Standards
- Use typed Python and clear module boundaries.
- Prefer small, testable functions and explicit dependencies.
- Use Pydantic settings for configuration.
- Use the custom exception hierarchy described in the plan.
- Use structured logging and safe text truncation for logs.
- Keep unit tests deterministic and fast.

## Testing Rules
- Every feature implementation must include relevant tests in the same pass.
- Every bugfix should include a regression test where practical.
- Unit tests must mock LLMs, external APIs, vectorstores, and network services.
- Default tests must not call paid APIs or real OpenAI/LangSmith services.
- Integration tests may use local embeddings/vectorstores when appropriate but should mock paid LLM calls.
- RAGAS/eval tests must be marked slow/on-demand.

## Documentation Rules
- Update relevant docs during implementation or bugfixes when setup, commands, public behavior, API behavior, architecture, or user-facing workflows change.
- Do not claim planned features are implemented before they exist.
- Prefer verified commands in docs; otherwise mark commands as planned.

## Safety Rules
- Do not commit or create secrets in repo files.
- Do not create `.env` with real values.
- Do not commit generated vectorstores, large local documents, generated reports, tokens, or credentials.
- Do not install packages unless explicitly approved.
- Do not run destructive commands.

## Commands
Slash commands live in `.claude/commands/*.md`. Invoke them as `/command-name <args>`.

- `/prepare-phase <phase>` — Check prerequisites only. No file edits.
- `/implement-phase <phase>` — Run prerequisite check then implement the requested phase with tests and docs.
- `/fix-bug <description>` — Localize and fix a bug with regression coverage.
- `/review-branch` — Review current changes for bugs, missing tests, docs drift, secrets, and plan compliance.
- `/review-rag` — Review RAG-specific retrieval and generation quality.
- `/refactor-safely <target>` — Refactor a target without behavior changes.
- `/review-pr-comments <PR number>` — Fetch all review comments on a PR, check each against the plan, apply valid ones, and reply inline with reasoning.

## Skills
Internal procedures live in `.claude/skills/*/SKILL.md`. They are invoked by commands via the Skill tool.

- `phase-prerequisite-check` — Determine if a phase is ready to implement.
- `implement-rag-milestone` — Implement one exact project phase after prerequisites pass.
- `rag-quality-review` — Review RAG retrieval and generation quality.
- `test-rag-system` — Design and verify the test suite.
- `review-pr-comments` — Evaluate PR review comments against the plan and handle each one.

## Agents
Specialist subagent personas live in `.claude/agents/*.md`. Spawn via the Agent tool when reviewing complex RAG, test, or API changes.

- `rag-architect` — RAG architecture, retrieval, reranking, HyDE, prompt grounding, citations.
- `test-strategist` — Test coverage, mocks, fixtures, slow eval separation.
- `api-reviewer` — FastAPI contracts, schemas, middleware, Docker healthchecks.

## GitHub Workflow Rules
- Always reply to PR review comments inline using `gh api repos/.../pulls/<PR>/comments/<id>/replies`.
- Always tag the reviewer (e.g. `@gemini-code-assist`) at the start of reply comments.
- Do not edit existing comments — post new ones.
- Resolve review threads after applying the fix using the GraphQL `resolveReviewThread` mutation.
- Check `required_conversation_resolution` branch protection before declaring a PR ready to merge.
