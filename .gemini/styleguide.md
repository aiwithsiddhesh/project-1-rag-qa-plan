# Code Review Style Guide

Review this repository as a senior AI engineering portfolio project for a Document Q&A RAG system.

## Project Rules

- Treat `project-1-rag-qa-plan.md` as the source of truth.
- Check that changes stay within the requested implementation phase.
- Flag any implementation that jumps ahead to later phases.
- Flag unrelated refactors or changes outside the requested scope.
- Do not recommend committing secrets, local `.env` values, generated vectorstores, generated reports, or large local documents.

## Python Standards

- Prefer typed Python with explicit dependencies and small testable functions.
- Prefer Pydantic settings for configuration.
- Use the custom RAG exception hierarchy for domain failures.
- Use structured logging and avoid logging secret values or full document contents.
- Keep default tests deterministic and free of paid API, OpenAI, LangSmith, vectorstore, or network calls.

## RAG Quality Checks

- Check chunk metadata, source attribution, citation behavior, and no-context behavior.
- Check retrieval quality logic for dense retrieval, BM25, Reciprocal Rank Fusion, deduplication, reranking, and HyDE when those phases are implemented.
- Check prompt grounding and context budgeting when generation is implemented.
- Require RAGAS and other slow evals to be marked on-demand, not part of default tests.

## Review Output

- Lead with concrete bugs, regressions, security risks, missing tests, and plan violations.
- Include file and line references when possible.
- Keep style-only comments low priority unless they affect maintainability or correctness.
