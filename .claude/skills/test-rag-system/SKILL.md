---
name: test-rag-system
description: Design and verify tests for the project RAG system. Use during implementation, bugfix, review, or refactor work to keep unit tests deterministic, mock paid services and network calls, separate integration tests, and mark RAGAS evals slow.
---

# Test RAG System Skill

Use this skill when implementation, bugfix, review, or refactor work requires tests.

## Goal
Keep the test suite useful, fast by default, and free of accidental paid/network calls.

## Test Layers
- **Unit tests**: mock LLMs, vectorstores, external APIs, and network services. Fast and deterministic.
- **Integration tests**: may use local embeddings/vectorstore behavior, but must mock paid LLM calls.
- **RAGAS/eval tests**: slow/on-demand only, marked `@pytest.mark.slow`.

## Procedure
1. Identify the behavior being changed.
2. Prefer unit tests for business logic and validation.
3. Add regression tests for bugfixes where practical.
4. Add integration tests only when component boundaries matter.
5. Add RAGAS/eval tests only for retrieval/generation quality gates or when requested by the phase.
6. Verify default test commands do not require OpenAI keys, LangSmith keys, internet access, or paid services.
7. Run `pytest tests -m "not slow"`.

## Rules
- Do not hide real LLM/API calls inside default tests.
- Do not make tests depend on local secrets.
- Use fixtures and mocks for deterministic behavior.
- Keep tests aligned with `project-1-rag-qa-plan.md`.
