# /review-rag

Review RAG-specific retrieval and generation quality for the current implementation.

## Procedure
1. Read `CLAUDE.md`.
2. Read `project-1-rag-qa-plan.md`.
3. Use the skill `.claude/skills/rag-quality-review/SKILL.md`.
4. Spawn the `rag-architect` subagent from `.claude/agents/rag-architect.md` for an independent second opinion on retrieval and generation correctness.
5. Review: retrieval, BM25+dense fusion, RRF, deduplication, reranking, HyDE, prompt grounding, citations, context budgeting, and no-context behavior.
6. Recommend tests or RAGAS eval scenarios where needed.

## Output Required
- RAG correctness bugs.
- RAG quality risks.
- Missing tests or eval scenarios.
- Concrete recommendations with file:line references.
