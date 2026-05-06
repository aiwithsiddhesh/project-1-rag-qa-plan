---
name: rag-quality-review
description: Review retrieval and generation quality for the project RAG system. Use to check chunk metadata, dense and BM25 retrieval, Reciprocal Rank Fusion, deduplication, reranking, HyDE, prompt grounding, citations, context budgeting, no-context behavior, and eval coverage.
---

# RAG Quality Review Skill

Use this skill to review retrieval and generation quality for the RAG system.

## Goal
Find RAG-specific correctness and quality risks that normal code review may miss.

## Review Areas
- Document chunking metadata: `source_file`, `file_type`, `chunk_index`, `total_chunks`.
- Dense retrieval behavior and top-k/fetch-k usage.
- BM25 retrieval behavior and tokenization assumptions.
- Hybrid fusion with Reciprocal Rank Fusion — `score = Σ 1/(rank + 60)` per doc.
- Deduplication behavior and stable document identity (page_content hash).
- CrossEncoder reranking order and fallback behavior.
- HyDE toggle behavior and cost/quality tradeoff.
- Prompt grounding and instruction hierarchy.
- Context budget (3000-char) and truncation behavior.
- Citation extraction and source display.
- No-context or out-of-domain fallback behavior.
- Prompt injection resistance.
- RAGAS eval coverage where appropriate.

## Output Format
- Correctness bugs.
- Quality risks.
- Missing tests or eval scenarios.
- Concrete recommendations with file:line references.

## Rules
- Separate verified bugs from quality tradeoffs.
- Do not require expensive evals for every change.
- Prefer targeted tests before broad changes.
