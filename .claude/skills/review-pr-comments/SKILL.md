---
name: review-pr-comments
description: Fetch all review comments on a PR, check each against project-1-rag-qa-plan.md, apply plan-aligned ones with inline replies, and reject out-of-scope ones with a reasoned explanation referencing the plan.
---

# Review PR Comments Skill

## Goal
For every review comment on a PR, determine whether it aligns with `project-1-rag-qa-plan.md`, then act on it with a reply that tags the reviewer and references the plan.

## Inputs
- PR number.
- `project-1-rag-qa-plan.md` (source of truth).
- `CLAUDE.md` (operating rules).

## Step 1 — Collect All Comments

Run both of these to get the full picture:

```bash
# Inline review thread comments (line-level)
gh api repos/{owner}/{repo}/pulls/{PR}/comments

# Review body summaries + general PR comments
gh pr view {PR} --json reviews,comments
```

For each comment note:
- `id` (for inline replies)
- `path` and `line` (for inline comments)
- `body` (the feedback text)
- `author.login` (for tagging)
- Whether it is inline (`path` present) or a review body comment

## Step 2 — Check Each Comment Against the Plan

For each comment:

1. Read the relevant section of `project-1-rag-qa-plan.md` for the affected file or phase.
2. Classify the comment as one of:

   **APPLY** — The comment identifies a real bug or gap that is consistent with the plan spec. The fix does not change the design, add out-of-scope features, or contradict plan decisions. Example: a correctness bug in a validator.

   **APPLY-WITH-MODIFICATION** — The comment is valid but the suggested implementation differs from the plan or has its own flaw. Apply a corrected version and explain the difference in the reply.

   **REJECT** — The comment proposes a design change that directly contradicts a plan decision, adds out-of-scope features, or is simply not required by the plan. Example: changing an eager singleton to a lazy loader when the plan explicitly requires `settings = Settings()` at module level.

3. Write a one-sentence verdict with a plan reference before acting.

## Step 3 — Act on Each Comment

### For APPLY and APPLY-WITH-MODIFICATION
1. Apply the code change (use Edit tool).
2. Run `pytest tests -m "not slow"` to verify.
3. If tests pass, commit:
   ```
   git add <files>
   git commit -m "<concise message explaining the fix>"
   ```
4. Reply inline to the thread:
   ```bash
   gh api repos/{owner}/{repo}/pulls/{PR}/comments/{comment_id}/replies \
     --method POST -f body="@{author} <reply>"
   ```
   The reply must:
   - Start with `@{reviewer_login}`
   - Confirm the fix was applied
   - Explain what was wrong and why the fix is correct
   - Reference the relevant plan section (e.g. "plan requires *safe* log truncation")
   - State test result (e.g. "14 tests pass")

### For REJECT
Reply inline to the thread:
```bash
gh api repos/{owner}/{repo}/pulls/{PR}/comments/{comment_id}/replies \
  --method POST -f body="@{author} <reply>"
```
The reply must:
- Start with `@{reviewer_login}`
- Acknowledge the concern
- Cite the exact plan line or section that makes this a rejection
- Explain why the current implementation is intentional

### For review body comments (non-inline)
If the comment was in a review body (not attached to a specific line), reply as a PR comment:
```bash
gh pr comment {PR} --body "@{reviewer_login} ..."
```

## Step 4 — Push and Resolve Threads

1. Push all commits:
   ```bash
   git push
   ```
2. For every APPLY or APPLY-WITH-MODIFICATION thread, resolve it via GraphQL:
   ```bash
   # First get the thread node IDs
   gh api graphql -f query='{ repository(owner:"{owner}", name:"{repo}") { pullRequest(number:{PR}) { reviewThreads(first:20) { nodes { id isResolved } } } } }'

   # Resolve each applied thread
   gh api graphql -f query='mutation { resolveReviewThread(input: { threadId: "{thread_id}" }) { thread { id isResolved } } }'
   ```
3. Do NOT resolve REJECT threads — leave them open for the reviewer to see.

## Step 5 — Verify Merge Readiness

```bash
gh pr view {PR} --json mergeStateStatus,statusCheckRollup
```

Report:
- Which comments were applied, rejected, or modified
- CI check status
- Whether `mergeStateStatus` is `CLEAN` or still `BLOCKED` (and why)

## Hard Rules
- Always reply inline to inline comments — use `/pulls/{PR}/comments/{id}/replies`, never a top-level PR comment.
- Always tag the reviewer by login at the start of every reply.
- Never resolve a thread without first applying the fix or explaining the rejection.
- Never apply a change that contradicts the plan without flagging it to the user first.
- If a comment is ambiguous (could be APPLY or REJECT depending on interpretation), ask the user before acting.
