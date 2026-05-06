# /review-pr-comments

Fetch all review comments on a PR, check each against the project plan, apply valid ones with inline replies, and reject out-of-scope ones with a reasoned explanation.

## Arguments
`$ARGUMENTS` — PR number, e.g. `2`. If omitted, use the PR for the current branch.

## Procedure
Use the skill `.claude/skills/review-pr-comments/SKILL.md` with the PR number as the argument.
