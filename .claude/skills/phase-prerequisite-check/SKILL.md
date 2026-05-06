---
name: phase-prerequisite-check
description: Check whether a requested implementation phase from project-1-rag-qa-plan.md is ready to implement. Use before implementing any phase to inspect prerequisites, identify blockers, and report the exact implementation scope without editing files.
---

# Phase Prerequisite Check Skill

Use this skill before implementing any phase from `project-1-rag-qa-plan.md`.

## Goal
Determine whether the requested phase is ready to implement. This skill is read-only and must not edit files.

## Inputs
- Requested phase number or name.
- `CLAUDE.md`.
- `project-1-rag-qa-plan.md`.
- Current repo state.

## Procedure
1. Read `CLAUDE.md`.
2. Read `project-1-rag-qa-plan.md`.
3. Locate the requested phase exactly.
4. Summarize the requested phase goal and expected outputs.
5. Identify previous phases that are required for this phase.
6. Inspect repo state for expected files, directories, tests, docs, and commands from previous phases.
7. Identify verification commands that should exist for the requested phase.
8. Decide readiness:
   - `READY` if required prerequisites exist or the phase is the first phase that creates them.
   - `BLOCKED` if earlier required skeleton/files/configs are missing.
9. Report exact blockers and recommended next command.

## Output Format
- Status: `READY` or `BLOCKED`.
- Requested phase and goal.
- Required previous phases.
- Existing repo state.
- Missing prerequisites.
- Implementation scope if ready.
- Recommended next command.

## Hard Rules
- Do not create, edit, delete, or overwrite files.
- Do not install packages.
- Do not implement code.
- Do not run destructive commands.
