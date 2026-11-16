# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-11-16

Initial public baseline.

### Added

- Plugin-based agent runtime with `@tool` hot loading.
- Safety core with AST checks, permission tiers, reviewer separation, and
  fail-closed reviewer degradation.
- Windows AppContainer / Job Object execution sandbox with mirror workspace.
- Token-budgeted context management, checkpoints, and rollback.
- Planner/DAG parallel execution, read-only subagents, and task-tree support.
- Repository index with Python AST and tree-sitter languages, symbol/reference/
  impact/diagnostics tools.
- MCP stdio client with dynamic tool registration, tool allowlists, and
  sensitive-path guards.
- Evaluation harness, local task set, baseline/regression reporting, and
  evolution scorer meta-evaluation.
- Approval-gated self-evolution: proposals, candidate isolation, frozen files,
  security probes, scorecards, canary runs, and rollback.
- Layered experience memory: SQLite/FTS5, fingerprint deduplication, optional
  fastembed vectors, clustering, patterns, playbooks, principles, compact
  recall cards, feedback/utility, correction, suspect handling, capability
  gaps, and deterministic A/B experiments.
- Open-source project scaffolding: README, MIT license, security policy,
  contributing guide, dependency manifests, environment template, CI, and
  smoke tests.

### Known limitations

- Windows-first; other platforms are not validated.
- No public benchmark result yet; the local eval baseline is not shipped.
- Reviewer defaults to the writer model unless `REVIEWER_*` is configured.
- Experience direct-routing requires accumulated evidence and approved
  playbooks; a fresh install starts in advisory mode.
- MCP is disabled by default; the bundled filesystem server is read-only and
  still third-party code.
