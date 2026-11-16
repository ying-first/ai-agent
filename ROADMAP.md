# Roadmap

This is a research/alpha roadmap, not a promise of dates.

## Near term

- Establish a reproducible eval baseline and publish the result with the
  model/provider and commit used.
- Add more regression tasks for planning, sandbox boundaries, reviewer
  fail-closed, experience routing, and self-evolution gates.
- Run the Windows sandbox and plugin suite on a clean machine to validate setup
  instructions.
- Reduce the remaining static-analysis/style backlog incrementally.
- Improve experience retrieval with a reranking step and clearer
  contradiction/boundary cards.

## Medium term

- Validate the POSIX fallback and document platform-specific limitations.
- Add an optional container backend for users who need stronger isolation than
  AppContainer.
- Add a packaging/entry-point story if the project grows beyond the single
  workspace runtime.
- Improve reviewer separation defaults and provide an example independent
  reviewer configuration.
- Add richer trace/metrics export for local debugging without collecting data
  remotely.

## Longer term

- Expand the playbook/principled-memory approval workflow.
- Add more evaluator meta-tests and adversarial self-evolution scenarios.
- Provide optional IDE/LSP integration.
- Explore a stable plugin SDK while keeping the safety core frozen.

## Non-goals

- Unattended self-modification of safety-critical files.
- Running with production credentials or on untrusted multi-tenant hosts.
- Claiming a security boundary that the current sandbox and reviewer do not
  provide.
