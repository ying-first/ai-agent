# Security Policy

## Supported versions

This is experimental alpha software. Security fixes are applied to the current
`main` branch only; no long-term support branches are promised.

## Reporting a vulnerability

Please do **not** open a public issue for a security vulnerability.

Use the hosting platform's private security advisory feature if available, or
contact the maintainers privately through the repository's listed security
contact. Include:

- affected version/commit
- steps to reproduce
- impact
- whether secrets, host files, or network access are exposed
- a minimal proof of concept if safe

Do not test against systems or data you do not own or have permission to test.

## Threat model

AI_Agent is an agent runtime that can:

- read and write workspace files
- execute Python and system commands under permission gates
- call external tools, MCP servers, and web endpoints
- modify its own runtime through a gated evolution pipeline

It is **not** a security boundary for hostile users. Run it only in an
environment you control.

### Defensive layers that exist

- `safety.py` local AST/dangerous-call checks before write/execute
- separate writer and reviewer clients; reviewer failures are fail-closed
- permission tiers (`read/write/delete/execute/network`)
- Windows AppContainer/Job Object execution sandbox for Python
- candidate isolation and frozen files in self-evolution
- MCP read-only allowlist and sensitive-path guard for the bundled filesystem
  example
- experience memory and traces are local and gitignored

### Known limitations

- The Windows sandbox is the strongest path. POSIX and non-AppContainer fallback
  modes are weaker.
- The reviewer is an LLM, not a formal verifier. By default it may use the same
  model/provider as the writer; configure `REVIEWER_*` for independent review.
- `REVIEW_MODE=sampled` does not LLM-review every code write. Use `strict` when
  the code is untrusted.
- MCP servers run as external processes. The bundled filesystem server is
  disabled by default and exposes only read-only tools when enabled, but MCP is
  still third-party code.
- Experience memory can contain sensitive task content. Do not paste secrets or
  regulated data into prompts.
- Self-evolution cannot modify frozen files and still requires human promotion,
  but a scored candidate should be reviewed before approval.

## Safe deployment checklist

- Use a scoped API key, never a production key.
- Keep `.env` out of version control and out of prompts.
- Use `REVIEW_MODE=strict` for shared or untrusted repositories.
- Keep MCP disabled unless you need it; review every server's tools and tier.
- Do not run on a machine with unrelated sensitive files if the agent may read
  the workspace.
- Keep the git baseline and checkpoints; review diffs before promotion.
- Treat tool output, web content, MCP content, and experience cards as untrusted
  data, never as instructions.
