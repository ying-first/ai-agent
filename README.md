# AI_Agent

A local, experimental, self-evolving coding agent for Windows.

> Status: **alpha / research software**. It can read and write files, execute code,
> call external tools, and modify its own runtime under explicit gates. Do not run
> it with production credentials or on machines containing data you cannot afford
> to lose.

AI_Agent is a small agent runtime built around a safety core, a tool/plugin
system, a repository index, an execution sandbox, a layered experience memory,
and an approval-gated self-evolution pipeline. It is intentionally more
transparent and hackable than a finished commercial product.

## What it can do

- **Code work**: read, patch, overwrite, smoke-test, run tests, index symbols,
  inspect callers/references/impact, and collect ruff diagnostics.
- **Planning and parallelism**: generate a DAG plan for complex tasks, execute
  independent read steps in parallel, and run read-only subagents with isolated
  histories.
- **Sandboxed execution**: on Windows, `run_python_file` uses an AppContainer
  mirror with network disabled by default; `execute_system_command` uses Job
  Object resource limits. POSIX has a lighter rlimit fallback.
- **Permission tiers**: `read / write / delete / execute / network`, with
  session grants and human approval for higher-risk actions.
- **Layered experience memory**: SQLite + FTS5 + optional fastembed vectors;
  fingerprint deduplication, clustering, patterns, playbooks, principles,
  compact recall cards, feedback/utility, correction and suspect handling.
- **Evaluation**: a local eval harness with pass/fail checks, regression
  comparison, JSON reports, and a baseline workflow.
- **Approval-gated self-evolution**: candidate isolation, static gates, security
  probes, scorecards, canary runs, automatic rollback, and frozen files.
- **External tools**: MCP stdio client, git/grep/test/web helpers, and optional
  external app launching.

## Requirements

- Windows 11 is the primary and tested platform.
- Python **3.12**.
- Node.js/npm only if you want to use an MCP server.
- Git for version control.
- An OpenAI-compatible API endpoint. The default model name is configurable.

Linux/macOS may work in a degraded mode (no AppContainer); it is not currently
validated as a supported platform.

## Quick start

```powershell
git clone <your-repository-url>
cd AI_Agent

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Optional capabilities: tokenizer, embeddings, tree-sitter, web search, lint
python -m pip install -r requirements-optional.txt

Copy-Item .env.example .env
# Edit .env and set your API key / model
python agent.py
```

The GUI entry point is:

```powershell
python gui_frontend.py
```

If you run from another directory, use the Python interpreter from the project
environment and keep the repository root as the working directory. The safety
core resolves the workspace from `safety.py` itself.

## Configuration

- `config.json` contains runtime policy and tool settings. It is intentionally
  commented and human-editable.
- `.env` contains secrets and is gitignored.
- `.env.example` lists the supported environment variables.

Important environment variables:

```env
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=your-main-model
REVIEWER_MODEL=your-reviewer-model
REVIEWER_API_KEY=
REVIEWER_BASE_URL=
REVIEW_MODE=strict
```

### Reviewer separation

The agent has a writer client and a reviewer client. They are separate clients,
but by default the reviewer falls back to the writer model. For genuinely
independent review, set `REVIEWER_MODEL`, `REVIEWER_BASE_URL`, and
`REVIEWER_API_KEY`.

When review is required, reviewer failures are **fail-closed**: writes are
rejected while the reviewer is unavailable. Local AST/dangerous-call checks run
before the LLM review and remain active.

`config.review.mode`:

- `off`: only local static checks
- `sampled`: default; samples some plugin/tool additions
- `strict`: every code write is reviewed

Use `strict` for untrusted or shared environments.

### MCP

MCP is disabled by default. A read-only filesystem server is preconfigured in
`config.json` but is not started while `mcp.enabled` is `false`.

To enable it, set `mcp.enabled` to `true` and restart the agent. The server is
restricted to a read-only tool allowlist, and `.env`, `.git`, `.experiences`,
and other sensitive paths are blocked by the MCP path guard.

## Experience memory

The running memory is stored under `.experiences/` (SQLite + legacy JSON). The
directory is gitignored. On first start, legacy JSON experiences are migrated
automatically.

Useful actions:

```text
manage_experience(action='stats')
manage_experience(action='recall', query='...')
manage_experience(action='feedback', exp_id='...', used=true, outcome='success')
manage_experience(action='correct', exp_id='...', lesson='...', reason='...')
manage_experience(action='gaps')
manage_experience(action='consolidate')
manage_experience(action='metrics')
```

High-confidence approved playbooks or experiences can be routed into a
"direct" mode; failures downgrade their utility and can mark them `suspect`.
The system never silently rewrites a wrong experience: use `correct` to create
a superseding revision.

## Evaluation

The eval suite lives in `evals/tasks.json`. Reports and baselines are written
under `evals/` and are gitignored.

```powershell
# Run the suite (API calls; can take several minutes)
python run_evals.py

# Establish a baseline for evolution scoring
python run_evals.py --update-baseline

# Run a subset
python run_evals.py --tag sandbox
python run_evals.py --tag evolution
```

A full public benchmark result is not shipped yet. Treat the local suite as a
regression aid, not proof of general coding ability.

## Self-evolution

The evolution pipeline is human-gated and should only be used after a baseline
exists:

```text
evolve(action='baseline')
evolve(action='propose', goal='...')
# edit only .evolution/candidates/<id>/
evolve(action='evaluate', proposal_id='...')
evolve(action='promote', proposal_id='...')
```

Frozen files include the safety core, sandbox, permissions, evaluator,
experience store, evolution code, task set, and config. Candidate changes to
those files are rejected. A scored candidate still requires human approval
before promotion.

## Project layout

```text
agent.py                 main agent loop and tools
safety.py                safety core, config, write/review gates
sandbox.py               AppContainer / Job Object / POSIX sandbox
permissions.py           permission tiers and session grants
experience_store.py      SQLite/FTS5 layered experience memory
checkpoints.py           history + file checkpoints
evolution.py             self-evolution pipeline
eval_harness.py          eval runner and checks
run_evals.py             CLI eval entry point
plugins/                 runtime-loaded tools
evals/tasks.json         local eval tasks
scripts/                 standalone probes/checks
tests/                   CI-safe unit tests
config.json              policy and tool configuration
.env.example             environment template
```

## Security and limitations

- This is not a formally audited security product.
- The Windows sandbox is the strongest isolation path. Other platforms use a
  weaker fallback.
- The reviewer is not a proof of correctness; it is one layer of defense.
- MCP servers are external processes. Keep the default allowlist and path guard
  unless you understand the risk.
- Experience memory and traces can contain user task content. Do not put secrets
  in prompts.
- Self-evolution is deliberately not fully autonomous.
- The default model/reviewer configuration may point at the same model. Set a
  separate reviewer for real separation.

See [SECURITY.md](SECURITY.md) for reporting and threat-model notes.

## License

MIT. See [LICENSE](LICENSE).
