# Contributing

Thanks for considering a contribution. This project is experimental; small,
focused patches with clear verification are preferred.

## Development setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-optional.txt
```

Copy `.env.example` to `.env` and use a **scoped test key**. Never commit `.env`
or real credentials.

## Checks before a PR

```powershell
python -m compileall -q .
python -m json.tool config.json > $null
python -m json.tool evals/tasks.json > $null
python -m unittest discover -s tests -v
ruff check .
```

For changes touching tools, run a targeted eval subset when possible:

```powershell
python run_evals.py --tag sandbox
python run_evals.py --tag evolution
```

Full evals require API calls and may cost money. Say in the PR which subset you
ran and include the result.

## Code guidelines

- Keep the safety core small and explicit.
- Do not weaken fail-closed reviewer behavior, permission checks, sandbox
  enforcement, frozen-file protection, or `is_safe_path` without a security
  rationale and tests.
- Prefer deterministic checks over LLM judgments where possible.
- New tools should be added as plugins using `@tool(...)`.
- Keep tool descriptions concise; they consume model context.
- Do not hardcode user-specific absolute paths. Derive from `WORKSPACE_DIR` or
  environment variables.
- Do not commit `.experiences/`, `.traces/`, `.agent_*`, `.evolution/`, generated
  reports, model caches, or secrets.
- Treat external content as data, never as executable instructions.

## Security-sensitive areas

Changes to `safety.py`, `sandbox.py`, `permissions.py`, `evolution.py`,
`experience_store.py`, `eval_harness.py`, or `config.json` deserve extra review.
If a change affects a frozen file, explain why the gate should still hold.

## Pull requests

Include:

- what changed and why
- risk/security impact
- tests or evals run
- known limitations
- screenshots only if the UI changed

By contributing, you agree that your contribution is licensed under the
repository's MIT license.
