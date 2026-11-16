## Summary

What changed and why?

## Risk / security

- [ ] Does not weaken permission tiers, fail-closed review, sandbox checks, or
      frozen-file protection
- [ ] No secrets, `.env` content, traces, experiences, or private paths added
- [ ] MCP/tool changes reviewed for least privilege

## Verification

- [ ] `python -m compileall -q .`
- [ ] `python -m json.tool config.json`
- [ ] `python -m json.tool evals/tasks.json`
- [ ] `python -m unittest discover -s tests -v`
- [ ] `ruff check .`
- [ ] Relevant eval subset (state which)

## Notes / limitations
