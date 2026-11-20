## Summary

<!-- What does this pull request change, and why? -->

## Checklist

- [ ] The change is focused on one reviewable unit.
- [ ] Tests were written first and fail without the change.
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run lint-imports && uv run pytest -q -m "not e2e and not torch"` passes.
- [ ] Docs and help text were updated for new behaviour or options.
- [ ] No model weights, datasets or secrets are committed.
