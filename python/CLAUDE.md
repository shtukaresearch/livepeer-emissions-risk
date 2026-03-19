# Project conventions

## Marimo notebooks

- Do NOT call `.to_pandas()` on Polars DataFrames before passing to Altair — Altair accepts Polars directly, and pandas is not in the dependency set.
- Every variable must be defined in exactly one cell. Loop variables and temporaries that appear in multiple cells must be prefixed with `_` to make them cell-private.
- Use `@app.function` to extract shared logic that multiple cells need.
- Do not hardcode data file paths — use `os.environ.get("LPT_DATA_SOURCE")` or similar env vars.
