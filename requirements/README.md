# Dependency Profiles

This project now uses dependency profiles instead of a single oversized list.

## Files

- `requirements/core.txt`: minimal runtime for core processing pipeline.
- `requirements/ml.txt`: adds sample-bank parquet support (`pyarrow`) for the upcoming ML workflow.
- `requirements/notebook.txt`: adds Jupyter + visualization stack.
- `requirements/dev.txt`: full setup for development and tests.

## Install examples

Core runtime:

```bash
pip install -r requirements/core.txt
```

Sample-bank / ML workflow:

```bash
pip install -r requirements/ml.txt
```

Notebook workflow:

```bash
pip install -r requirements/notebook.txt
```

Full development setup:

```bash
pip install -r requirements/dev.txt
```

## Backward compatibility

`requirements.txt` now points to `requirements/dev.txt` to preserve
existing `pip install -r requirements.txt` behavior for contributors.
