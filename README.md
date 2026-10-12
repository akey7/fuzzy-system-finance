---
title: Fuzzy System Finance
emoji: 🏢
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.22.0
python_version: "3.13.5"
app_file: app.py
pinned: false
license: mit
short_description: Portfolio optimization and time series analysis app.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# fuzzy-system-finance
A simple portfolio optimization and time series analysis app. It is the front end for a backend analysis scripts. Data is transferred from the backend to the front end via DigitalOcean Spaces, an S3 hosting service.

The files the backend delivers, and how the app must read them, are specified in [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md). That file is a read-only mirror of the copy in the backend repo. The app requires contract v4 or later data; older files are reported on the dashboard instead of being displayed.

**Not investment advice.** This is a hobby project, and every value it shows is a backtest or model output.

## Installation

### Development environment (uv)

Local development uses [uv](https://docs.astral.sh/uv/) to manage both the Python version and the dependencies. The relevant files are:

| File | Role |
| ---- | ---- |
| `.python-version` | Exact Python version used by uv for local development (`3.13.5`). |
| `pyproject.toml` | Direct production dependencies, the `dev` dependency group, and pytest settings. **Edit dependencies here.** |
| `uv.lock` | Exact resolved versions of every package. Commit it. |
| `requirements.txt` | **Generated** from `uv.lock` for Hugging Face Spaces, which installs only from this file. Do not edit it by hand. |
| `.pre-commit-config.yaml` | Git pre-commit hook that keeps `uv.lock` and `requirements.txt` in sync with `pyproject.toml` on every commit. |

Install uv (once), then create the virtual environment in `.venv/` with the pinned Python version and all production and development dependencies:

```
brew install uv
uv sync
```

Then install the pre-commit hook (once per clone):

```
uv run pre-commit install
```

Run the app and the tests through uv:

```
uv run python app.py
uv run pytest
```

### Changing dependencies

Add or remove packages with uv, which updates `pyproject.toml` and `uv.lock` together:

```
uv add <package>
uv add --dev <package>
uv remove <package>
```

Commit `pyproject.toml`, `uv.lock`, and `requirements.txt` together. The pre-commit hook regenerates `uv.lock` and `requirements.txt` whenever they are out of date. The hook runs in GitHub Desktop as well as on the command line.

**If a commit fails because the hook changed files:** this is expected. The hook updated `uv.lock` and/or `requirements.txt` and stopped the commit so you can include them. In GitHub Desktop, the updated files appear in the Changes list; make sure they are checked and commit again.

### Before deploying to Hugging Face Spaces

Hugging Face installs dependencies only from `requirements.txt`, so it must match `uv.lock` before you push to the Space. The pre-commit hook normally handles this. If the hook was not installed, was skipped, or you are unsure, regenerate the file manually from the repo root and commit any changes:

```
uv export --format requirements-txt --no-dev --no-hashes -o requirements.txt
```

`--no-dev` keeps development tools out of the deployment, and `--no-hashes` avoids pip hash-checking conflicts with the packages Hugging Face pre-installs. After deploying, check the Space's build log to confirm it used the expected Python version and installed from `requirements.txt`.

### Keeping versions in sync with Hugging Face Spaces

Hugging Face reads the Python and Gradio versions from this README's frontmatter, not from `pyproject.toml`, so these must be kept in agreement by hand:

- **Python:** `python_version` in the frontmatter and `.python-version` must name the same exact version (currently `3.13.5`). `requires-python` in `pyproject.toml` (`==3.13.*`) must include it; it covers the whole 3.13 series, so a patch upgrade does not require re-locking.
- **Gradio:** `sdk_version` in the frontmatter and the `gradio==` pin in `pyproject.toml`.

Development is on macOS while Hugging Face runs Debian Linux, so the environments match in Python and package versions but not in operating system or CPU architecture. One practical consequence: macOS file names are case-insensitive by default and Linux file names are case-sensitive, so keep file name casing exact in code.

### `.env` file and environment variables

On the production deployment on HuggingFace Spaces, these environment variables are held by the secrets manager. On a development environment, these are managed in a `.env` file **that should never be committed to GitHub or HuggingFace**.

1. `FSF_FRONT_END_BUCKET_ENDPOINT`, `FSF_FRONT_END_BUCKET_REGION`, `FSF_FRONT_END_BUCKET_READ_ONLY`, `FSF_FRONT_END_BUCKET_READ_ONLY_KEY_ID`: Specifications of the S3 bucket hosted on DigitalOcean. This bucket specification points the app to the files from the backend.

2. `PORTFOLIO_OPTIMIZATION_SPACE_NAME`: DigitalOcean space name for portfolio optimization files.

3. `TIME_SERIES_SPACE_NAME`: DigitalOcean space name for time series analysis data.

## Purposes of folders and scripts

### Folders

1. `input/`: Holds input data downloaded from backend. If it does not exist, `app.py` creates it before access.

2. `tests/`: Unit tests for the data loaders. They build synthetic files in a temporary folder and never touch `input/`. Run them from the repo root with `uv run pytest`.

### Scripts

| Filename           | Standalone Execution? | Purpose                                                                                  |
| ------------------ | --------------------- | ---------------------------------------------------------------------------------------- |
| `app.py`           | Yes                   | Run the Fuzzy System Finance Gradio app.                                                 |
| `s3_downloader.py` | No                    | Downloads data from S3 buckets for display on the app.                                   |
| `data_loading.py`  | No                    | Loads and validates the downloaded files against `INTERFACE_CONTRACT.md`.                |
