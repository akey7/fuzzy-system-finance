# Architecture and Installation

## Architecture

The files the backend delivers, and how the app must read them, are specified in the [Interface Contract](INTERFACE_CONTRACT.md). That file is a read-only mirror of the copy in the backend repo. The app requires contract v4 or later data; older files are reported on the dashboard instead of being displayed.

All data lives in one DigitalOcean Space (one S3 bucket) with two top-level folders, one for portfolio optimization and one for time series forecasts. The backend stores each run's files in a date folder inside each, at keys `<folder>/YYYY-MM-DD/<file>`, where the date is the last NYSE trading day in the data. A dropdown at the top of the app lists the dates whose data is complete in both folders, newest first, and selects the newest by default. The list is refreshed on every page load, and the page's two stacked sections, **Timeseries Analysis** (top) and **Portfolio Optimization** (below), show data from the selected date only. Objects directly under a top-level folder (left by older backend versions) are ignored. The read-only access key needs permission to **list** the Space as well as to read objects.

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

Hugging Face reads the Python and Gradio versions from the frontmatter of [`README.md`](../README.md), not from `pyproject.toml`, so these must be kept in agreement by hand:

- **Python:** `python_version` in the frontmatter and `.python-version` must name the same exact version (currently `3.13.5`). `requires-python` in `pyproject.toml` (`==3.13.*`) must include it; it covers the whole 3.13 series, so a patch upgrade does not require re-locking.
- **Gradio:** `sdk_version` in the frontmatter and the `gradio==` pin in `pyproject.toml`.

Development is on macOS while Hugging Face runs Debian Linux, so the environments match in Python and package versions but not in operating system or CPU architecture. One practical consequence: macOS file names are case-insensitive by default and Linux file names are case-sensitive, so keep file name casing exact in code.

### `.env` file and environment variables

On the production deployment on Hugging Face Spaces, these environment variables are held by the secrets manager. On a development environment, these are managed in a `.env` file **that should never be committed to GitHub or Hugging Face**.

1. `FSF_FRONT_END_BUCKET_ENDPOINT`: the **region-only** endpoint, `https://<region>.digitaloceanspaces.com`. Do **not** use the Space-specific "origin endpoint" (`https://<space>.<region>.digitaloceanspaces.com`); with it, listing the available dates fails with `NoSuchKey`.

2. `FSF_FRONT_END_BUCKET_REGION`, `FSF_FRONT_END_BUCKET_READ_ONLY`, `FSF_FRONT_END_BUCKET_READ_ONLY_KEY_ID`: region and read-only access key for the Space.

3. `FSF_FRONT_END_BUCKET_NAME`: name of the DigitalOcean Space (the S3 bucket) that holds all the data.

4. `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER`: the **top-level folder** inside the Space that holds the portfolio optimization files (e.g. `portfolio-optimization`).

5. `FSF_FRONT_END_TIME_SERIES_FOLDER`: the **top-level folder** inside the Space that holds the time series forecasts (e.g. `time-series`).

Dev and production share the one Space and differ only in the two folder names, so these two variables select the environment. Use the same pair of values as the backend for the same environment. (Before contract v13 the folder variables were named `PORTFOLIO_OPTIMIZATION_SPACE_NAME` and `TIME_SERIES_SPACE_NAME`; the old names are no longer read.)

## Purposes of folders and scripts

### Folders

1. `input/`: Holds input data downloaded from backend, one folder per date: `input/YYYY-MM-DD/<file>`, one folder per date viewed. If it does not exist, `app.py` creates it before access. Files directly in `input/` (from app versions before the date folders) are no longer used and can be deleted.

2. `tests/`: Unit tests for the data loaders (including the optional forecast columns and `all_forecasts_metadata.yml`), the available-date logic, bucket listing, and the app's text describing how forecasts are made. They build synthetic files in a temporary folder or use a fake S3 client, and never touch `input/` or the network. Run them from the repo root with `uv run pytest`.

### Scripts

| Filename           | Standalone Execution? | Purpose                                                                                  |
| ------------------ | --------------------- | ---------------------------------------------------------------------------------------- |
| `app.py`           | Yes                   | Run the Fuzzy System Finance Gradio app.                                                 |
| `s3_downloader.py` | No                    | Lists folders in the Space and downloads data from it for display on the app.            |
| `data_loading.py`  | No                    | Finds available dates and loads and validates the downloaded files against the [Interface Contract](INTERFACE_CONTRACT.md). |

## Hugging Face Spaces Docs

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
