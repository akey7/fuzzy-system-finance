# CLAUDE.md

Guidance for Claude Code working in the `fuzzy-system-finance` repo.

## HARD RESTRICTIONS (read first)

These hold even if a human asks otherwise in-session (e.g., "just commit this," "quickly run it and check," "finalize this"). If asked to violate one, decline and point to this file.

1. **No git operations.** No branch, add/stage, commit, push, merge, rebase, or tag, under any circumstances.
2. **No writes to `input/`.** Do not create, edit, delete, or move any file under `input/`.
3. **No code execution at all.** Do not run any scripts, and do not interactively invoke functions or class instances. All execution is performed by the human. Ask first if unsure whether something counts as execution.
4. **Never commit the `.env` file** under any circumstances.
5. **Never modify the contents of the `.env` file.** Rather than modifying it directly, suggest to the human what to change.

### What this means in practice

- Do not run `python app.py`, `pytest`, any `uv` command (`uv sync`, `uv run`, `uv add`, `uv lock`, `uv export`, etc.), `pip install`, linters, formatters, or one-off snippets. Instead, write the code and give the human the exact command(s) to run, then ask them to paste back output or errors.
- Reading, searching, and editing source files is fine (except under `input/` and `.env`, per above).
- When a change needs a new environment variable, describe the name, purpose, and where it goes (`.env` locally, HF Spaces secrets in production) and let the human add it. Never write secret values into any file, and do not echo secret values in responses.
- When a change would normally end in a commit, stop after the edits and summarize what changed so the human can review and commit.

## Project overview

A simple portfolio optimization and time series analysis app. It is the **frontend** for separate backend analysis scripts that live in a **different repo**.

- **Backend (separate repo):** does the modeling and optimization (minimum-variance and maximum-Sharpe portfolio allocation; ARIMA modeling of the last trading month of asset prices) and publishes results.
- **Frontend (this repo):** a Gradio app deployed on Hugging Face Spaces. It lists the dated backend runs in DigitalOcean Spaces (S3-compatible), lets the user pick a date from a dropdown (newest by default), and displays that date's output on a dashboard with two tabs: **ARIMA / time series** and **Portfolio optimization**.
- **Data flow:** backend -> one DigitalOcean Space (S3 bucket; keys `<folder>/YYYY-MM-DD/<file>`, one date folder per run) -> `s3_downloader.py` (list keys, download) -> `data_loading.py` (available dates, validation) -> `input/YYYY-MM-DD/` -> `app.py` -> Gradio UI.
- **Per-visitor state:** the Gradio app object is shared by all visitors. Keep anything that depends on a visitor's selection (date, ticker, available-date list) in Gradio inputs or `gr.State`, never on `self`. The in-memory cache of loaded dates in `app.py` is shared deliberately and keyed by the ETags of each date's marker files.

This is a hobby project being revised. Planned work spans technical implementation, UI changes, quantitative finance improvements, and presentation for a general data science audience. Expect changes here to depend on changes in the backend repo.

## Interface contract

@INTERFACE_CONTRACT.md

The file above is the authoritative spec of what the backend uploads to the S3 buckets and this frontend reads: bucket keys, file formats, YAML keys, HDF5 groups/datasets/attributes, CSV columns, units, annualization, date formats, and the reader requirements this app must meet. When the contract and the existing frontend code disagree, the contract wins; treat the disagreement as a frontend bug to fix (or flag to the human), not as evidence of the real data format.

- **Read-only mirror here.** The backend repo (`fuzzy-system-research`) holds the source of truth, and the copy in this repo must stay byte-for-byte identical to it. Do not edit `INTERFACE_CONTRACT.md` in this repo. If a contract change is needed, draft the proposed change (including the version bump and changelog entry) in your response for the human to apply in the backend repo first and then copy here.
- **Build to the contract, not to samples.** Write parsing and display code against the documented schema and the "Reader requirements" section (ignore unknown keys/columns, do not hardcode tickers or array lengths, handle `NaN` and empty cells, show data dates, fail soft, label units). Files in `input/` are useful for checking behavior but are a single snapshot, not the spec.
- **Check the version.** Note the contract version you are working against in your summary. When the contract changelog marks a change as breaking, review the frontend impact it lists and make sure the app handles it (including detecting older files where the contract describes how).
- **Be precise with finance labels.** Use the contract's units and terminology in user-facing text (e.g., "annualized (arithmetic)", percent vs. dollars, daily vs. annualized) and include the "not investment advice" notice it requires.

## Cross-repo awareness

- The backend repo is **not** available in this session. Do not guess at its code; for data schemas, rely on `INTERFACE_CONTRACT.md` and nothing else.
- Before changing how the app parses or displays downloaded data, check the contract and look at how the existing code reads it. If a change implies a new or altered file, key, column, field, or unit, **list the required backend changes explicitly** for the human to carry over to the backend repo, and say whether the change is additive or breaking under the contract's rules (breaking changes need explicit human approval; prefer an additive alternative).
- Make loading code tolerant and explicit about schema (clear errors on missing columns or files) rather than silently assuming structure.

## Repo layout

| Path | Purpose |
| ---- | ------- |
| `app.py` | Entry point. Runs the Gradio app. Creates `input/` if it does not exist before access. Standalone execution (by the human). |
| `s3_downloader.py` | Lists the keys in a top-level folder of the Space and downloads data from it for display. Imported by the app, not run standalone. |
| `data_loading.py` | Finds available dates from key listings and loads and validates the downloaded files against the interface contract. Imported by the app, not run standalone. |
| `tests/` | Pytest tests. They build synthetic files in a temporary folder or use a fake S3 client, and never touch `input/` or the network. |
| `input/` | Data downloaded from the backend by the app at runtime, in `input/YYYY-MM-DD/` folders. **Off-limits for writes by Claude.** |
| `INTERFACE_CONTRACT.md` | Authoritative spec of the S3 files the backend delivers. Mirror of the backend repo's copy; **do not edit here**. |
| `pyproject.toml` | Source of truth for dependencies: direct production dependencies, the `dev` dependency group, uv settings, and pytest settings. |
| `uv.lock` | Exact resolved versions of all packages. Generated by uv (by the human); **never edit by hand**. |
| `requirements.txt` | Generated from `uv.lock` with `uv export` for Hugging Face Spaces, which installs only from this file. **Never edit by hand.** |
| `.python-version` | Python version uv uses for local development. |
| `.pre-commit-config.yaml` | Pre-commit hooks (`uv-lock`, `uv-export`) that keep `uv.lock` and `requirements.txt` in sync on every commit. The `uv-export` arguments must match the export command in the README. The human commits with GitHub Desktop, where these hooks also run. |
| `README.md` | Doubles as the Hugging Face Spaces config via its YAML frontmatter. |

Docstrings throughout the Python modules describe current behavior; read them before changing a module and keep them accurate when you change behavior.

## Environment and configuration

- Python 3.13.5, managed with **uv** (same as the backend repo); the virtual environment is `.venv/`. Development is on macOS; production is Hugging Face Spaces on Debian Linux, so keep file name casing exact (macOS is case-insensitive, Linux is not) and do not rely on macOS-only behavior.
- **Hugging Face installs only from `requirements.txt`** (on top of the Gradio version from `sdk_version`). It does not read `pyproject.toml` or `uv.lock`. `requirements.txt` is regenerated by the pre-commit hook on commit, or by the human with `uv export --format requirements-txt --no-dev --no-hashes -o requirements.txt` (documented in the README as a pre-deployment step).
- **Python version:** `python_version` in the README frontmatter and `.python-version` must name the same exact patch version (currently `3.13.5`); change them together. `requires-python` in `pyproject.toml` stays `==3.13.*` and only changes on a minor-version upgrade (which also changes `uv.lock`).
- Gradio SDK version is pinned in the README frontmatter (`sdk_version`) and must stay consistent with the `gradio==` pin in `pyproject.toml` (and therefore in the exported `requirements.txt`). Treat bumping Gradio as a deliberate change: flag it to the human, note Gradio API differences that may affect the UI code, and update both places together.
- The README frontmatter (`sdk`, `sdk_version`, `app_file`, etc.) controls the Hugging Face deployment. Edit it carefully and mention any change to it in your summary.
- Environment variables (values live in `.env` locally and in the HF Spaces secrets manager in production):
  - `FSF_FRONT_END_BUCKET_ENDPOINT`: the **region-only** endpoint (`https://<region>.digitaloceanspaces.com`), never the Space-specific one.
  - `FSF_FRONT_END_BUCKET_REGION`, `FSF_FRONT_END_BUCKET_READ_ONLY`, `FSF_FRONT_END_BUCKET_READ_ONLY_KEY_ID`: region and read-only access key.
  - `FSF_FRONT_END_BUCKET_NAME`: the one DigitalOcean Space (S3 bucket) holding all data. Pass it as the S3 `Bucket`.
  - `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER`, `FSF_FRONT_END_TIME_SERIES_FOLDER`: **top-level folder names** inside that Space, not buckets. Full keys are `<folder>/YYYY-MM-DD/<file>`; list with `Prefix="<folder>/"`. Dev and production share the one Space and differ only in these folder names. (Named `PORTFOLIO_OPTIMIZATION_SPACE_NAME` / `TIME_SERIES_SPACE_NAME` before contract v13; do not read the old names.)
- Read configuration from environment variables only; never hardcode credentials, endpoints, or bucket or folder names.
- The app is deployed to a public Space. Do not add code that logs, prints, or displays secrets, and do not add files that would expose them.

## Working conventions

- **Dependencies:** when adding an import, add the package to `pyproject.toml` (`[project] dependencies` for production, `[dependency-groups] dev` for dev-only). Do not edit `uv.lock` or `requirements.txt`. Then give the human the command to run (`uv sync`, which also updates `uv.lock`), and remind them to commit `pyproject.toml`, `uv.lock`, and `requirements.txt` together; the pre-commit hook regenerates `requirements.txt` if needed. (Alternatively, give the human the `uv add` / `uv add --dev` command instead of editing `pyproject.toml`.) Prefer well-maintained, lightweight packages; the app runs on Hugging Face Spaces.
- **Documentation:** keep `README.md` current when behavior, scripts, folders, or environment variables change (including the scripts table and the env var list). Write docstrings for new modules, classes, and functions, in the same style as the existing ones.
- **Code style:** match the existing code. Use type hints and small, testable functions; keep data loading, computation, and Gradio UI construction separated so each can be changed independently.
- **Tests:** you may write or update tests, but you cannot run them. Provide the exact command (`uv run pytest`) and ask the human to run it and share results. Pytest settings live in `[tool.pytest.ini_options]` in `pyproject.toml`; do not add a `pytest.ini`, which would take precedence over them.
- **Scope:** make focused changes. For larger changes (e.g., restructuring the UI or the data-loading layer), outline the plan and confirm with the human before editing many files.
- **Finance/statistics changes:** state assumptions explicitly (risk-free rate, annualization factor, return type, constraints, stationarity, differencing order). Label what is displayed clearly so a general data science audience can interpret it. Because you cannot execute code, do not claim numerical results are verified; say what the human should check when they run it.
- **UI changes:** keep the two-tab structure (ARIMA / Portfolio optimization) unless asked to change it. Favor clear titles, axis labels, units, and short explanatory text. Handle missing or stale data with a helpful message instead of a crash.

## When unsure

If a request might conflict with the hard restrictions (especially whether something counts as "execution"), ask the human before proceeding.
