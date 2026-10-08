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

- Do not run `python app.py`, `pytest`, `pip install`, linters, formatters, or one-off snippets. Instead, write the code and give the human the exact command(s) to run, then ask them to paste back output or errors.
- Reading, searching, and editing source files is fine (except under `input/` and `.env`, per above).
- When a change needs a new environment variable, describe the name, purpose, and where it goes (`.env` locally, HF Spaces secrets in production) and let the human add it. Never write secret values into any file, and do not echo secret values in responses.
- When a change would normally end in a commit, stop after the edits and summarize what changed so the human can review and commit.

## Project overview

A simple portfolio optimization and time series analysis app. It is the **frontend** for separate backend analysis scripts that live in a **different repo**.

- **Backend (separate repo):** does the modeling and optimization (minimum-variance and maximum-Sharpe portfolio allocation; ARIMA modeling of the last trading month of asset prices) and publishes results.
- **Frontend (this repo):** a Gradio app deployed on Hugging Face Spaces. It downloads the latest backend output from DigitalOcean Spaces (S3-compatible) and displays it on a dashboard with two tabs: **ARIMA / time series** and **Portfolio optimization**.
- **Data flow:** backend -> DigitalOcean Spaces (S3) -> `s3_downloader.py` -> `input/` -> `app.py` -> Gradio UI.

This is a hobby project being revised. Planned work spans technical implementation, UI changes, quantitative finance improvements, and presentation for a general data science audience. Expect changes here to depend on changes in the backend repo.

## Cross-repo awareness

- The backend repo is **not** available in this session. Do not guess at its code or data schemas.
- The files in the S3 buckets are the contract between the repos. Before changing how the app parses or displays downloaded data, look at how the existing code reads it, and if a change implies a new or altered file format, column, or field, **list the required backend changes explicitly** for the human to carry over to the backend repo.
- Make loading code tolerant and explicit about schema (clear errors on missing columns or files) rather than silently assuming structure.

## Repo layout

| Path | Purpose |
| ---- | ------- |
| `app.py` | Entry point. Runs the Gradio app. Creates `input/` if it does not exist before access. Standalone execution (by the human). |
| `s3_downloader.py` | Downloads data from the S3 buckets for display. Imported by the app, not run standalone. |
| `input/` | Data downloaded from the backend. **Off-limits for writes.** |
| `requirements.txt` | Production dependencies. |
| `requirements-dev.txt` | Development + production dependencies. |
| `README.md` | Doubles as the Hugging Face Spaces config via its YAML frontmatter. |

Docstrings throughout the Python modules describe current behavior; read them before changing a module and keep them accurate when you change behavior.

## Environment and configuration

- Python 3.12, conda environment named `fuzzy-system-finance`. Development is on macOS.
- Gradio SDK version is pinned in the README frontmatter (`sdk_version`) and must stay consistent with the Gradio version in the requirements files. Treat bumping Gradio as a deliberate change: flag it to the human, note Gradio API differences that may affect the UI code, and update both places together.
- The README frontmatter (`sdk`, `sdk_version`, `app_file`, etc.) controls the Hugging Face deployment. Edit it carefully and mention any change to it in your summary.
- Environment variables (values live in `.env` locally and in the HF Spaces secrets manager in production):
  - `FSF_FRONT_END_BUCKET_ENDPOINT`, `FSF_FRONT_END_BUCKET_REGION`, `FSF_FRONT_END_BUCKET_READ_ONLY`, `FSF_FRONT_END_BUCKET_READ_ONLY_KEY_ID`: S3 bucket specification on DigitalOcean.
  - `PORTFOLIO_OPTIMIZATION_SPACE_NAME`: space for portfolio optimization files.
  - `TIME_SERIES_SPACE_NAME`: space for time series analysis data.
- Read configuration from environment variables only; never hardcode credentials, endpoints, or bucket names.
- The app is deployed to a public Space. Do not add code that logs, prints, or displays secrets, and do not add files that would expose them.

## Working conventions

- **Dependencies:** when adding an import, add the package to `requirements.txt` (production) or `requirements-dev.txt` (dev-only) and tell the human so they can install it. Prefer well-maintained, lightweight packages; the app runs on Hugging Face Spaces.
- **Documentation:** keep `README.md` current when behavior, scripts, folders, or environment variables change (including the scripts table and the env var list). Write docstrings for new modules, classes, and functions, in the same style as the existing ones.
- **Code style:** match the existing code. Use type hints and small, testable functions; keep data loading, computation, and Gradio UI construction separated so each can be changed independently.
- **Tests:** you may write or update tests, but you cannot run them. Provide the exact command and ask the human to run it and share results.
- **Scope:** make focused changes. For larger changes (e.g., restructuring the UI or the data-loading layer), outline the plan and confirm with the human before editing many files.
- **Finance/statistics changes:** state assumptions explicitly (risk-free rate, annualization factor, return type, constraints, stationarity, differencing order). Label what is displayed clearly so a general data science audience can interpret it. Because you cannot execute code, do not claim numerical results are verified; say what the human should check when they run it.
- **UI changes:** keep the two-tab structure (ARIMA / Portfolio optimization) unless asked to change it. Favor clear titles, axis labels, units, and short explanatory text. Handle missing or stale data with a helpful message instead of a crash.

## When unsure

If a request might conflict with the hard restrictions (especially whether something counts as "execution"), ask the human before proceeding.
