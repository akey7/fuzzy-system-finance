---
title: Fuzzy System Finance
emoji: 🏢
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
license: mit
short_description: A simple portfolio optimization and time series analysis app.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# fuzzy-system-finance
A simple portfolio optimization and time series analysis app. It is the front end for a backend analysis scripts. Data is transferred from the backend to the front end via DigitalOcean Spaces, an S3 hosting service.

## Installation

### Development Dependencies

This project can be installed into a conda environment. First create the conda environment with:

```
conda create -n fuzzy-system-finance python=3.12
```

Then install all development and production dependencies:

```
pip install -r requirements-dev.txt
```

To install only the production dependencies, activate the virtual environment and run the following:

```
pip install -r requirements.txt
```

### `.env` file and environment variables

On the production deployment on HuggingFace Spaces, these environment variables are held by the secrets manager. On a development environment, these are managed in a `.env` file **that should never be committed to GitHub or HuggingFace**.

1. `FSF_FRONT_END_BUCKET_ENDPOINT`, `FSF_FRONT_END_BUCKET_REGION`, `FSF_FRONT_END_BUCKET_READ_ONLY`, `FSF_FRONT_END_BUCKET_READ_ONLY_KEY_ID`: Specifications of the S3/Spaces bucket hosted on AWS or DigitalOcean. This bucket specification points the app to the files from the backend.

2. `PORTFOLIO_OPTIMIZATION_SPACE_NAME`: DigitalOcean space name for portfolio optimization files.

3. `TIME_SERIES_SPACE_NAME`: DigitalOcean space name for time series analysis data.

## Purposes of folders and scripts

### Folders

1. `input/`: Holds input data downloaded from backend. If it does not exist, `app.py` creates it before access.

### Scripts

| Filename           | Standalone Execution? | Purpose                                                | Downloads from front end? |
| ------------------ | --------------------- | ------------------------------------------------------ | ------------------------- |
| `app.py`           | Yes                   | Run the Fuzzy System Finance Gradio app.               | Yes                       |
| `s3_downloader.py` | No                    | Downloads data from S3 buckets for display on the app. | Yes (used by `app.py`)    |
