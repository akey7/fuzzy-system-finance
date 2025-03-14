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
short_description: fuzzy-system-finance is coming soon!
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# fuzzy-system-finance
A sample time series finance app.

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

### Production Dependencies

To install only the production dependencies, activate the virtual environment and run the following:

```
pip install -r requirements.txt
```

### `.env` file

**Never commit the `.env` file to GitHub!** It should never be made publicly accessible because it contains API keys.

Here are the API keys it needs:

1. `MARKETSTACK_API_KEY`: Your key to the MarketStack API. It needs to be on a price tier that gives 10 years of history and end-of-day prices.
