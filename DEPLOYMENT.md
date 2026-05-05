# Streamlit Cloud Deployment

This dashboard is designed to run from live SERENE API data in production.
The large `local_data/` folder is only a local fallback/demo dataset and should
not be uploaded to GitHub or Streamlit Cloud.

## Required Secret

In Streamlit Community Cloud, add this app secret:

```toml
SERENE_API_TOKEN = "your-serene-api-token"
```

The app stores downloaded SERENE output in `api_cache/`, which is generated at
runtime and should not be committed.

## Entry Point

Use this file as the Streamlit entry point:

```text
dashboard.py
```

## Local Run

For local development, either set `SERENE_API_TOKEN` in the environment or use
the AIDA default config file:

```text
~/.config/aida/api_config.ini
```

If the SERENE token is unavailable, keep `local_data/` temporarily for offline
demo and testing.
