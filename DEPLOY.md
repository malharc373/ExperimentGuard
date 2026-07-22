# Deploying ExperimentGuard to Streamlit Community Cloud

The Streamlit demo (`app.py`) deploys to [Streamlit Community Cloud](https://share.streamlit.io)
for free. The repo is already set up for it — `requirements.txt` and
`.streamlit/config.toml` are at the root, and `app.py` imports only from the
`experimentguard` package (no local data files required, since the sample
scenarios are generated in memory).

## Prerequisites

- The repo is on GitHub and **public** (it is: `malharc373/ExperimentGuard`).
- A Streamlit Community Cloud account, signed in with the same GitHub account.

## Deploy

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `malharc373/ExperimentGuard`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. (Optional) Click **Advanced settings** and pin **Python version 3.11** to match CI.
5. Click **Deploy**. First build takes 1–3 minutes while dependencies install.

Your app gets a URL like `https://experimentguard-<hash>.streamlit.app`.

## What Streamlit Cloud reads

| File | Purpose |
|---|---|
| `requirements.txt` | Installed with `pip` in the build. `streamlit` is already listed. |
| `.streamlit/config.toml` | Applies the dark instrument theme. |
| `app.py` | Entry point. |

No secrets or environment variables are needed — the app has no external
dependencies or API keys.

## Updating

Every push to `main` triggers an automatic rebuild and redeploy. To update the
live app, just:

```bash
git push origin main
```

You can also **Reboot** or **Delete** the app from the Streamlit Cloud dashboard.

## Troubleshooting

- **App is asleep / "Yes, get this app back up":** free apps sleep after a period
  of inactivity. The first visitor wakes it (a few seconds). This is expected.
- **`ModuleNotFoundError`:** a dependency is missing from `requirements.txt`.
  Add it, commit, and push. Confirm locally first with
  `pip install -r requirements.txt`.
- **Fonts not loading:** the theme uses Google Fonts via `@import`. Streamlit
  Cloud allows this; if a corporate network blocks Google Fonts, the app falls
  back to the system sans-serif and still works.
- **Wrong Python version:** set it under Advanced settings before deploying, or
  edit it later via **Manage app → Settings**.

## Local preview (parity check before deploying)

```bash
source .venv/bin/activate
streamlit run app.py
# opens http://localhost:8501
```

What you see locally is what Cloud serves, since both read the same
`requirements.txt` and `.streamlit/config.toml`.
