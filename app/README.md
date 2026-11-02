# Crash Verdict Console, Streamlit app

The web interface for the on-device crash classifier. It runs the **same models the ESP32 runs**:
the flattened LightGBM and XGBoost trees exported by `ml/scripts/export_model.py`, evaluated with
numpy. No xgboost, lightgbm or scikit-learn at runtime, so it starts fast and hosts cheaply.

```
streamlit_app.py      the app (entry point)
app/engine.py         trigger + features + tree inference (mirrors the firmware)
app/data/models.npz   both models as flat arrays   (240 KB)
app/data/demo.npz     replay drive + event library (252 KB)
app/data/meta.json    class names, event descriptions, result tables (27 KB)
requirements.txt      streamlit, numpy, pandas, altair
```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens on http://localhost:8501. Views are shareable: `?view=Drive+replay`, `?view=Event+library`,
`?view=Model+comparison`, `?view=Test+your+own+data`, `?view=How+it+runs+on+the+ESP32`.

## Host it on Streamlit Community Cloud (free)

The app is already sized for it, the whole payload is about 520 KB, and no dataset needs to ship.

1. **Push this repo to GitHub.** The app must be on a branch the cloud can read:
   ```bash
   git add .
   git commit -m "Add Streamlit interface for the crash classifier"
   git push origin main
   ```
2. Sign in at **https://share.streamlit.io** with the GitHub account that owns the repo.
3. **Create app → Deploy a public app from a repo**, then fill in:
   - Repository: `Arush-O7/Smart-Accident-Detection-Emergency-Alert-System`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. **Deploy.** The first build takes a couple of minutes; after that it redeploys on every push.
   You get a URL like `https://<app-name>.streamlit.app` to hand to anyone.

Notes:
- `.streamlit/config.toml` carries the theme; the cloud picks it up automatically.
- Everything the app needs is committed. `ml/data/` (the 7.6 GB of datasets) is gitignored and is
  **not** required to run or host the app, only to retrain.
- If you fork rather than push to the original repo, deploy from your fork.

## Rebuild the app's data after retraining

```bash
python ml/scripts/export_model.py       # firmware headers + demo/data/model.js
python ml/scripts/export_streamlit.py   # app/data/*.npz + meta.json
```

## What it cannot do

Live USB streaming from the ESP32 needs direct serial access, which a hosted web app cannot have.
Use the HTML dashboard in `demo/index.html` (Chrome or Edge, Web Serial) for the live hardware view,
or paste a serial capture into the app's **Test your own data** page.
