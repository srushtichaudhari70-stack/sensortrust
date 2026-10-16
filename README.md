# SensorTrust

A working prototype for the problem statement: *an AI-based sensor intelligence system that
distinguishes genuine environmental events from sensor faults, drift, spikes and other unreliable
measurements, and generates an explainable sensor health indication.*

---

## Deploy to Vercel

SensorTrust is configured to run on **Vercel** as a serverless WebAssembly web application via **Stlite** (Pyodide in-browser runtime). This allows the entire LightGBM model, Streamlit UI, and SHAP explainability engine to execute client-side with zero server maintenance and zero hosting costs.

### Option A: Deploy via Vercel CLI

In PowerShell / Terminal from the project directory:

```bash
# 1. Log in to your Vercel account
npx vercel login

# 2. Deploy directly to production
npx vercel --prod
```

When prompted:
- Set up and deploy: **`Y`**
- Which scope: *(select your personal account or team)*
- Link to existing project: **`N`**
- Project name: **`sensortrust`**
- In which directory is your code located: **`./`**
- Want to modify settings: **`N`**

Your app will be deployed and available at `https://sensortrust-<your-username>.vercel.app`!

### Option B: Deploy via GitHub + Vercel Dashboard

1. Create a repository on [GitHub](https://github.com/new) (e.g. `sensortrust`).
2. Push this directory to your new repo:
   ```bash
   git remote add origin https://github.com/<your-username>/sensortrust.git
   git branch -M main
   git push -u origin main
   ```
3. Go to [vercel.com/new](https://vercel.com/new) and click **Import** next to your `sensortrust` repository.
4. Leave all build settings at their defaults and click **Deploy**.

---

## Run Locally

### Option 1: Native Streamlit Server (Local Python)

Double-click `run.bat` on Windows, or execute:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### Option 2: Test the Vercel Static Build Locally

To test the exact WebAssembly build that Vercel serves:

```bash
npx serve .
```

Open the provided local URL (e.g., `http://localhost:3000`) in your browser.

### Option 3: Terminal Benchmark Evaluation

To inspect benchmark metrics against Z-score and Isolation Forest baselines:

```bash
python evaluate.py
```

---

## What it does

1. **Sanity rules** catch missing readings (dropout) and frozen readings (stuck) outright.
2. **Ask the neighbours**: every sensor is compared with the median of the other sensors. A fault
   shows on one sensor; a real event shows on all of them.
3. **Classifier** (LightGBM) labels each reading: normal, genuine event, spike, drift, noise or bias.
   If it is not confident whether a sensor is at fault, it says "needs review".
4. **Explain and score**: each verdict gets a plain-language reason built from measured numbers, the
   model's top evidence (TreeSHAP values), a suggested action, and a 0-100 health score per sensor.
   A genuine event never lowers a sensor's health score.

---

## Files

| File | Purpose |
|---|---|
| `index.html` | Stlite WebAssembly entrypoint for Vercel deployment |
| `vercel.json` | Vercel deployment and routing configuration |
| `core.py` | Simulator, fault injection, features, model, health score, explanations |
| `app.py` | Streamlit dashboard UI |
| `evaluate.py` | Held-out test against z-score and Isolation Forest baselines |
| `requirements.txt` | Python dependencies |
| `run.bat` | One-click Windows launcher |
| `.gitignore` | Excludes `.venv`, caches, and build artifacts |

---

## Limits to be upfront about

- The data is **simulated** temperature-like sensors. Nothing here has been tested on real sensor data yet.
- The model is trained and tested on faults from the same simulator. The test uses new random runs
  and fault sizes outside the training range, but real faults will look different.
- It needs several sensors measuring the same thing. With a single sensor there are no neighbours to ask.
- Verdicts use a look-ahead of two readings (10 minutes), so the live verdict lags by that much.
- Slow drift is only flagged once the gap to neighbours passes about 0.5 °C.
- The health-score formula and its 80 / 50 bands are our own design choices, not a standard.
