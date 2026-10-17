# SensorTrust

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/srushtichaudhari70-stack/sensortrust)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=srushtichaudhari70-stack/sensortrust&branch=main&mainModule=app.py)

A working prototype for the problem statement: *an AI-based sensor intelligence system that
distinguishes genuine environmental events from sensor faults, drift, spikes and other unreliable
measurements, and generates an explainable sensor health indication.*

---

## 🌐 Live Web Demos

* **Direct Live Web App (GitHub Pages):** [https://srushtichaudhari70-stack.github.io/sensortrust/](https://srushtichaudhari70-stack.github.io/sensortrust/)
* **One-Click Vercel Deploy:** [Deploy on Vercel](https://vercel.com/new/clone?repository-url=https://github.com/srushtichaudhari70-stack/sensortrust)
* **One-Click Streamlit Cloud Deploy:** [Deploy on Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=srushtichaudhari70-stack/sensortrust&branch=main&mainModule=app.py)

---

## Run Locally

### Option 1: Native Streamlit Server (Local Python)

Double-click `run.bat` on Windows, or execute:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### Option 2: Test the WebAssembly Static Build Locally

```bash
npx serve .
```

### Option 3: Terminal Benchmark Evaluation

```bash
python evaluate.py
```

---

## Benchmark Results (Held-Out Simulated Data)

Evaluated against baseline approaches on 50,688 unseen readings:

| Method | Fault Precision | Fault Recall | Fault F1-Score | Event Miss Rate *(Lower is better)* |
| :--- | :---: | :---: | :---: | :---: |
| **Robust Z-Score** | 0.443 | 0.093 | 0.154 | 2.7% |
| **Isolation Forest** | 0.259 | 0.262 | 0.260 | 23.7% |
| **SensorTrust (Our Model)** | **0.961** | **0.794** | **0.870** | **1.0%** |

* Overall Macro F1: **0.852**
* Ambiguity filter ("Needs Review"): **1.19%** of readings
* Genuine events correctly distinguished with **84.3% recall** and **0.821 F1** without penalizing sensor health scores.

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
| `index.html` | Stlite WebAssembly entrypoint for Vercel / GitHub Pages deployment |
| `vercel.json` | Vercel deployment and routing configuration |
| `core.py` | Simulator, fault injection, features, model, health score, explanations |
| `app.py` | Streamlit dashboard UI |
| `evaluate.py` | Held-out test against z-score and Isolation Forest baselines |
| `requirements.txt` | Python dependencies |
| `run.bat` | One-click Windows launcher |
| `.gitignore` | Excludes `.venv`, caches, and build artifacts |
