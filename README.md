# SensorTrust

A working prototype for the problem statement: *an AI-based sensor intelligence system that
distinguishes genuine environmental events from sensor faults, drift, spikes and other unreliable
measurements, and generates an explainable sensor health indication.*

## Run it

Windows: double-click `run.bat`. It installs the libraries the first time, then opens the dashboard.

Or from a terminal in this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501. The first start takes about 15 seconds while the
model trains.

To print the test results in a terminal:

```bash
python evaluate.py
```

## What it does

1. **Sanity rules** catch missing readings (dropout) and frozen readings (stuck) outright.
2. **Ask the neighbours**: every sensor is compared with the median of the other sensors. A fault
   shows on one sensor; a real event shows on all of them.
3. **Classifier** (LightGBM) labels each reading: normal, genuine event, spike, drift, noise or bias.
   If it is not confident whether a sensor is at fault, it says "needs review".
4. **Explain and score**: each verdict gets a plain-language reason built from measured numbers, the
   model's top evidence (TreeSHAP values), a suggested action, and a 0-100 health score per sensor.
   A genuine event never lowers a sensor's health score.

## Using the dashboard

- **Scenario**: pick a ready-made situation (faults and events, events only, faults only, clean).
- **Break something yourself**: add any fault to any sensor, or add a genuine event, and watch the verdict.
- **Replay slider**: move through time to see health scores fall and recover.
- **Why?**: pick an incident to read the reason and the suggested action.
- **How well does it work?**: runs the test against two simple baselines.
- **Your own CSV** (experimental): first column a timestamp, then one column per sensor. Needs at
  least 3 sensors measuring the same quantity at evenly spaced times.

## Files

| File | Purpose |
|---|---|
| `core.py` | Simulator, fault injection, features, model, health score, explanations |
| `app.py` | Streamlit dashboard |
| `evaluate.py` | Held-out test against z-score and Isolation Forest baselines |
| `requirements.txt` | Python dependencies |
| `run.bat` | One-click launcher script for Windows |

## Limits to be upfront about

- The data is **simulated** temperature-like sensors. Nothing here has been tested on real sensor data yet.
- The model is trained and tested on faults from the same simulator. The test uses new random runs
  and fault sizes outside the training range, but real faults will look different.
- It needs several sensors measuring the same thing. With a single sensor there are no neighbours to ask.
- Verdicts use a look-ahead of two readings (10 minutes), so the live verdict lags by that much.
- Slow drift is only flagged once the gap to neighbours passes about 0.5 °C.
- The health-score formula and its 80 / 50 bands are our own design choices, not a standard.
