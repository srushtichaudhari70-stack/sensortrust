# 🏆 SensorTrust: Hackathon Pitch & Presentation Playbook

Use this playbook to deliver a winning hackathon presentation, live demo, and portal submission.

---

## 🎯 1. The 30-Second Elevator Pitch

> *"Every day, industrial plants and smart cities lose millions to sensor false alarms and undetected drift. Current anomaly detection algorithms have a fatal flaw: when a room gets hot, they flag every working sensor as broken!  
> **SensorTrust** solves this with peer-consensus machine learning. By cross-validating spatial neighbors in real-time, our LightGBM model distinguishes genuine environmental shifts from 6 distinct hardware faults with **96.1% precision**, while **Gemini 3.8 Flash** generates audible root-cause voice diagnostics and automated maintenance work orders. It’s the difference between a false emergency shutdown and actionable intelligence."*

---

## 🎬 2. The 3-Minute Live Demo Script (Step-by-Step for Judges)

| Time | What to Click / Show | What to Say |
|---|---|---|
| **0:00 - 0:30** | Open **Tab 1 (Telemetry)** with Preset *"Mixed: six faults and two real events"* | *"Here is a 48-hour cluster of 8 industrial sensors. Notice how at hour 16 and 44, all sensors shift upward together. Traditional Z-score triggers false alarms, but SensorTrust recognizes a collective consensus event—protecting sensor health."* |
| **0:30 - 1:15** | Go to **Sidebar: "Break something yourself"**<br>Add a **Drift** on **S3**, click **Add** | *"Let's simulate a real-world hardware failure. I'll inject a slow thermal resistance drift into Sensor 3. Notice how S3 steadily pulls away from the cluster median. SensorTrust catches it, drops S3's health into the 'Watch' band, and TreeSHAP highlights exactly which features drove the verdict."* |
| **1:15 - 2:00** | Go to **Tab 2 (AI Root-Cause & Voice)**<br>Select S3 Incident, click **Run AI Diagnostic** and **🔊 Read Diagnosis (Voice)** | *"Instead of raw telemetry charts, our Gemini 3.8 Flash integration translates ML weights into an audible field engineering diagnosis. Listen to the physical root-cause explanation—it diagnoses electrochemical thermistor aging and gives technicians a step-by-step SOP work order."* |
| **2:00 - 2:30** | Go to **Tab 3 (AI Sensor Copilot)**<br>Click chip: *“Why is S3 failing?”* | *"Technicians on the factory floor can speak directly to the AI Sensor Copilot via voice dictation to query maintenance schedules, safety risks, and calibration standards."* |
| **2:30 - 3:00** | Open **Tab 5 (Benchmark Baselines)** | *"On 50,688 unseen test readings, SensorTrust beats Isolation Forest by over 3x on F1-score (0.870 vs 0.260) and reduces false event alarms down to just 1.0%. Best of all, it's deployed live on the web!"* |

---

## 📑 3. Hackathon Portal Submission Write-up (Devpost / Unstop / HackerEarth)

### Project Title
**SensorTrust: Spatial Consensus AI Model & Voice-Enabled IoT Diagnostic Intelligence**

### Tagline
*Distinguishing genuine environmental events from hardware sensor faults with 96% precision and automated Gemini 3.8 Flash voice diagnostics.*

### The Problem
Industrial IoT networks rely on thousands of temperature, pressure, and vibration sensors. However:
1. **False Alarms:** Standard anomaly detectors (Z-score, Isolation Forest) flag collective environmental events (e.g. heating cycles, weather fronts) as sensor failures.
2. **Silent Degradation:** Electrochemical drift and stuck ADC registers go unnoticed until equipment fails.
3. **Unexplainable Black Boxes:** Operators don't know *why* a sensor was flagged or what physical maintenance to perform.

### What It Does
* **Peer-Consensus Intelligence:** Compares each sensor against dynamic median peer clusters, neutralizing shared environmental shifts.
* **LightGBM Multi-Head Classification:** Classifies readings into 8 states: `normal`, `genuine event`, `spike`, `drift`, `stuck`, `noise`, `bias`, and `dropout`.
* **Explainable TreeSHAP Weights:** Every flagged reading provides game-theoretic feature importances.
* **Continuous 0–100 Health Scoring:** Fades historical noise while heavily penalizing verified hardware failures.
* **Gemini 3.8 Flash Diagnostics & Voice System:** Synthesizes physical failure mechanics (e.g., thermal paste desiccation, capacitive EFT coupling) and speaks work orders aloud via text-to-speech.

### How We Built It
* **Core Machine Learning:** LightGBM, Scikit-learn, NumPy, Pandas, TreeSHAP
* **Generative AI & Audio:** Google Gemini 3.8 Flash (`gemini-3.8-flash`), Gemini TTS (`gemini-3.8-flash-tts`), Web Speech API
* **Interactive Frontend:** Streamlit, Plotly, Stlite WebAssembly
* **Deployment:** Vercel Edge CDN & GitHub Pages

### Accomplishments That We're Proud Of
* Achieved **0.961 fault precision** and **0.870 F1-score** on held-out benchmarks vs. 0.260 for Isolation Forest.
* Achieved **1.0% event miss rate**, virtually eliminating false emergency shutdowns.
* Enabled **zero-server WebAssembly execution** via Stlite so anyone can run the complete AI model in their browser.

---

## 📊 4. Key Metrics to Highlight on Slides

* **0.961** Fault Precision (vs 0.259 Isolation Forest)
* **0.870** Fault F1-Score (vs 0.260 Isolation Forest)
* **1.0%** Event Miss Rate (vs 23.7% Isolation Forest)
* **<50ms** AI Model Inference Latency
* **100% Client-Side Ready** with WebAssembly & Web Speech API
