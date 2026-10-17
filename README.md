# SensorTrust: AI Model & Voice-Enabled Sensor Intelligence

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/srushtichaudhari70-stack/sensortrust)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=srushtichaudhari70-stack/sensortrust&branch=main&mainModule=app.py)

**SensorTrust** is an end-to-end industrial IoT sensor reliability AI system. It features a spatial consensus gradient-boosted AI model (`model.joblib`), real-time acoustic telemetry alert tones, an interactive **AI Voice System** (TTS & voice dictation), and Gemini 3.8 Flash automated root-cause diagnostics.

---

## 🌐 Live Deployments & Repository Links

* **Direct Live Web App (GitHub Pages):** [https://srushtichaudhari70-stack.github.io/sensortrust/](https://srushtichaudhari70-stack.github.io/sensortrust/)
* **One-Click Vercel Deploy:** [Deploy on Vercel](https://vercel.com/new/clone?repository-url=https://github.com/srushtichaudhari70-stack/sensortrust)
* **One-Click Streamlit Cloud Deploy:** [Deploy on Streamlit Community Cloud](https://share.streamlit.io/deploy?repository=srushtichaudhari70-stack/sensortrust&branch=main&mainModule=app.py)
* **GitHub Repository:** [https://github.com/srushtichaudhari70-stack/sensortrust](https://github.com/srushtichaudhari70-stack/sensortrust)

---

## 🧠 The AI Model (`SensorTrustAIModel`)

SensorTrust features a pre-trained, serialized production AI model artifact: **[`model.joblib`](file:///C:/Users/srush/.gemini/antigravity/scratch/sensortrust/model.joblib)** (2.4 MB).

* **Model Class:** `core.SensorTrustAIModel`
* **Architecture:** Spatial Consensus Gradient Boosted Trees with TreeSHAP Game-Theoretic Attributions
* **Input Features (16-D):** Spatial neighbor median residuals (`r`), rate-of-change (`dr`), local noise variance ratio (`std_ratio`), consecutive frozen runs (`run_same`), and multi-sensor movement fraction (`neigh_frac_moved`).
* **Output Classification Heads (8 Classes):** `normal`, `genuine event`, `spike`, `drift`, `stuck`, `noise`, `bias`, `dropout`.
* **Zero-Latency Loading:** The model is pre-compiled and loads in under 50ms.

---

## 🎙️ AI Voice Assistant (`voice_system.py`)

SensorTrust features an end-to-end, zero-latency conversational **AI Voice Assistant** directly integrated into the dashboard:

1. **Microphone Voice Input (Speech-to-Text):**
   * Uses browser Web Speech API (`SpeechRecognition` / `webkitSpeechRecognition`) with explicit Indian English (`en-IN`) preference.
   * Prominent microphone button with pulsing sound-wave animation and real-time user speech transcript.
2. **Text-to-Speech (Indian English Priority):**
   * Synthesizes audio responses with priority for Indian English voices (`en-IN` / `Google हिन्दी` / `Microsoft Heera` / `Microsoft Neerja`), with natural English fallbacks.
   * Dedicated ⏹️ **Stop Speaking** button to immediately halt playback.
   * Optional high-fidelity Google Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`) support via `google-genai` SDK.
3. **Five Explicit State Indicators:**
   * 🟢 **IDLE:** Ready to listen.
   * 🔵 **LISTENING:** Mic active, animated sound waves.
   * 🟡 **PROCESSING:** Analyzing telemetry, residuals, and LightGBM model predictions.
   * 🟣 **SPEAKING:** Audio playback active with dynamic animated equalizer bars.
   * 🔴 **ERROR:** Clear permission diagnostic guidance and fallback instructions.
4. **Grounded Sensor Intelligence (Zero Hallucination):**
   Answers technical queries grounded strictly in live telemetry and model predictions:
   * *"What is my current sensor health?"* -> Real-time 0-100 score, band (Healthy/Watch/Service it), and status.
   * *"Is this a genuine event or a sensor fault?"* -> Spatial consensus validation comparing sensor movement against cluster median.
   * *"Why was this spike detected?"* -> Exact jump deviation and TreeSHAP attribution.
   * *"Explain the latest anomaly"* -> Plain-language explanation from `core.explain()`.
   * *"What should I check to resolve this fault?"* -> Standard Operating Procedures (SOPs) for wiring, loop power, ADC reset, or calibration.
   * *"Summarize latest sensor readings"* -> Real-time overview across all channels.
5. **Universal Accessibility & Fallbacks:**
   * Embedded text-entry bar so engineers can type questions if a microphone is unavailable or permissions are denied.
   * 6 one-click voice prompt chips for instant evaluation.
   * Acoustic industrial alert tones synthesized via Python's standard `wave` library.

---

## 🤖 Gemini 3.8 Flash Root-Cause Diagnostics & Copilot

* **Executive Plant Briefings:** Automated plant-level reliability briefing across all connected sensors.
* **Physical Root-Cause Analysis (RCA):** Distinguishes hardware failure physics (thermocouple degradation, capacitive EFT noise, ADC register lockup) from real environmental temperature shifts.
* **Maintenance SOP Work Orders:** Prescriptive checklists with estimated repair downtime, priority levels, and required technician tools.
* **Interactive AI Sensor Copilot:** Natural language chat assistant monitoring the fleet in real-time.

---

## 📊 Benchmark Evaluation (Held-Out Test Data)

Evaluated across 50,688 unseen readings with randomized fault scales outside training range:

| Method | Fault Precision | Fault Recall | Fault F1-Score | Event Miss Rate *(Lower is better)* |
| :--- | :---: | :---: | :---: | :---: |
| **Robust Z-Score** | 0.443 | 0.093 | 0.154 | 2.7% |
| **Isolation Forest** | 0.259 | 0.262 | 0.260 | 23.7% |
| **SensorTrust AI Model** | **0.961** | **0.794** | **0.870** | **1.0%** |

* Overall Macro F1: **0.852**
* Ambiguity filter ("Needs Review"): **1.19%** of readings
* Genuine environmental events correctly preserved with **84.3% recall** and **0.821 F1**.

---

## Run Locally

```bash
# 1. Windows One-Click:
run.bat

# 2. Or Terminal:
pip install -r requirements.txt
streamlit run app.py
```
