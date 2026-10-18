# 🚀 SensorTrust Production Deployment Guide

This guide provides step-by-step instructions for deploying the **SensorTrust** AI-powered sensor health monitoring system to secure, publicly accessible production environments.

---

## 🏛️ Deployment Architecture Overview

SensorTrust consists of two decoupled, independently deployable tiers:

```
┌────────────────────────────────────────────────────────┐
│                   TIER 1: FRONTEND                     │
│  Option A: GitHub Pages (Stlite WebAssembly in-browser)│
│  Option B: Streamlit Community Cloud (Python native)   │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTPS REST calls
                           │ (POST /api/voice/ask)
                           ▼
┌────────────────────────────────────────────────────────┐
│                   TIER 2: BACKEND                      │
│  FastAPI ASGI Server (api_server.py + Uvicorn)         │
│  Hosted on Render, Railway, or Google Cloud Run        │
│                                                        │
│  🔒 Privately holds GEMINI_API_KEY                     │
│  🔒 Pre-loads trained ML artifact (model.joblib)       │
│  🔒 Restricts CORS to authorized frontend domains      │
└──────────────────────────┬─────────────────────────────┘
                           │ Google Gen AI SDK
                           ▼
┌────────────────────────────────────────────────────────┐
│                  GOOGLE GEMINI AI                      │
│  Model: gemini-3.5-flash-lite / gemini-3.5-flash       │
└────────────────────────────────────────────────────────┘
```

---

## 1. Deploying the Backend (FastAPI on Render Free Tier)

Render provides free hosting for containerized or Python ASGI web services.

### Step 1: Connect Repository to Render
1. Go to [https://dashboard.render.com](https://dashboard.render.com) and sign in.
2. Click **New +** $\to$ **Web Service**.
3. Connect your GitHub repository: `https://github.com/srushtichaudhari70-stack/sensortrust`.

### Step 2: Configure Service Settings
- **Name:** `sensortrust-backend`
- **Region:** Any (e.g., `Oregon (US West)` or `Frankfurt (EU)`)
- **Branch:** `main`
- **Runtime:** `Python 3` (or choose `Docker` to use the bundled `Dockerfile`)
- **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
- **Start Command:** `uvicorn api_server:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips '*'`
- **Plan:** `Free`

### Step 3: Configure Private Environment Variables
In the **Environment Variables** section on Render, add:
| Key | Value | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | `AIza...` or `AQ...` | **Required.** Your private Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey). |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Verified ultra-low-latency model. |
| `ENVIRONMENT` | `production` | Enforces secure CORS and suppresses debug tracebacks. |
| `CORS_ORIGINS` | `https://srushtichaudhari70-stack.github.io` | Your frontend domain (comma-separated if multiple). |

Click **Create Web Service**. Once deployed, Render will provide a public URL like:
`https://sensortrust-backend.onrender.com`

### Step 4: Verify Backend Health
Test the health endpoint in your browser:
`https://sensortrust-backend.onrender.com/health`

Expected response:
```json
{
  "status": "healthy",
  "service": "SensorTrust Gemini AI Voice Backend",
  "gemini_configured": true,
  "model": "gemini-3.5-flash-lite",
  "configuration_status": "ready",
  "version": "2.0.0"
}
```

---

## 2. Deploying the Frontend

You can choose either GitHub Pages or Streamlit Community Cloud (or both).

### Option A: GitHub Pages (Currently Deployed)
The repository includes an `index.html` configured with **Stlite** (WebAssembly Streamlit):
1. In your GitHub repository, navigate to **Settings** $\to$ **Pages**.
2. Under **Build and deployment**, select:
   - **Source:** Deploy from a branch
   - **Branch:** `main` / `/ (root)`
3. Save. GitHub will publish your site at:
   `https://srushtichaudhari70-stack.github.io/sensortrust/`
4. **Connect the Backend:**
   - In the deployed dashboard, open the Voice Assistant panel.
   - Click the **"🌐 Connect Gemini Backend"** badge in the header.
   - Enter your deployed backend URL (e.g. `https://sensortrust-backend.onrender.com`).
   - The badge will turn green: `✨ Gemini AI Online (gemini-3.5-flash-lite)`.

### Option B: Streamlit Community Cloud
1. Go to [https://share.streamlit.io](https://share.streamlit.io).
2. Click **New app** $\to$ select `srushtichaudhari70-stack/sensortrust`, branch `main`, main file `app.py`.
3. In **Advanced settings** $\to$ **Secrets**, configure your backend URL:
   ```toml
   BACKEND_URL = "https://sensortrust-backend.onrender.com"
   ```
4. Click **Deploy**.

---

## 3. Alternative: Docker Deployment (Google Cloud Run / AWS ECS)

Build and run the production container locally or on any cloud provider:

```bash
# 1. Build Docker image
docker build -t sensortrust-backend:latest .

# 2. Run container locally
docker run -d -p 8000:8000 \
  -e GEMINI_API_KEY="your_api_key_here" \
  -e GEMINI_MODEL="gemini-3.5-flash-lite" \
  --name sensortrust sensortrust-backend:latest

# 3. Test health
curl http://localhost:8000/health
```

---

## 4. Free-Tier Operational Constraints & Best Practices

1. **Render Free Tier Cold Starts:**
   - Free instances spin down after 15 minutes of inactivity.
   - The initial request takes approximately 30–50 seconds to wake up.
   - **Zero-Downtime Fallback:** SensorTrust's voice HUD gracefully falls back to the client-side grounded diagnostic engine if the backend is waking up or offline, ensuring uninterrupted demonstrations.
2. **HTTPS Microphone Access:**
   - Modern browsers (Chrome, Edge, Safari) strictly require an HTTPS origin for `SpeechRecognition` and `getUserMedia`.
   - GitHub Pages and Streamlit Cloud are served over HTTPS by default.
3. **Gemini API Free-Tier Quotas:**
   - Google Gemini free tier provides 15 Requests Per Minute (RPM) and 1,000,000 Tokens Per Minute (TPM), which is plenty for demonstrations and interactive testing.
