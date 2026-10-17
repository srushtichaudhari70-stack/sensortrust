"""SensorTrust FastAPI Backend Server.

Provides a secure proxy API connecting the SensorTrust frontend to Google Gemini AI
while protecting API keys in server-side environment variables and strictly grounding
responses in live SensorTrust telemetry and ML classifications.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import ai_agent

# Load server-side environment variables from local .env using python-dotenv
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass

# Configure safe logging (NEVER print or leak API keys)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("sensortrust_api")

app = FastAPI(
    title="SensorTrust Gemini AI Voice Backend",
    description="Backend API proxy connecting SensorTrust voice assistant to Google Gemini AI.",
    version="2.0.0"
)

# CORS middleware allowing GitHub Pages and local development
allowed_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://srushtichaudhari70-stack.github.io",
]
custom_origins = os.environ.get("CORS_ORIGINS", "")
if custom_origins:
    for o in custom_origins.split(","):
        if o.strip() and o.strip() not in allowed_origins:
            allowed_origins.append(o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, including GitHub Pages previews
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class VoiceAskRequest(BaseModel):
    query: str = Field(..., description="The user's voice transcript or question")
    context: Dict[str, Any] = Field(default_factory=dict, description="Current SensorTrust telemetry context")


class VoiceAskResponse(BaseModel):
    response: str
    source: str  # "gemini" or "grounded_fallback"
    model: str
    status: str  # "ok", "no_key", or "error"
    error: Optional[str] = None


@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health_check():
    """Health check endpoint indicating service availability and Gemini configuration status."""
    conf = ai_agent.validate_gemini_configuration()
    return {
        "status": "healthy",
        "service": "SensorTrust Gemini AI Voice Backend",
        "gemini_configured": conf["configured"],
        "model": conf["model"],
        "configuration_status": conf["status"],
        "message": conf["message"],
        "version": "2.0.0"
    }


@app.post("/api/voice/ask", response_model=VoiceAskResponse)
def voice_ask(req: VoiceAskRequest):
    """Process a voice question using Gemini AI with strict SensorTrust telemetry grounding."""
    query = req.query.strip() if req.query else ""
    if not query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    logger.info(f"Processing question: '{query[:80]}'")
    result = ai_agent.ask_gemini_voice_assistant(query, req.context)
    logger.info(f"Answer generated via {result.get('source')} (status={result.get('status')})")
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting SensorTrust Gemini AI Backend on http://{host}:{port}")
    uvicorn.run("api_server:app", host=host, port=port, reload=False)
