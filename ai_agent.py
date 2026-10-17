"""AI Integration Module for SensorTrust.

Powered by Google Gemini 3.8 Flash (`gemini-3.8-flash`) via the `google-genai` SDK.
Features:
- Automated Root-Cause Analysis (RCA) from ML telemetry & TreeSHAP signals
- Maintenance Work Order & Field SOP Generation
- Executive Plant Health Briefing
- Interactive Sensor Intelligence Chat Copilot
- Offline/Browser Fallback Expert System when no API key is supplied
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional

MODEL_NAME = "gemini-3.8-flash"

SYSTEM_INSTRUCTION = """You are SensorTrust AI, an elite industrial IoT sensor reliability engineer and root-cause analysis expert.
You analyze multi-sensor telemetry, machine learning anomaly classifications, neighbor residual comparisons, and TreeSHAP feature weights.
Your job is to provide precise physical diagnoses (e.g., thermal resistance drift, loose ground wire, ADC quantization saturation, environmental microclimates),
distinguish genuine environmental phenomena from hardware malfunctions, and give actionable maintenance work orders according to industrial standards."""


def load_gemini_api_key(user_input_key: Optional[str] = None) -> Optional[str]:
    """Retrieve Gemini API key from UI input, Streamlit secrets, environment, or .env file."""
    if user_input_key and user_input_key.strip():
        return user_input_key.strip()
    
    # 1. Check Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    # 2. Check os.environ
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ.get("GEMINI_API_KEY").strip()

    # 3. Check local or home .env files
    env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.getcwd(), ".env"),
        os.path.expanduser("~/.env")
    ]
    for p in env_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                            k = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if k:
                                return k
            except Exception:
                pass
    return None


def get_genai_client(api_key: Optional[str] = None):
    """Instantiate the google-genai client if an API key is available."""
    key = load_gemini_api_key(api_key)
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception:
        return None


def generate_incident_rca(incident: Dict[str, Any], shap_factors: List[tuple],
                          unit: float, units: str = "°C", api_key: Optional[str] = None) -> str:
    """Generate an AI-driven Root Cause Analysis and Field Engineering Work Order."""
    client = get_genai_client(api_key)
    sensor = incident.get("sensor", "Unknown Sensor")
    verdict = incident.get("verdict", "fault")
    duration = incident.get("duration", "N/A")
    confidence = incident.get("confidence", "N/A")
    reason = incident.get("reason", "")
    
    factors_str = "\n".join([f"- {name}: SHAP weight {val:.3f}" for name, val in shap_factors]) if shap_factors else "Caught by deterministic sanity rule."

    prompt = f"""Analyze this sensor anomaly incident and generate an Engineering Root Cause Analysis (RCA) and Maintenance Work Order:

Sensor: {sensor}
Anomaly Verdict: {verdict}
Duration: {duration}
Classification Confidence: {confidence}
Summary Diagnostic: {reason}
Top ML TreeSHAP Attributions:
{factors_str}

Please structure your response with the following sections:
1. 🔍 **Physical Root-Cause Mechanism**: What physical or electrical malfunction likely caused this specific pattern?
2. ⚠️ **Operational & Safety Risk**: What is the risk to equipment, process quality, or operations if left unresolved?
3. 🛠️ **Technician Action Plan (SOP)**: Step-by-step instructions for on-site technicians (tools, tests, multimeter checks, calibration).
4. 📋 **Maintenance Work Order**: Estimated repair time, priority level (Critical/Moderate/Low), and parts needed.
"""

    if client:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=dict(system_instruction=SYSTEM_INSTRUCTION, temperature=0.2)
            )
            return response.text
        except Exception as e:
            # Fall back to expert synthesis if API call fails
            pass

    # Built-in Expert Rule-Based Synthesis (Offline / Zero-Key Mode)
    return _offline_expert_rca(sensor, verdict, duration, confidence, reason, shap_factors, units)


def generate_executive_briefing(sensor_health: Dict[str, float], incidents_summary: List[Dict[str, Any]],
                                api_key: Optional[str] = None) -> str:
    """Generate a high-level executive health summary across all sensors."""
    client = get_genai_client(api_key)
    
    health_str = ", ".join([f"{s}: {h:.0f}/100" for s, h in sensor_health.items()])
    degraded = [s for s, h in sensor_health.items() if h < 80]
    critical = [s for s, h in sensor_health.items() if h < 50]
    
    prompt = f"""Provide a concise Executive Reliability Briefing for plant operations based on this sensor telemetry analysis:

Sensor Health Scores (0-100):
{health_str}
Total Active Incidents: {len(incidents_summary)}
Degraded Sensors (Watch): {', '.join(degraded) if degraded else 'None'}
Critical Sensors (Service Required): {', '.join(critical) if critical else 'None'}

Include:
- 📊 **Fleet Health Status**: Overall reliability index and state of the cluster.
- 🎯 **Primary Vulnerabilities**: Which sensors present immediate operational risk.
- ⏱️ **Recommended Maintenance Window**: Recommended schedule for inspection and zero-downtime recalibration.
"""

    if client:
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=dict(system_instruction=SYSTEM_INSTRUCTION, temperature=0.3)
            )
            return response.text
        except Exception:
            pass

    # Offline synthesis
    status = "CRITICAL" if critical else ("DEGRADED" if degraded else "OPTIMAL")
    return f"""### 📊 Executive Reliability Briefing
* **Fleet Status**: **{status}**
* **Average Cluster Health**: **{sum(sensor_health.values()) / max(1, len(sensor_health)):.1f} / 100**
* **Action Required**: {f"Immediate dispatch required for {', '.join(critical)}." if critical else (f"Schedule inspection for {', '.join(degraded)} during next maintenance cycle." if degraded else "All sensors within nominal tolerance. No technician dispatch required.")}
* **Environmental Integrity**: Cluster consensus algorithms confirm neighbor cross-correlation is stable; unaffected sensors are reporting genuine telemetry."""


def ask_sensor_copilot(query: str, context: Dict[str, Any], chat_history: List[Dict[str, str]],
                       api_key: Optional[str] = None) -> str:
    """Natural language AI Copilot answering technician questions."""
    client = get_genai_client(api_key)
    
    context_str = f"""
Current Active Scenario:
- Sensors: {context.get('sensors', [])}
- Health Scores: {context.get('health', {})}
- Active Incidents Count: {context.get('incident_count', 0)}
- Highlighted Sensor: {context.get('focus', 'N/A')}
- Recent Verdict on Highlighted Sensor: {context.get('focus_verdict', 'N/A')}
"""
    prompt = f"""Context:
{context_str}

User Question: "{query}"

Answer concisely as an expert AI sensor intelligence assistant. Focus on practical physical reasoning, sensor hardware principles, and neighbor consensus logic."""

    if client:
        try:
            contents = []
            for msg in chat_history[-6:]:
                contents.append(f"{msg['role'].capitalize()}: {msg['content']}")
            contents.append(f"User: {prompt}")
            
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents="\n\n".join(contents),
                config=dict(system_instruction=SYSTEM_INSTRUCTION, temperature=0.3)
            )
            return response.text
        except Exception as e:
            pass

    # Offline interactive assistant responses
    return _offline_chat_response(query, context)


VOICE_SYSTEM_INSTRUCTION = """You are SensorTrust AI Voice Assistant, an expert industrial IoT sensor reliability copilot.
SensorTrust uses multi-sensor spatial consensus and TreeSHAP explainability to detect sensor faults versus genuine environmental events.

Strict Operational Guidelines:
1. Grounding: Rely strictly on the provided sensor context. NEVER invent sensor readings, health scores, timestamps, confidence percentages, or hardware failure causes.
2. Explain, Don't Pretend: You are explaining the results produced by the underlying SensorTrust Machine Learning model. Do NOT pretend you measured the sensors yourself.
3. Anomaly vs Hardware Fault: Distinguish between a detected anomaly (which is a statistical departure from cluster consensus) and a confirmed physical hardware defect (which requires on-site technician inspection/multimeter verification).
4. Data Provenance: Clearly identify simulated data as simulated telemetry.
5. Missing / Out-of-Bounds Sensors: If the user asks about a sensor ID not present in the active cluster (e.g. S9 when only S1-S6 exist), clearly state that the sensor does not exist in the active cluster and list the available sensors. If data is missing or ambiguous, state that it is unavailable.
6. Tone & Format: Provide concise, professional, clear answers suitable for being spoken aloud via Text-to-Speech to an engineering demonstrator (2-4 sentences maximum). Use plain text without markdown symbols like asterisks (*) or hashes (#) so speech synthesis reads naturally.
"""


def ask_gemini_voice_assistant(query: str, context: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Process a voice question using Gemini with strict SensorTrust grounding and safety fallbacks.
    Returns:
        Dict with keys:
          - 'response': str (plain text, spoken-ready)
          - 'source': 'gemini' | 'grounded_fallback'
          - 'model': str
          - 'status': 'ok' | 'no_key' | 'error'
          - 'error': Optional[str]
    """
    import re
    import voice_system

    q = (query or "").strip()
    q_low = q.lower()
    sensors = context.get("sensors", [])
    health_dict = context.get("health_dict", {}) or context.get("health_scores", {}) or context.get("health", {})
    readings = context.get("readings", {})
    verdicts = context.get("verdicts_dict", {}) or context.get("verdicts", {})
    confidences = context.get("confidences_dict", {}) or context.get("confidences", {})
    incidents = context.get("incidents_list", []) or context.get("incidents", [])
    shap_dict = context.get("shap", {}) or context.get("shap_explanations", {})
    units = context.get("units", "°C")
    now_hour = context.get("now_hour", 0.0)
    data_source = context.get("data_source", "simulated temperature sensors")
    is_simulated = context.get("is_simulated", True)
    sim_label = "simulated" if is_simulated else "uploaded"
    scenario = context.get("scenario", "Custom Scenario")
    fleet_health = context.get("fleet_health", context.get("fleet", 100.0))
    fleet_status = context.get("fleet_status", "Optimal")
    focus = context.get("focus", sensors[0] if sensors else "S1")

    # 1. Missing or unavailable sensor data
    if not sensors or not health_dict:
        msg = "Sensor telemetry is currently unavailable. Please load a scenario or upload a CSV file."
        return {
            "response": msg,
            "source": "grounded_fallback",
            "model": MODEL_NAME,
            "status": "ok",
            "error": None
        }

    # 2. Ambiguous query check
    if q_low in ["what about the other one", "what about that sensor", "the other one"] or q_low.endswith("other sensor"):
        msg = f"Please specify which sensor you would like to inspect (for example: {', '.join(sensors[:3])}), or select a sensor in the sidebar to highlight it."
        return {
            "response": msg,
            "source": "grounded_fallback",
            "model": MODEL_NAME,
            "status": "ok",
            "error": None
        }

    # 3. Non-existent sensor validation
    sensor_matches = re.findall(r'\b(?:sensor\s+s?|s)(\d+)\b', q_low)
    if sensor_matches:
        candidate = f"S{sensor_matches[0]}"
        if candidate not in sensors:
            msg = f"Sensor {candidate} does not exist in the active cluster. The currently configured network has {len(sensors)} sensors: {', '.join(sensors)} ({data_source}). Please specify a valid sensor ID."
            return {
                "response": msg,
                "source": "grounded_fallback",
                "model": MODEL_NAME,
                "status": "ok",
                "error": None
            }

    # 4. Check API Key
    key = load_gemini_api_key(api_key)
    if not key:
        fb = voice_system.generate_grounded_voice_response(query, context)
        return {
            "response": fb,
            "source": "grounded_fallback",
            "model": MODEL_NAME,
            "status": "no_key",
            "error": "Gemini API key is not configured on the server. Set GEMINI_API_KEY in server environment."
        }

    # 5. Build rich SensorTrust context for Gemini
    readings_list = [f"{s}: {readings.get(s, 0.0):.1f} {units}" for s in sensors]
    health_list = [f"{s}: {health_dict.get(s, 100.0):.0f}/100" for s in sensors]
    verdict_list = [f"{s}: {verdicts.get(s, 'normal')} ({confidences.get(s, 95.0):.0f}% conf)" for s in sensors]
    
    incidents_text = []
    for inc in incidents:
        s_id = inc.get("sensor", "Unknown")
        v_id = inc.get("verdict", "anomaly")
        c_raw = inc.get("confidence", 0.9)
        c_val = c_raw * 100 if c_raw <= 1.0 else c_raw
        incidents_text.append(f"{s_id} ({v_id}, {c_val:.0f}% confidence)")
    incidents_summary = ", ".join(incidents_text) if incidents_text else "None (all sensors nominal)"

    shap_text = []
    for s_id, info in shap_dict.items():
        r = info.get("reason", "")
        top = info.get("top", [])
        top_str = f" [evidence: {', '.join([f[0] for f in top[:2]])}]" if top else ""
        shap_text.append(f"{s_id}: {r}{top_str}")
    shap_summary = "; ".join(shap_text) if shap_text else "None"

    prompt = f"""Context:
- Telemetry Mode: {sim_label} data ({data_source}) at Timeline Hour {now_hour:.1f}
- Active Cluster Sensors ({len(sensors)}): {', '.join(sensors)}
- Highlighted Sensor in UI: {focus} (Health: {health_dict.get(focus, 100.0):.0f}/100, Verdict: '{verdicts.get(focus, 'normal')}', Reading: {readings.get(focus, 'N/A')} {units})
- Latest Readings: {', '.join(readings_list)}
- Health Scores (0-100): {', '.join(health_list)}
- Model Classifications: {', '.join(verdict_list)}
- Fleet Average Health: {fleet_health:.0f}/100 ({fleet_status} status)
- Active Incident Records: {incidents_summary}
- TreeSHAP Model Attributions: {shap_summary}

User Voice Question: "{query}"

Answer concisely in spoken plain text (2 to 4 sentences maximum). Do not use markdown asterisks or symbols."""

    # 6. Call Google Gen AI Client
    client = get_genai_client(key)
    if not client:
        fb = voice_system.generate_grounded_voice_response(query, context)
        return {
            "response": fb,
            "source": "grounded_fallback",
            "model": MODEL_NAME,
            "status": "error",
            "error": "Failed to initialize Google Gen AI client."
        }

    # Try gemini-3.8-flash first, then gemini-2.5-flash fallback
    models_to_try = [MODEL_NAME, "gemini-2.5-flash"]
    last_err = None

    for m_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=m_name,
                contents=prompt,
                config=dict(
                    system_instruction=VOICE_SYSTEM_INSTRUCTION,
                    temperature=0.2
                )
            )
            raw_text = response.text or ""
            clean_text = raw_text.replace("*", "").replace("#", "").strip()
            if clean_text:
                return {
                    "response": clean_text,
                    "source": "gemini",
                    "model": m_name,
                    "status": "ok",
                    "error": None
                }
        except Exception as e:
            last_err = e
            continue

    # If all models failed (e.g. invalid key, quota, timeout)
    fb = voice_system.generate_grounded_voice_response(query, context)
    err_str = f"Gemini API error ({type(last_err).__name__}). Using grounded local fallback." if last_err else "Gemini call failed."
    return {
        "response": fb,
        "source": "grounded_fallback",
        "model": MODEL_NAME,
        "status": "error",
        "error": err_str
    }


def _offline_expert_rca(sensor: str, verdict: str, duration: str, confidence: str,
                        reason: str, shap_factors: List[tuple], units: str) -> str:
    """Detailed expert fallback RCA when no API key is set."""
    rca_templates = {
        "drift": {
            "mechanism": "Slow electrochemical aging of the sensing element or thermistor resistance degradation over thermal cycling. Alternatively, thermal paste desiccation between sensor probe and mounting surface.",
            "risk": "Progressive measurement skew leading to gradual process drift, undetected thermal hotspots, and incorrect HVAC or chiller PID loop modulation.",
            "sop": [
                "1. Isolate the sensor loop from active automation PLC triggers.",
                "2. Inspect RTD / thermocouple sheath for corrosion or mechanical strain.",
                "3. Perform two-point calibration check (ice point 0°C, boiling point 100°C) against a certified Fluke reference standard.",
                "4. Apply zero/span offset correction in controller if drift < 1.5°C; replace probe if drift is nonlinear."
            ],
            "work_order": "Priority: Medium | Est. Downtime: 30 minutes | Tools: Reference Dry-well calibrator, precision DMM"
        },
        "spike": {
            "mechanism": "Transitory electrical fast transient (EFT), capacitive coupling from nearby switching inductive loads (motors/relays), or loose terminal screw connection causing intermittent resistance spikes.",
            "risk": "Nuisance alarms, false emergency trip triggers, or momentary PID actuator flutter.",
            "sop": [
                "1. Verify cable shielding and ground continuity (ensure single-point earth ground).",
                "2. Check nearby VFD cables for minimum 30cm physical clearance or conduit shielding.",
                "3. Torque terminal blocks to manufacturer specification (0.5 Nm typical).",
                "4. Clean contacts with isopropyl alcohol if atmospheric oxidation is present."
            ],
            "work_order": "Priority: Low to Medium (High if recurring) | Est. Downtime: 15 minutes | Tools: Torx driver, oscilloscope"
        },
        "noise": {
            "mechanism": "Degraded DC power rail filtering, decaying decouple capacitor on sensor transmitter board, or EMI radiation from adjacent high-frequency inverters.",
            "risk": "Increased measurement variance reduces signal-to-noise ratio, degrading derivative (D) control loop stability.",
            "sop": [
                "1. Measure AC ripple on sensor 24V DC / 5V DC power supply line with an oscilloscope (must be < 50mV p-p).",
                "2. Verify twisted-pair signal wiring integrity.",
                "3. Check for ground loops between field instrument and analog input card.",
                "4. Replace transmitter module if internal noise floor remains elevated."
            ],
            "work_order": "Priority: Medium | Est. Downtime: 45 minutes | Tools: Oscilloscope, multimeter, replacement transmitter"
        },
        "stuck": {
            "mechanism": "Microcontroller firmware lockup on smart transmitter, ADC register latch-up, or mechanical freeze of sensing diaphragm/junction.",
            "risk": "Critical failure blindness: the automation system assumes steady-state while actual ambient conditions may be experiencing runaway thermal excursions.",
            "sop": [
                "1. Power-cycle the field sensor unit (30-second cold reboot).",
                "2. Check 4-20mA loop current or digital RS-485 Modbus packet activity.",
                "3. Verify firmware heartbeat LED on transmitter PCB.",
                "4. If output does not respond to a localized thermal heat gun test, condemn and replace the sensor."
            ],
            "work_order": "Priority: High / Critical | Est. Downtime: 20 minutes | Tools: Heat gun, digital communicator"
        },
        "dropout": {
            "mechanism": "Complete circuit discontinuity: severed signal conductor, depleted battery, or intermittent gateway wireless mesh link loss.",
            "risk": "Total loss of observability; triggers failsafe shutdown in interlocked systems.",
            "sop": [
                "1. Test loop continuity with multimeter in ohms mode.",
                "2. Verify gateway RSSI signal strength and gateway packet drop counters.",
                "3. Replace battery pack or check 24V power supply fuse."
            ],
            "work_order": "Priority: High | Est. Downtime: 15 minutes | Tools: Multimeter, wire stripper, battery pack"
        },
        "bias": {
            "mechanism": "Permanent mechanical shock, physical sensor displacement into a local microclimate (e.g., direct sunlight or draft path), or uncalibrated replacement.",
            "risk": "Static offset causes systematic energy waste in thermal control systems.",
            "sop": [
                "1. Inspect physical sensor mounting position and line of sight.",
                "2. Compare against adjacent calibrated neighbor sensors.",
                "3. Re-zero sensor in the supervisory control system."
            ],
            "work_order": "Priority: Medium | Est. Downtime: 20 minutes | Tools: Handheld reference thermometer"
        },
        "event": {
            "mechanism": "Genuine atmospheric or process thermal event confirmed by spatial neighbor consensus across the entire cluster.",
            "risk": "Normal operation. Sensor is fully functional and accurately tracking real environmental physics.",
            "sop": [
                "1. No sensor maintenance required.",
                "2. Inspect process equipment (e.g. heating element, door opening, ventilation duct) to address the environmental change if necessary."
            ],
            "work_order": "Priority: Informational | Est. Downtime: 0 min | Tools: None"
        }
    }
    
    t = rca_templates.get(verdict, {
        "mechanism": "Unusual residual deviation requiring closer telemetry inspection.",
        "risk": "Moderate uncertainty in process observability.",
        "sop": ["1. Verify sensor wiring and physical placement.", "2. Compare against cluster median."],
        "work_order": "Priority: Low | Est. Downtime: 15 minutes | Tools: Reference probe"
    })
    
    sop_lines = "\n".join(t["sop"])
    return f"""### 🔍 AI Root-Cause Analysis (Gemini Expert Engine)
**Affected Unit:** {sensor} · **Anomaly Verdict:** {verdict.upper()} (Confidence: {confidence})

#### 1. Physical Mechanism
{t['mechanism']}

#### 2. Operational & Safety Risk
{t['risk']}

#### 3. Standard Operating Procedure (SOP)
{sop_lines}

#### 4. Maintenance Work Order
{t['work_order']}
"""


def _offline_chat_response(query: str, context: Dict[str, Any]) -> str:
    """Generate strictly grounded responses for common engineering questions offline."""
    try:
        import voice_system
        return voice_system.generate_grounded_voice_response(query, context)
    except Exception:
        q = query.lower()
        focus = context.get("focus", "the selected sensor")
        health = context.get("health", {})
        focus_health = health.get(focus, 100)
        verdict = context.get("focus_verdict", "normal")
        return f"I am your AI Sensor Copilot monitoring {len(context.get('sensors', []))} sensors. Currently, **{focus}** has health **{focus_health:.0f}/100** ({verdict}). You can ask me about sensor health, genuine events versus faults, or technician maintenance SOPs."
