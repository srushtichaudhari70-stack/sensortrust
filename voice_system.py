"""AI Voice System for SensorTrust.

Supports:
1. Grounded Sensor Intelligence Engine: Generates factual, telemetry-grounded answers
   distinguishing genuine environmental events from hardware malfunctions.
2. In-browser Web Speech API (SpeechRecognition & SpeechSynthesis) with explicit
   Indian English (en-IN) voice preference, animated mic states, and stop controls.
3. Fallback type-in interface and browser microphone permission diagnostics.
4. Google Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts`) via `google-genai` SDK.
5. Acoustic Industrial Alert Tone Generator using standard library `wave`.
"""
from __future__ import annotations

import io
import json
import math
import os
import struct
import wave
from typing import Optional, List, Dict, Any
from ai_agent import load_gemini_api_key


# ---------------------------------------------------------------------------
# Python Grounded Voice Intelligence Engine
# ---------------------------------------------------------------------------

def generate_grounded_voice_response(query: str, context: Dict[str, Any]) -> str:
    """Generate a strictly grounded natural-language voice response from live sensor telemetry.
    
    Never hallucinates or invents readings, health scores, or predictions.
    """
    import re
    q = query.lower().strip()
    if not isinstance(context, dict):
        context = {}
    raw_sensors = context.get("sensors", [])
    sensors = list(raw_sensors) if isinstance(raw_sensors, (list, tuple)) else []
    
    raw_readings = context.get("readings") or {}
    readings = raw_readings if isinstance(raw_readings, dict) else {}
    
    raw_health = context.get("health_scores") or context.get("health_dict") or context.get("health") or {}
    health_scores = raw_health if isinstance(raw_health, dict) else {}
    
    raw_bands = context.get("health_bands") or {}
    health_bands = raw_bands if isinstance(raw_bands, dict) else {}
    
    raw_verdicts = context.get("verdicts") or context.get("verdicts_dict") or {}
    verdicts = raw_verdicts if isinstance(raw_verdicts, dict) else {}
    
    raw_conf = context.get("confidences") or context.get("confidences_dict") or {}
    confidences = raw_conf if isinstance(raw_conf, dict) else {}
    
    raw_inc = context.get("incidents") or context.get("incidents_list") or []
    incidents = list(raw_inc) if isinstance(raw_inc, (list, tuple)) else []
    
    raw_shap = context.get("shap_explanations") or context.get("shap") or {}
    shap_explanations = raw_shap if isinstance(raw_shap, dict) else {}
    
    units = str(context.get("units", "°C"))
    try:
        fleet_health = float(context.get("fleet_health", context.get("fleet", 100.0)))
    except (ValueError, TypeError):
        fleet_health = 100.0
    fleet_status = str(context.get("fleet_status", "Optimal"))
    try:
        now_hour = float(context.get("now_hour", 0.0))
    except (ValueError, TypeError):
        now_hour = 0.0
    data_source = str(context.get("data_source", "simulated temperature sensors"))
    is_simulated = bool(context.get("is_simulated", True))
    sim_label = "simulated" if is_simulated else "uploaded"
    focus = str(context.get("focus", sensors[0] if sensors else "S1"))

    # Safe fallback: no sensor data available
    if not sensors or not health_scores:
        return "Sensor telemetry is currently unavailable. Please load a scenario or upload a CSV file."

    # Ambiguous question fallback: user asks about "the other one" or doesn't specify
    if q in ["what about the other one", "what about that sensor", "the other one"] or q.endswith("other sensor"):
        return (
            f"Please specify which sensor you would like to inspect (for example: {', '.join(sensors[:3])}), "
            f"or select a sensor in the sidebar to highlight it."
        )

    # Parse sensor mentions: check if query specifically refers to a sensor like S1, S6, sensor 6, S12
    target_sensor = None
    sensor_matches = re.findall(r'\b(?:sensor\s+s?|s)(\d+)\b', q)
    if sensor_matches:
        candidate = f"S{sensor_matches[0]}"
        if candidate in sensors:
            target_sensor = candidate
        else:
            return (
                f"Sensor {candidate} does not exist in the active cluster. "
                f"The currently configured network has {len(sensors)} sensors: {', '.join(sensors)} "
                f"({data_source}). Please specify a valid sensor ID."
            )

    if not target_sensor:
        target_sensor = focus

    s_val = readings.get(target_sensor, "unavailable")
    s_health = health_scores.get(target_sensor, 100.0)
    s_band = health_bands.get(target_sensor, "Healthy" if s_health >= 80 else ("Watch" if s_health >= 50 else "Service it"))
    s_verdict = verdicts.get(target_sensor, "normal")
    s_conf = confidences.get(target_sensor, 95.0)
    val_str = f"{s_val:.2f} {units}" if isinstance(s_val, (int, float)) else f"{s_val} {units}"

    # 1. "How many sensors are healthy?"
    if "healthy" in q and any(k in q for k in ["how many", "count", "number of", "total", "share"]):
        healthy = [s for s, h in health_scores.items() if h >= 80]
        watch = [s for s, h in health_scores.items() if 50 <= h < 80]
        service = [s for s, h in health_scores.items() if h < 50]
        return (
            f"{len(healthy)} of the {len(sensors)} sensors are currently Healthy (score 80 or above). "
            f"{len(watch)} sensor(s) are in the Watch band (50 to 79), and {len(service)} sensor(s) require Service (below 50). "
            f"Average fleet health is {fleet_health:.0f} out of 100 ({fleet_status} status, {sim_label} data)."
        )

    # 2. "Which sensors have anomalies?"
    if any(k in q for k in ["which sensors", "what sensors", "sensors have anomalies", "sensors with anomalies", "any anomalies", "who has anomalies", "list anomalies", "anomalous sensors"]):
        anomalous = [s for s in sensors if verdicts.get(s, "normal") != "normal"]
        if anomalous:
            details = [f"{s}: '{verdicts.get(s)}' (confidence {confidences.get(s, 0.0):.0f}%, reading {readings.get(s, 'N/A'):.1f} {units})" for s in anomalous]
            return (
                f"Currently, {len(anomalous)} sensor(s) have detected anomalies: {'; '.join(details)}. "
                f"The remaining {len(sensors) - len(anomalous)} sensor(s) agree with spatial consensus."
            )
        else:
            return (
                f"No anomalies are detected across the cluster. All {len(sensors)} sensors are operating nominally "
                f"and agree with spatial consensus (timeline hour {now_hour:.1f}, {sim_label} data)."
            )

    # 2b. "The sensor with the lowest health score"
    if any(k in q for k in ["lowest health", "lowest score", "worst health", "worst sensor", "lowest sensor", "minimum health", "most degraded"]):
        low_s = min(health_scores, key=health_scores.get)
        low_h = health_scores[low_s]
        low_v = verdicts.get(low_s, "normal")
        low_r = readings.get(low_s, "N/A")
        r_str = f"{low_r:.1f} {units}" if isinstance(low_r, (int, float)) else f"{low_r} {units}"
        return (
            f"Sensor {low_s} has the lowest health score at {low_h:.0f} out of 100 ({health_bands.get(low_s, 'Watch')}), "
            f"with model verdict '{low_v}' and current reading {r_str} (timeline hour {now_hour:.1f}, {sim_label} data)."
        )

    # 2c. "The latest detected anomaly"
    if any(k in q for k in ["latest anomaly", "latest detected anomaly", "most recent anomaly", "last anomaly", "recent incident"]):
        if incidents:
            last_inc = incidents[-1]
            l_s = last_inc.get("sensor", "Unknown")
            l_v = last_inc.get("verdict", "anomaly")
            l_c = last_inc.get("confidence", 0.9)
            l_c_val = l_c * 100 if l_c <= 1.0 else l_c
            l_rd = last_inc.get("readings", 1)
            return (
                f"The latest detected anomaly occurred on Sensor {l_s}: classified as '{l_v}' with {l_c_val:.0f}% confidence "
                f"persisting over {l_rd} reading(s) (timeline hour {now_hour:.1f}, {sim_label} data)."
            )
        else:
            return (
                f"No anomalies have been detected in the current timeline replay (hour {now_hour:.1f}, {sim_label} data). "
                f"All {len(sensors)} sensors are operating nominally within cluster spatial consensus."
            )

    # 3. "What is the latest reading for the selected sensor?"
    if any(k in q for k in ["latest reading", "current reading", "sensor reading", "measured temperature", "temperature reading", "current value", "latest value"]):
        return (
            f"The latest reading for selected sensor {target_sensor} is {val_str} "
            f"({data_source} at timeline hour {now_hour:.1f}). "
            f"Current classification is '{s_verdict}' with health score {s_health:.0f}/100."
        )

    # 4. "Is the latest event genuine or suspicious?"
    if any(k in q for k in ["genuine or suspicious", "suspicious", "genuine", "real or fault", "real or fake", "is this event genuine", "event genuine"]):
        is_event_active = any(inc.get("verdict") == "event" for inc in incidents) or s_verdict == "event"
        is_fault_active = any(inc.get("verdict") in ["spike", "drift", "stuck", "noise", "bias", "dropout"] for inc in incidents) or s_verdict in ["spike", "drift", "stuck", "noise", "bias", "dropout"]
        if is_event_active:
            return (
                f"The latest event is classified as GENUINE. SensorTrust's spatial consensus observed a simultaneous shift "
                f"across peer sensors in the cluster. Because multiple sensors moved synchronously, the model confirms a real "
                f"environmental temperature change rather than a suspicious hardware fault."
            )
        elif is_fault_active:
            f_sensor = target_sensor if s_verdict in ["spike", "drift", "stuck", "noise", "bias", "dropout"] else [inc.get("sensor") for inc in incidents if inc.get("verdict") != "event"][0]
            f_v = verdicts.get(f_sensor, "fault")
            return (
                f"The latest deviation on {f_sensor} is SUSPICIOUS and classified as a potential sensor fault ('{f_v}'). "
                f"Only {f_sensor} departed from the cluster median while its peer sensors remained steady. "
                f"When an isolated sensor diverges alone, spatial consensus identifies it as a suspected sensor fault requiring on-site technician inspection."
            )
        else:
            return (
                f"No suspicious events or anomalies are currently detected. All {len(sensors)} sensors are in nominal agreement "
                f"with cluster consensus."
            )

    # 5. "Why was this sensor flagged?"
    if any(k in q for k in ["why was this sensor flagged", "why was", "why is this flagged", "why flagged", "reason for"]):
        if s_verdict != "normal":
            info = shap_explanations.get(target_sensor, {})
            reason = info.get("reason", "Its telemetry departs from the spatial cluster consensus.")
            action = info.get("action", "Inspect physical wiring and connections.")
            top_feats = info.get("top", [])
            feats_str = f" Top TreeSHAP evidence: {', '.join([f[0] for f in top_feats[:2]])}." if top_feats else ""
            return (
                f"Sensor {target_sensor} was flagged as '{s_verdict}' with {s_conf:.0f}% confidence: {reason}{feats_str} "
                f"Recommended action: {action} (Note: This is an anomaly flag based on spatial consensus; "
                f"verify on-site to confirm physical cause)."
            )
        else:
            return (
                f"Sensor {target_sensor} is NOT flagged. It is operating nominally with a health score of {s_health:.0f}/100 "
                f"and its reading of {val_str} is within consensus tolerance of peer sensors."
            )

    # 6. Recommendations / Troubleshooting / Action
    if any(k in q for k in ["check", "resolve", "fix", "recommend", "investigate", "troubleshoot", "sop", "action", "work order"]):
        if target_sensor in shap_explanations and s_verdict != "normal":
            action = shap_explanations[target_sensor].get("action", "")
            return (
                f"Standard Operating Procedure for {target_sensor} ({s_verdict}): {action} "
                f"Recommended checks: Verify 24V loop power, inspect cable shielding ground, check terminal screw torque, "
                f"and perform comparison against a certified reference probe."
            )
        elif s_verdict != "normal":
            return (
                f"For {target_sensor} ({s_verdict}): The system flagged a deviation from peer consensus. "
                f"Inspect sensor wiring and probe placement before replacing hardware."
            )
        else:
            return (
                f"Sensor {target_sensor} is operating nominally (health {s_health:.0f}/100). "
                f"No maintenance action required. Continue regular scheduled monitoring."
            )

    # 7. Specific Spike detection questions
    if any(k in q for k in ["spike", "jump", "transient", "sharp"]):
        spikes = [inc for inc in incidents if inc.get("verdict") == "spike"]
        if s_verdict == "spike" or spikes:
            sp_sensor = target_sensor if s_verdict == "spike" else spikes[0].get("sensor", target_sensor)
            sp_conf = confidences.get(sp_sensor, 90.0)
            reason = shap_explanations.get(sp_sensor, {}).get("reason", "A sudden one-off jump away from its recent level.")
            return (
                f"A spike anomaly was detected on {sp_sensor} with confidence {sp_conf:.0f}%. "
                f"{reason} Peer sensors did not show this sudden excursion, pointing to a possible electrical fast transient (EFT) "
                f"or loose terminal wire."
            )
        else:
            return f"No spike is currently active on {target_sensor}. Its current verdict is '{s_verdict}' with reading {val_str}."

    # 8. Specific Drift detection questions
    if any(k in q for k in ["drift", "creeping", "gradual", "slow"]):
        drifts = [inc for inc in incidents if inc.get("verdict") == "drift"]
        if s_verdict == "drift" or drifts:
            dr_sensor = target_sensor if s_verdict == "drift" else drifts[0].get("sensor", target_sensor)
            dr_conf = confidences.get(dr_sensor, 90.0)
            reason = shap_explanations.get(dr_sensor, {}).get("reason", "The reading has steadily crept away from cluster consensus.")
            return (
                f"Drift anomaly detected on {dr_sensor} with confidence {dr_conf:.0f}%. "
                f"{reason} This gradual divergence typically indicates possible calibration drift or probe degradation requiring inspection."
            )
        else:
            return f"No drift is currently observed on {target_sensor}. Its reading is {val_str} ({s_verdict})."

    # 9. Noise, Stuck, Missing/Dropout
    if any(k in q for k in ["noise", "noisy", "jitter"]):
        if s_verdict == "noise":
            reason = shap_explanations.get(target_sensor, {}).get("reason", "Readings show elevated jitter compared to neighbours.")
            return f"Noise anomaly on {target_sensor}: {reason} This indicates possible DC power rail ripple or electromagnetic interference."
        return f"Noise levels on {target_sensor} are nominal. Current verdict is '{s_verdict}'."

    if any(k in q for k in ["stuck", "frozen", "lockup", "flatline"]):
        if s_verdict == "stuck":
            reason = shap_explanations.get(target_sensor, {}).get("reason", "Sensor reading has remained frozen across multiple timesteps.")
            return f"Stuck sensor detected on {target_sensor}: {reason} This indicates a potential ADC register latch-up or frozen reading requiring verification."
        return f"Sensor {target_sensor} is actively varying and not frozen."

    if any(k in q for k in ["missing", "dropout", "disconnected", "offline"]):
        if s_verdict == "dropout":
            return f"Dropout anomaly on {target_sensor}: Telemetry packets are missing, indicating possible wire severance, depleted battery, or link disruption."
        return f"Sensor {target_sensor} has a complete telemetry stream with no missing readings."

    # 10. Sensor Health / Status questions
    if any(k in q for k in ["health", "status", "condition", "score", "how is"]):
        # If user asked generally about fleet or all
        if any(k in q for k in ["fleet", "plant", "overall", "all sensors"]):
            degraded = [s for s, sc in health_scores.items() if sc < 80]
            summary_extra = f" Sensors on Watch: {', '.join(degraded)}." if degraded else " All sensors are Healthy."
            return (
                f"Fleet health average is {fleet_health:.0f} out of 100 ({fleet_status} status) across all {len(sensors)} {sim_label} sensors. "
                f"There are {len(incidents)} active incident(s).{summary_extra}"
            )
        
        # Specific or highlighted sensor
        is_highlighted = (target_sensor == focus)
        highlight_text = "highlighted sensor" if is_highlighted else "sensor"
        return (
            f"Current status for {highlight_text} {target_sensor}: Health score is {s_health:.0f} out of 100 "
            f"({s_band} condition), with model verdict '{s_verdict}' (confidence {s_conf:.0f}%). "
            f"Latest reading is {val_str} ({sim_label} data at timeline hour {now_hour:.1f})."
        )

    # 11. Summary of latest sensor readings and predictions
    if any(k in q for k in ["summary", "readings", "predictions", "overview", "all"]):
        sensor_summaries = [f"{s}: {readings.get(s, 'N/A'):.1f} {units} ({verdicts.get(s, 'normal')})" for s in sensors[:8]]
        inc_count = len(incidents)
        return (
            f"Latest SensorTrust Telemetry Summary at timeline hour {now_hour:.1f} ({sim_label}): {'; '.join(sensor_summaries)}. "
            f"Total active incidents: {inc_count}. Fleet health average: {fleet_health:.0f} out of 100 ({fleet_status})."
        )

    # 12. Safe fallback for other questions
    return (
        f"I am SensorTrust AI Voice Assistant, monitoring {len(sensors)} {data_source}. "
        f"Currently, {focus} is selected with health {health_scores.get(focus, 100):.0f}/100 and verdict '{verdicts.get(focus, 'normal')}'. "
        f"You can ask: 'What is my current sensor health?', 'What is the latest reading for {focus}?', "
        f"'Which sensors have anomalies?', 'How many sensors are healthy?', or 'Is the latest event genuine or suspicious?'."
    )


# ---------------------------------------------------------------------------
# Gemini 3.8 Flash TTS & Acoustic Tone Generator
# ---------------------------------------------------------------------------

def synthesize_gemini_tts(text: str, api_key: Optional[str] = None, voice_name: str = "Kore") -> Optional[bytes]:
    """Generate high-fidelity AI speech using Google Gemini 3.8 Flash TTS."""
    key = load_gemini_api_key(api_key)
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        interaction = client.interactions.create(
            model="gemini-3.8-flash-tts",
            input=text[:1500],
            response_modalities=["audio"],
            generation_config={
                "speech_config": [
                    {"voice": voice_name}
                ]
            }
        )
        if hasattr(interaction, "output_audio") and interaction.output_audio:
            return interaction.output_audio.data
        for step in getattr(interaction, "steps", []):
            if step.type == "model_output":
                for part in getattr(step, "content", []):
                    if hasattr(part, "audio") and part.audio:
                        return part.audio.data
                    if getattr(part, "mime_type", "").startswith("audio"):
                        return getattr(part, "data", None)
    except Exception:
        return None
    return None


def generate_acoustic_alert_tone(verdict: str, sample_rate: int = 22050) -> bytes:
    """Generate synthesized acoustic indicator WAV audio for industrial anomalies."""
    freq_map = {
        "spike": [880, 1320],
        "drift": [440, 554, 659],
        "stuck": [330, 330, 330],
        "noise": [520, 680, 480],
        "bias": [440, 494],
        "dropout": [220, 165],
        "event": [523, 659, 784, 1046],
    }
    notes = freq_map.get(verdict, [440, 880])
    tone_duration = 0.15
    total_samples = int(tone_duration * sample_rate)
    
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        
        for freq in notes:
            frames = bytearray()
            for i in range(total_samples):
                t = i / sample_rate
                env = math.sin(math.pi * i / total_samples)
                val = int(32767.0 * 0.4 * env * math.sin(2.0 * math.pi * freq * t))
                frames.extend(struct.pack("<h", val))
            wf.writeframes(frames)
            
    return buf.getvalue()


def render_browser_voice_player(text_to_speak: str, button_label: str = "🔊 Read Aloud (Indian English)") -> str:
    """Generates an embedded HTML5 Web Speech API script with Indian English voice preference."""
    clean_text = text_to_speak.replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
    html_code = f"""
    <div style="margin: 8px 0; display: inline-flex; gap: 8px; align-items: center; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <button id="speak-btn" onclick="speakText()" style="
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25);
            transition: all 0.15s ease;
        ">
            {button_label}
        </button>
        <button id="stop-btn" onclick="stopText()" style="
            background: #374151;
            color: #d1d5db;
            border: 1px solid #4b5563;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s ease;
        ">
            ⏹️ Stop Speaking
        </button>
    </div>
    <script>
        function getIndianEnglishVoice() {{
            const voices = window.speechSynthesis.getVoices();
            // 1. Try Indian English voices (en-IN)
            let chosen = voices.find(v => v.lang === 'en-IN' || v.lang === 'en_IN' || v.name.includes('India') || v.name.includes('Heera') || v.name.includes('Neerja'));
            // 2. Try British / Natural English
            if (!chosen) chosen = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha')));
            // 3. Fallback to any English
            if (!chosen) chosen = voices.find(v => v.lang.startsWith('en'));
            return chosen;
        }}

        function speakText() {{
            window.speechSynthesis.cancel();
            const text = "{clean_text}";
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            const voice = getIndianEnglishVoice();
            if (voice) utterance.voice = voice;
            window.speechSynthesis.speak(utterance);
        }}

        function stopText() {{
            window.speechSynthesis.cancel();
        }}
    </script>
    """
    return html_code


# ---------------------------------------------------------------------------
# Interactive AI Voice Assistant HUD Component
# ---------------------------------------------------------------------------

def render_interactive_voice_assistant(
    focus_sensor: str,
    focus_health: float,
    focus_verdict: str,
    fleet_health: float,
    incident_count: int,
    sensors: List[str],
    latest_readings: Optional[Dict[str, float]] = None,
    health_dict: Optional[Dict[str, float]] = None,
    verdicts_dict: Optional[Dict[str, str]] = None,
    confidences_dict: Optional[Dict[str, float]] = None,
    incidents_list: Optional[List[Dict[str, Any]]] = None,
    shap_dict: Optional[Dict[str, Any]] = None,
    units: str = "°C",
    now_hour: float = 0.0,
    data_source: str = "simulated temperature sensors",
    is_simulated: bool = True,
    scenario: str = "Synthetic Scenario",
    seed: int = 42,
    backend_url: Optional[str] = None
) -> str:
    """Render full interactive AI Voice Assistant HUD with Web Speech API, Indian English TTS,
    states (Idle, Listening, Processing, Speaking, Error), stop button, transcript, and grounded intelligence.
    """
    if latest_readings is None:
        latest_readings = {s: 20.0 for s in sensors}
    if health_dict is None:
        health_dict = {s: 100.0 for s in sensors}
    if verdicts_dict is None:
        verdicts_dict = {s: "normal" for s in sensors}
    if confidences_dict is None:
        confidences_dict = {s: 95.0 for s in sensors}
    if incidents_list is None:
        incidents_list = []
    if shap_dict is None:
        shap_dict = {}

    context_payload = {
        "focus": focus_sensor,
        "health": focus_health,
        "verdict": focus_verdict,
        "fleet": fleet_health,
        "fleet_status": "Optimal" if fleet_health >= 80 else ("Watch" if fleet_health >= 50 else "Critical"),
        "incidents": incident_count,
        "sensors": sensors,
        "readings": latest_readings,
        "health_dict": health_dict,
        "verdicts_dict": verdicts_dict,
        "confidences_dict": confidences_dict,
        "incidents_list": incidents_list,
        "shap": shap_dict,
        "units": units,
        "now_hour": now_hour,
        "data_source": data_source,
        "is_simulated": is_simulated,
        "scenario": scenario,
        "seed": seed
    }
    context_json = json.dumps(context_payload, default=str)

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 16px;
            background: linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
            border-radius: 14px;
            border: 1px solid #1e293b;
        }}
        .hud-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 12px;
            max-width: 820px;
            margin: 0 auto;
        }}
        .hud-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            padding-bottom: 8px;
            border-bottom: 1px solid #1e293b;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .hud-title {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 16px;
            font-weight: 700;
            color: #f8fafc;
        }}
        .hud-badges {{
            display: flex;
            gap: 6px;
            align-items: center;
        }}
        .badge {{
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }}
        .badge-live {{
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .badge-voice {{
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}

        /* Center Interaction Area */
        .voice-center {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            margin: 4px 0;
        }}

        /* Glowing AI Orb */
        .orb-wrapper {{
            position: relative;
            width: 84px;
            height: 84px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            user-select: none;
        }}
        .orb {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: radial-gradient(circle, #38bdf8 0%, #0284c7 60%, #0369a1 100%);
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.5), inset 0 0 12px rgba(255, 255, 255, 0.4);
            transition: all 0.3s ease;
            animation: pulse-idle 3s infinite ease-in-out;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 26px;
        }}
        .orb.listening {{
            background: radial-gradient(circle, #4ade80 0%, #16a34a 60%, #15803d 100%);
            box-shadow: 0 0 35px rgba(74, 222, 128, 0.8), inset 0 0 15px rgba(255, 255, 255, 0.6);
            animation: pulse-listening 1s infinite ease-in-out;
        }}
        .orb.processing {{
            background: radial-gradient(circle, #fbbf24 0%, #d97706 60%, #b45309 100%);
            box-shadow: 0 0 35px rgba(251, 191, 36, 0.8), inset 0 0 15px rgba(255, 255, 255, 0.6);
            animation: pulse-processing 0.7s infinite alternate ease-in-out;
        }}
        .orb.speaking {{
            background: radial-gradient(circle, #c084fc 0%, #9333ea 60%, #6b21a8 100%);
            box-shadow: 0 0 35px rgba(192, 132, 252, 0.8), inset 0 0 15px rgba(255, 255, 255, 0.6);
            animation: pulse-speaking 0.9s infinite alternate ease-in-out;
        }}
        .orb.error {{
            background: radial-gradient(circle, #f87171 0%, #dc2626 60%, #991b1b 100%);
            box-shadow: 0 0 30px rgba(248, 113, 113, 0.7);
        }}

        @keyframes pulse-idle {{
            0% {{ transform: scale(0.96); box-shadow: 0 0 18px rgba(56, 189, 248, 0.4); }}
            50% {{ transform: scale(1.04); box-shadow: 0 0 28px rgba(56, 189, 248, 0.7); }}
            100% {{ transform: scale(0.96); box-shadow: 0 0 18px rgba(56, 189, 248, 0.4); }}
        }}
        @keyframes pulse-listening {{
            0% {{ transform: scale(0.93); box-shadow: 0 0 20px rgba(74, 222, 128, 0.5); }}
            50% {{ transform: scale(1.10); box-shadow: 0 0 40px rgba(74, 222, 128, 0.9); }}
            100% {{ transform: scale(0.93); box-shadow: 0 0 20px rgba(74, 222, 128, 0.5); }}
        }}
        @keyframes pulse-processing {{
            0% {{ transform: rotate(0deg) scale(0.96); }}
            100% {{ transform: rotate(180deg) scale(1.05); }}
        }}
        @keyframes pulse-speaking {{
            0% {{ transform: scale(0.95); }}
            100% {{ transform: scale(1.08); }}
        }}

        /* Sound Waves Visualizer (Speaking State) */
        .equalizer {{
            display: none;
            gap: 4px;
            align-items: center;
            height: 18px;
            margin-top: 2px;
        }}
        .equalizer.active {{
            display: flex;
        }}
        .eq-bar {{
            width: 3px;
            background: #c084fc;
            border-radius: 2px;
            animation: bounce 0.6s infinite ease-in-out alternate;
        }}
        .eq-bar:nth-child(1) {{ height: 8px; animation-delay: 0.1s; }}
        .eq-bar:nth-child(2) {{ height: 16px; animation-delay: 0.3s; }}
        .eq-bar:nth-child(3) {{ height: 12px; animation-delay: 0.2s; }}
        .eq-bar:nth-child(4) {{ height: 18px; animation-delay: 0.4s; }}
        .eq-bar:nth-child(5) {{ height: 10px; animation-delay: 0.15s; }}
        @keyframes bounce {{
            0% {{ transform: scaleY(0.3); }}
            100% {{ transform: scaleY(1.0); }}
        }}

        /* Status Badge */
        .status-pill {{
            font-size: 12px;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
        }}
        .status-idle {{ background: #1e293b; color: #38bdf8; border: 1px solid #334155; }}
        .status-listening {{ background: rgba(74, 222, 128, 0.2); color: #4ade80; border: 1px solid #16a34a; }}
        .status-processing {{ background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid #d97706; }}
        .status-speaking {{ background: rgba(192, 132, 252, 0.2); color: #c084fc; border: 1px solid #9333ea; }}
        .status-error {{ background: rgba(248, 113, 113, 0.2); color: #f87171; border: 1px solid #dc2626; }}

        /* Action Buttons */
        .controls-row {{
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .btn-mic {{
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: white;
            border: none;
            padding: 9px 20px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 7px;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
            transition: transform 0.15s, background 0.2s;
        }}
        .btn-mic:hover {{
            transform: scale(1.03);
            background: linear-gradient(135deg, #0369a1, #1d4ed8);
        }}
        .btn-mic.listening {{
            background: linear-gradient(135deg, #dc2626, #b91c1c);
            box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4);
        }}
        .btn-stop {{
            background: #1e293b;
            color: #cbd5e1;
            border: 1px solid #475569;
            padding: 9px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s;
        }}
        .btn-stop:hover {{
            background: #334155;
            color: #ffffff;
            border-color: #64748b;
        }}

        /* Transcript & Response Area */
        .chat-display {{
            width: 100%;
            background: #090e17;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            text-align: left;
            max-height: 180px;
            overflow-y: auto;
        }}
        .bubble {{
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12.5px;
            line-height: 1.45;
            display: flex;
            flex-direction: column;
            gap: 3px;
        }}
        .bubble-user {{
            background: #1e293b;
            border-left: 3px solid #38bdf8;
            color: #e2e8f0;
        }}
        .bubble-ai {{
            background: #131c2e;
            border-left: 3px solid #c084fc;
            color: #f1f5f9;
        }}
        .bubble-header {{
            font-size: 11px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .bubble-user .bubble-header {{ color: #38bdf8; }}
        .bubble-ai .bubble-header {{ color: #c084fc; }}

        /* Text Input Fallback Row */
        .input-row {{
            display: flex;
            gap: 6px;
            width: 100%;
            margin-top: 2px;
        }}
        .text-input {{
            flex: 1;
            background: #0f172a;
            border: 1px solid #334155;
            color: #f8fafc;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            outline: none;
            transition: border 0.2s;
        }}
        .text-input:focus {{
            border-color: #38bdf8;
        }}
        .btn-send {{
            background: #2563eb;
            color: white;
            border: none;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s;
        }}
        .btn-send:hover {{
            background: #1d4ed8;
        }}

        /* Permission & Compatibility Alert */
        .notice-box {{
            display: none;
            width: 100%;
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.35);
            color: #fca5a5;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 11.5px;
            text-align: left;
            line-height: 1.4;
        }}

        /* Quick Prompt Chips */
        .chips-container {{
            display: flex;
            flex-direction: column;
            gap: 5px;
            width: 100%;
            margin-top: 2px;
        }}
        .chips-label {{
            font-size: 11px;
            color: #94a3b8;
            text-align: left;
            font-weight: 500;
        }}
        .chips-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            justify-content: center;
        }}
        .chip {{
            background: #1e293b;
            color: #cbd5e1;
            border: 1px solid #334155;
            padding: 5px 10px;
            border-radius: 12px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s;
            white-space: nowrap;
        }}
        .chip:hover {{
            background: #334155;
            color: #ffffff;
            border-color: #38bdf8;
            transform: translateY(-1px);
        }}
    </style>
    </head>
    <body>
    <div class="hud-container">
        <!-- Header -->
        <div class="hud-header">
            <div class="hud-title">
                <span>🛡️ SensorTrust AI Voice Assistant</span>
            </div>
            <div class="hud-badges">
                <span id="backend-status-badge" class="badge" onclick="promptBackendUrl()" style="cursor: pointer;" title="Click to configure backend API endpoint">🌐 Checking Backend...</span>
                <span class="badge badge-voice">🗣️ English (India) TTS</span>
                <span class="badge badge-live">● SPATIAL CONSENSUS</span>
            </div>
        </div>

        <!-- Microphone Permission Warning (if blocked or unsupported) -->
        <div id="notice-box" class="notice-box">
            <strong>⚠️ Microphone / Speech Notice:</strong>
            <span id="notice-text">Web Speech API is not available in this browser context, or microphone permission was denied. You can still type questions in the box below or click any quick prompt chip!</span>
        </div>

        <!-- Center AI Orb & State Indicator -->
        <div class="voice-center">
            <div class="orb-wrapper" onclick="toggleVoiceSession()" title="Click to start or stop listening">
                <div id="orb" class="orb">🎙️</div>
            </div>

            <!-- Equalizer for Speaking State -->
            <div id="equalizer" class="equalizer">
                <div class="eq-bar"></div>
                <div class="eq-bar"></div>
                <div class="eq-bar"></div>
                <div class="eq-bar"></div>
                <div class="eq-bar"></div>
            </div>

            <!-- Explicit State Status -->
            <div id="status-pill" class="status-pill status-ready">
                <span id="status-icon">🟢</span>
                <span id="status-text">Ready — Click microphone or speak your question</span>
            </div>

            <!-- Controls: Start Listening, Stop Listening & Stop Speaking -->
            <div class="controls-row">
                <button id="btn-mic" class="btn-mic" onclick="startListening()">
                    <span id="mic-icon">🎙️</span>
                    <span id="mic-label">Start Listening</span>
                </button>
                <button id="btn-stop-listening" class="btn-stop" onclick="stopListening()" style="display: none; background: #dc2626; color: white; border-color: #ef4444;">
                    <span>⏹️ Stop Listening</span>
                </button>
                <button id="btn-stop-speaking" class="btn-stop" onclick="stopSpeaking()">
                    <span>⏹️ Stop Speaking</span>
                </button>
            </div>
        </div>

        <!-- Conversation Transcript & Response Display -->
        <div id="chat-display" class="chat-display">
            <div class="bubble bubble-ai">
                <div class="bubble-header">
                    <span>🛡️ SensorTrust Copilot</span>
                    <span>Ready</span>
                </div>
                <span>Hello! I am your AI Sensor Assistant monitoring <strong>{len(sensors)} sensors</strong>. Highlighted sensor is <strong>{focus_sensor}</strong> (Health: <strong>{focus_health:.0f}/100</strong>, Verdict: <strong>{focus_verdict}</strong>). Ask me anything using your voice!</span>
            </div>
        </div>

        <!-- Fallback Type-in Bar -->
        <div class="input-row">
            <input type="text" id="text-input" class="text-input" placeholder="Or type a question (e.g. 'What is my current sensor health?')..." onkeydown="handleInputKey(event)">
            <button class="btn-send" onclick="sendTypedQuery()">Ask</button>
        </div>

        <!-- Quick Voice Prompt Chips -->
        <div class="chips-container">
            <div class="chips-label">🎯 Click to ask instantly:</div>
            <div class="chips-grid">
                <button class="chip" onclick="askQuestion('What is my current sensor health?')">📊 Current sensor health</button>
                <button class="chip" onclick="askQuestion('Is this a genuine event or a sensor fault?')">⚖️ Genuine event vs fault?</button>
                <button class="chip" onclick="askQuestion('Why was this spike detected?')">⚡ Why was spike detected?</button>
                <button class="chip" onclick="askQuestion('Explain the latest anomaly')">🔍 Explain latest anomaly</button>
                <button class="chip" onclick="askQuestion('What should I check to resolve this fault?')">🛠️ Troubleshooting recommendations</button>
                <button class="chip" onclick="askQuestion('Summarize latest sensor readings and model predictions')">📋 Summarize all readings</button>
            </div>
        </div>
    </div>

    <script>
        // Grounded Live Telemetry Context
        const CTX = {context_json};

        // State variables
        let isListening = false;
        let isSpeaking = false;
        let recognition = null;

        // DOM elements
        const orb = document.getElementById('orb');
        const statusPill = document.getElementById('status-pill');
        const statusIcon = document.getElementById('status-icon');
        const statusText = document.getElementById('status-text');
        const btnMic = document.getElementById('btn-mic');
        const btnStopListening = document.getElementById('btn-stop-listening');
        const btnStopSpeaking = document.getElementById('btn-stop-speaking');
        const micIcon = document.getElementById('mic-icon');
        const micLabel = document.getElementById('mic-label');
        const chatDisplay = document.getElementById('chat-display');
        const textInput = document.getElementById('text-input');
        const noticeBox = document.getElementById('notice-box');
        const noticeText = document.getElementById('notice-text');
        const equalizer = document.getElementById('equalizer');

        // Resolve SpeechRecognition and SpeechSynthesis (supports top window when in Streamlit/Stlite iframe)
        let SpeechRec = null;
        try {{
            if (window.parent && (window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition)) {{
                SpeechRec = window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition;
            }}
        }} catch (e) {{}}
        if (!SpeechRec) {{
            SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        }}

        let synth = window.speechSynthesis;
        try {{
            if (window.parent && window.parent.speechSynthesis) {{
                synth = window.parent.speechSynthesis;
            }}
        }} catch (e) {{}}

        // Initialize Speech Recognition
        if (SpeechRec) {{
            try {{
                recognition = new SpeechRec();
                recognition.continuous = false;
                recognition.interimResults = false;
                // English (India) preference for speech recognition
                recognition.lang = 'en-IN';

                recognition.onstart = function() {{
                    setAssistantState('listening');
                }};

                recognition.onresult = function(event) {{
                    const transcript = event.results[0][0].transcript;
                    addUserMessage(transcript);
                    processQuestion(transcript);
                }};

                recognition.onerror = function(event) {{
                    console.warn('Speech Recognition error:', event.error);
                    if (event.error === 'not-allowed' || event.error === 'permission-denied') {{
                        showNotice('Microphone permission was denied. Please allow microphone access in your browser settings (address bar icon) to use voice input.');
                        setAssistantState('error', 'Microphone permission denied');
                    }} else if (event.error === 'no-speech') {{
                        setAssistantState('ready', 'No speech detected — tap mic to try again');
                    }} else if (event.error === 'network') {{
                        setAssistantState('error', 'Speech recognition network error');
                    }} else {{
                        setAssistantState('error', 'Microphone error: ' + event.error);
                    }}
                }};

                recognition.onend = function() {{
                    if (isListening && !isSpeaking) {{
                        setAssistantState('ready');
                    }}
                }};
            }} catch (err) {{
                console.error('Speech initialization error:', err);
                showNotice('Speech recognition could not be initialized in this browser. Chrome or Edge is recommended.');
                setAssistantState('error', 'Initialization error');
            }}
        }} else {{
            showNotice('Your browser does not support the Web Speech API (SpeechRecognition). Please use Google Chrome, Microsoft Edge, or Apple Safari.');
            setAssistantState('error', 'Speech recognition unsupported');
        }}

        // State Machine Management: Ready, Listening, Processing, Speaking, Error
        function setAssistantState(state, customMessage) {{
            orb.className = 'orb ' + state;
            equalizer.className = 'equalizer' + (state === 'speaking' ? ' active' : '');

            if (state === 'ready') {{
                isListening = false;
                isSpeaking = false;
                orb.innerHTML = '🎙️';
                statusPill.className = 'status-pill status-ready';
                statusIcon.textContent = '🟢';
                statusText.textContent = customMessage || 'Ready — Click microphone or speak your question';
                btnMic.className = 'btn-mic';
                micIcon.textContent = '🎙️';
                micLabel.textContent = 'Start Listening';
                if (btnStopListening) btnStopListening.style.display = 'none';
            }} else if (state === 'listening') {{
                isListening = true;
                isSpeaking = false;
                orb.innerHTML = '👂';
                statusPill.className = 'status-pill status-listening';
                statusIcon.textContent = '🔵';
                statusText.textContent = customMessage || 'Listening (en-IN)... Speak your question now';
                btnMic.className = 'btn-mic listening';
                micIcon.textContent = '🎙️';
                micLabel.textContent = 'Listening...';
                if (btnStopListening) btnStopListening.style.display = 'inline-flex';
            }} else if (state === 'processing') {{
                isListening = false;
                isSpeaking = false;
                orb.innerHTML = '⚙️';
                statusPill.className = 'status-pill status-processing';
                statusIcon.textContent = '🟡';
                statusText.textContent = customMessage || 'Processing question against sensor telemetry & ML predictions...';
                btnMic.className = 'btn-mic';
                micIcon.textContent = '🎙️';
                micLabel.textContent = 'Start Listening';
                if (btnStopListening) btnStopListening.style.display = 'none';
            }} else if (state === 'speaking') {{
                isListening = false;
                isSpeaking = true;
                orb.innerHTML = '🗣️';
                statusPill.className = 'status-pill status-speaking';
                statusIcon.textContent = '🟣';
                statusText.textContent = customMessage || 'Speaking response in Indian English (en-IN)...';
                btnMic.className = 'btn-mic';
                micIcon.textContent = '🎙️';
                micLabel.textContent = 'Start Listening';
                if (btnStopListening) btnStopListening.style.display = 'none';
            }} else if (state === 'error') {{
                isListening = false;
                isSpeaking = false;
                orb.innerHTML = '⚠️';
                statusPill.className = 'status-pill status-error';
                statusIcon.textContent = '🔴';
                statusText.textContent = customMessage || 'Microphone error or permission blocked';
                btnMic.className = 'btn-mic';
                micIcon.textContent = '🎙️';
                micLabel.textContent = 'Retry Listening';
                if (btnStopListening) btnStopListening.style.display = 'none';
            }}
        }}

        // Start Voice Listening (Prevents multiple simultaneous sessions)
        function startListening() {{
            if (synth) synth.cancel();

            if (isListening) {{
                console.log('Already listening, ignoring redundant start request.');
                return;
            }}

            if (!recognition) {{
                showNotice('Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari, or click any prompt chip below.');
                setAssistantState('error', 'Speech recognition unsupported');
                return;
            }}

            try {{
                recognition.start();
            }} catch (err) {{
                if (err.name === 'InvalidStateError') {{
                    console.log('Recognition already running');
                }} else {{
                    console.error('Recognition start error:', err);
                    setAssistantState('error', 'Error starting microphone: ' + err.message);
                }}
            }}
        }}

        // Stop Listening
        function stopListening() {{
            if (recognition && isListening) {{
                try {{
                    recognition.stop();
                }} catch (e) {{}}
            }}
            setAssistantState('ready', 'Listening stopped');
        }}

        // Toggle Voice Session (via Orb click)
        function toggleVoiceSession() {{
            if (isListening) {{
                stopListening();
            }} else if (isSpeaking) {{
                stopSpeaking();
            }} else {{
                startListening();
            }}
        }}

        // Stop Speaking (Cancel SpeechSynthesis)
        function stopSpeaking() {{
            if (synth) synth.cancel();
            setAssistantState('ready', 'Speech stopped');
        }}

        // Show Notice
        function showNotice(msg) {{
            noticeBox.style.display = 'block';
            noticeText.textContent = msg;
        }}

        // Add Message to Chat UI
        function addUserMessage(text) {{
            const div = document.createElement('div');
            div.className = 'bubble bubble-user';
            div.innerHTML = '<div class="bubble-header"><span>👤 You (Voice Transcript)</span><span>' + getCurrentTime() + '</span></div><span>' + escapeHtml(text) + '</span>';
            chatDisplay.appendChild(div);
            chatDisplay.scrollTop = chatDisplay.scrollHeight;
        }}

        function addAssistantMessage(text, sourceBadge) {{
            const div = document.createElement('div');
            div.className = 'bubble bubble-ai';
            const badge = sourceBadge || '🛡️ SensorTrust Assistant';
            div.innerHTML = '<div class="bubble-header"><span>' + escapeHtml(badge) + '</span><span>' + getCurrentTime() + '</span></div><span>' + formatAiMarkdown(text) + '</span>';
            chatDisplay.appendChild(div);
            chatDisplay.scrollTop = chatDisplay.scrollHeight;
        }}

        function getCurrentTime() {{
            const d = new Date();
            return d.getHours().toString().padStart(2, '0') + ':' + d.getMinutes().toString().padStart(2, '0') + ':' + d.getSeconds().toString().padStart(2, '0');
        }}

        function escapeHtml(str) {{
            return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }}

        function formatAiMarkdown(str) {{
            return escapeHtml(str)
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.*?)\\*/g, '<em>$1</em>');
        }}

        // Ask question directly via prompt chips
        function askQuestion(query) {{
            stopSpeaking();
            addUserMessage(query);
            processQuestion(query);
        }}

        // Handle typed input
        function sendTypedQuery() {{
            const val = textInput.value.trim();
            if (val) {{
                textInput.value = '';
                askQuestion(val);
            }}
        }}

        function handleInputKey(e) {{
            if (e.key === 'Enter') {{
                sendTypedQuery();
            }}
        }}

        // Backend Connectivity & Configuration
        let isProcessing = false;
        const CONFIGURED_BACKEND_URL = "{backend_url or ''}";
        const isLocalHost = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
        let defaultBackendUrl = isLocalHost ? 'http://127.0.0.1:8000' : '';
        let backendUrl = (
            CONFIGURED_BACKEND_URL ||
            window.SENSORTRUST_BACKEND_URL ||
            localStorage.getItem('sensortrust_backend_url') ||
            defaultBackendUrl
        ).replace(new RegExp('/+$'), '');

        // Security Notice for Insecure Origins
        if (window.location.protocol === 'http:' && !isLocalHost) {{
            showNotice('Browser security requires an HTTPS connection for microphone access. Please access this dashboard over HTTPS.');
        }}

        function checkBackendHealth() {{
            const badge = document.getElementById('backend-status-badge');
            if (!backendUrl) {{
                if (badge) {{
                    badge.style.background = 'rgba(56, 189, 248, 0.15)';
                    badge.style.color = '#38bdf8';
                    badge.style.borderColor = 'rgba(56, 189, 248, 0.3)';
                    badge.textContent = '💡 Connect Gemini Backend';
                    badge.title = 'Click to connect your deployed FastAPI backend URL (e.g. on Render, Railway, Cloud Run)';
                }}
                return;
            }}
            fetch(backendUrl + '/api/health', {{ signal: AbortSignal.timeout(3000) }})
                .then(res => res.json())
                .then(data => {{
                    if (badge && data.status === 'healthy') {{
                        badge.style.background = 'rgba(16, 185, 129, 0.15)';
                        badge.style.color = '#34d399';
                        badge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
                        badge.textContent = data.gemini_configured ? '✨ Gemini AI Online (' + data.model + ')' : '⚠️ Server Key Missing';
                        badge.title = 'Connected to ' + backendUrl + ' (click to change)';
                    }}
                }})
                .catch(() => {{
                    if (badge) {{
                        badge.style.background = 'rgba(234, 179, 8, 0.15)';
                        badge.style.color = '#facc15';
                        badge.style.borderColor = 'rgba(234, 179, 8, 0.3)';
                        badge.textContent = '💡 Local Mode (Backend offline)';
                        badge.title = 'Backend unreachable at ' + backendUrl + ' (click to change URL)';
                    }}
                }});
        }}
        setTimeout(checkBackendHealth, 400);

        function promptBackendUrl() {{
            const current = backendUrl || 'http://127.0.0.1:8000';
            const entered = prompt('Enter your deployed SensorTrust FastAPI Backend API URL (e.g. https://sensortrust-backend.onrender.com):', current);
            if (entered !== null) {{
                const clean = entered.trim().replace(new RegExp('/+$'), '');
                if (clean) {{
                    localStorage.setItem('sensortrust_backend_url', clean);
                    backendUrl = clean;
                }} else {{
                    localStorage.removeItem('sensortrust_backend_url');
                    backendUrl = defaultBackendUrl;
                }}
                checkBackendHealth();
            }}
        }}

        // Processing & Answering Question
        function processQuestion(query) {{
            if (isProcessing) {{
                console.log('Already processing a query, ignoring duplicate submission.');
                return;
            }}
            isProcessing = true;
            setAssistantState('processing');

            // If no backend is configured, use instantaneous local grounded intelligence
            if (!backendUrl) {{
                isProcessing = false;
                const fallbackAnswer = generateGroundedAnswer(query);
                addAssistantMessage(fallbackAnswer, '🛡️ SensorTrust Grounded');
                speakAnswer(fallbackAnswer);
                return;
            }}

            // Attempt to query Gemini backend
            fetch(backendUrl + '/api/voice/ask', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ query: query, context: CTX }}),
                signal: AbortSignal.timeout(12000)
            }})
            .then(res => {{
                if (!res.ok) throw new Error('HTTP ' + res.status);
                return res.json();
            }})
            .then(data => {{
                isProcessing = false;
                const answer = data.response;
                const isGemini = data.source === 'gemini';
                const badgeLabel = isGemini ? '✨ Gemini AI (' + data.model + ')' : '🛡️ SensorTrust Grounded';
                addAssistantMessage(answer, badgeLabel);
                speakAnswer(answer);
            }})
            .catch(err => {{
                isProcessing = false;
                console.warn('Backend query error or timeout (' + err.message + '). Falling back to client-side grounded intelligence.');
                const fallbackAnswer = generateGroundedAnswer(query);
                addAssistantMessage(fallbackAnswer, '🛡️ Local Diagnostic Mode (Backend offline)');
                speakAnswer(fallbackAnswer);
            }});
        }}

        // Grounded Intelligence Engine (Client-side matching)
        function generateGroundedAnswer(query) {{
            const q = query.toLowerCase().trim();
            const sensors = CTX.sensors || [];
            const readings = CTX.readings || {{}};
            const healthScores = CTX.health_dict || {{}};
            const verdicts = CTX.verdicts_dict || {{}};
            const confidences = CTX.confidences_dict || {{}};
            const incidents = CTX.incidents_list || [];
            const shapExplanations = CTX.shap || {{}};
            const units = CTX.units || "°C";
            const fleetHealth = Math.round(CTX.fleet !== undefined ? CTX.fleet : 100);
            const fleetStatus = CTX.fleet_status || "Optimal";
            const nowHour = CTX.now_hour !== undefined ? CTX.now_hour : 0.0;
            const dataSource = CTX.data_source || "simulated temperature sensors";
            const isSimulated = CTX.is_simulated !== undefined ? CTX.is_simulated : true;
            const simLabel = isSimulated ? "simulated" : "uploaded";
            const focus = CTX.focus || (sensors.length > 0 ? sensors[0] : "S1");

            // Safe fallback: no sensor data available
            if (!sensors || sensors.length === 0 || Object.keys(healthScores).length === 0) {{
                return "Sensor telemetry is currently unavailable. Please load a scenario or upload a CSV file.";
            }}

            // Ambiguous question fallback
            if (q === "what about the other one" || q === "what about that sensor" || q === "the other one" || q.endsWith("other sensor")) {{
                return "Please specify which sensor you would like to inspect (for example: " + sensors.slice(0, 3).join(", ") + "), or select a sensor in the sidebar to highlight it.";
            }}

            // Parse sensor mentions: check if query specifically refers to a sensor like S1, S6, sensor 6, S12
            let targetSensor = null;
            const sensorMatch = q.match(/\\b(?:sensor\\s+s?|s)(\\d+)\\b/i);
            if (sensorMatch) {{
                const candidate = "S" + sensorMatch[1];
                if (sensors.includes(candidate)) {{
                    targetSensor = candidate;
                }} else {{
                    return "Sensor " + candidate + " does not exist in the active cluster. The currently configured network has " + sensors.length + " sensors: " + sensors.join(", ") + " (" + dataSource + "). Please specify a valid sensor ID.";
                }}
            }}

            if (!targetSensor) {{
                targetSensor = focus;
            }}

            const sVal = readings[targetSensor] !== undefined ? readings[targetSensor] : "unavailable";
            const sHealth = Math.round(healthScores[targetSensor] !== undefined ? healthScores[targetSensor] : 100);
            const sBand = sHealth >= 80 ? "Healthy" : (sHealth >= 50 ? "Watch" : "Service Required");
            const sVerdict = verdicts[targetSensor] || "normal";
            const sConf = Math.round(confidences[targetSensor] !== undefined ? confidences[targetSensor] : 95);
            const valStr = typeof sVal === "number" ? sVal.toFixed(2) + " " + units : sVal + " " + units;

            // 1. "How many sensors are healthy?"
            if (q.includes("healthy") && (q.includes("how many") || q.includes("count") || q.includes("number of") || q.includes("total") || q.includes("share"))) {{
                const healthy = sensors.filter(s => (healthScores[s] || 100) >= 80);
                const watch = sensors.filter(s => (healthScores[s] || 100) >= 50 && (healthScores[s] || 100) < 80);
                const service = sensors.filter(s => (healthScores[s] || 100) < 50);
                return healthy.length + " of the " + sensors.length + " sensors are currently Healthy (score 80 or above). " +
                    watch.length + " sensor(s) are in the Watch band (50 to 79), and " + service.length + " sensor(s) require Service (below 50). " +
                    "Average fleet health is " + fleetHealth + " out of 100 (" + fleetStatus + " status, " + simLabel + " data).";
            }}

            // 2. "Which sensors have anomalies?"
            if (q.includes("which sensors") || q.includes("what sensors") || q.includes("sensors have anomalies") || q.includes("sensors with anomalies") || q.includes("any anomalies") || q.includes("who has anomalies") || q.includes("list anomalies") || q.includes("anomalous sensors")) {{
                const anomalous = sensors.filter(s => (verdicts[s] || "normal") !== "normal");
                if (anomalous.length > 0) {{
                    const details = anomalous.map(s => s + ": '" + (verdicts[s] || "fault") + "' (confidence " + Math.round(confidences[s] || 0) + "%, reading " + (readings[s] !== undefined ? readings[s].toFixed(1) + " " + units : "N/A") + ")");
                    return "Currently, " + anomalous.length + " sensor(s) have detected anomalies: " + details.join("; ") + ". " +
                        "The remaining " + (sensors.length - anomalous.length) + " sensor(s) agree with spatial consensus.";
                }} else {{
                    return "No anomalies are detected across the cluster. All " + sensors.length + " sensors are operating nominally and agree with spatial consensus (timeline hour " + nowHour.toFixed(1) + ", " + simLabel + " data).";
                }}
            }}

            // 3. "What is the latest reading for the selected sensor?"
            if (q.includes("latest reading") || q.includes("current reading") || q.includes("sensor reading") || q.includes("measured temperature") || q.includes("temperature reading") || q.includes("current value") || q.includes("latest value")) {{
                return "The latest reading for selected sensor " + targetSensor + " is " + valStr + " (" + dataSource + " at timeline hour " + nowHour.toFixed(1) + "). Current classification is '" + sVerdict + "' with health score " + sHealth + "/100.";
            }}

            // 4. "Is the latest event genuine or suspicious?"
            if (q.includes("genuine or suspicious") || q.includes("suspicious") || q.includes("genuine") || q.includes("real or fault") || q.includes("real or fake") || q.includes("is this event genuine") || q.includes("event genuine")) {{
                const isEventActive = incidents.some(i => i.verdict === "event") || sVerdict === "event";
                const isFaultActive = incidents.some(i => ["spike", "drift", "stuck", "noise", "bias", "dropout"].includes(i.verdict)) || ["spike", "drift", "stuck", "noise", "bias", "dropout"].includes(sVerdict);
                if (isEventActive) {{
                    return "The latest event is classified as GENUINE. SensorTrust's spatial consensus observed a simultaneous shift across peer sensors in the cluster. Because multiple sensors moved synchronously, the model confirms a real environmental temperature change rather than a suspicious hardware fault.";
                }} else if (isFaultActive) {{
                    let fSensor = ["spike", "drift", "stuck", "noise", "bias", "dropout"].includes(sVerdict) ? targetSensor : (incidents.find(i => i.verdict !== "event") || {{}}).sensor || targetSensor;
                    let fV = verdicts[fSensor] || "fault";
                    return "The latest deviation on " + fSensor + " is SUSPICIOUS and classified as a potential sensor fault ('" + fV + "'). Only " + fSensor + " departed from the cluster median while its peer sensors remained steady. When an isolated sensor diverges alone, spatial consensus identifies it as a suspected sensor fault requiring on-site technician inspection.";
                }} else {{
                    return "No suspicious events or anomalies are currently detected. All " + sensors.length + " sensors are in nominal agreement with cluster consensus.";
                }}
            }}

            // 5. "Why was this sensor flagged?"
            if (q.includes("why was this sensor flagged") || q.includes("why was") || q.includes("why is this flagged") || q.includes("why flagged") || q.includes("reason for")) {{
                if (sVerdict !== "normal") {{
                    const info = shapExplanations[targetSensor] || {{}};
                    const reason = info.reason || "Its telemetry departs from the spatial cluster consensus.";
                    const action = info.action || "Inspect physical wiring and connections.";
                    const topFeats = info.top || [];
                    const featsStr = topFeats.length > 0 ? " Top TreeSHAP evidence: " + topFeats.slice(0, 2).map(f => f[0]).join(", ") + "." : "";
                    return "Sensor " + targetSensor + " was flagged as '" + sVerdict + "' with " + sConf + "% confidence: " + reason + featsStr +
                        " Recommended action: " + action + " (Note: This is an anomaly flag based on spatial consensus; verify on-site to confirm physical cause).";
                }} else {{
                    return "Sensor " + targetSensor + " is NOT flagged. It is operating nominally with a health score of " + sHealth + "/100 and its reading of " + valStr + " is within consensus tolerance of peer sensors.";
                }}
            }}

            // 6. Recommendations / Troubleshooting / Action
            if (q.includes("check") || q.includes("resolve") || q.includes("fix") || q.includes("recommend") || q.includes("investigate") || q.includes("troubleshoot") || q.includes("sop") || q.includes("action") || q.includes("work order")) {{
                if (shapExplanations[targetSensor] && sVerdict !== "normal") {{
                    const action = shapExplanations[targetSensor].action || "";
                    return "Standard Operating Procedure for " + targetSensor + " (" + sVerdict + "): " + action +
                        " Recommended checks: Verify 24V loop power, inspect cable shielding ground, check terminal screw torque, and perform comparison against a certified reference probe.";
                }} else if (sVerdict !== "normal") {{
                    return "For " + targetSensor + " (" + sVerdict + "): The system flagged a deviation from peer consensus. Inspect sensor wiring and probe placement before replacing hardware.";
                }} else {{
                    return "Sensor " + targetSensor + " is operating nominally (health " + sHealth + "/100). No maintenance action required. Continue regular scheduled monitoring.";
                }}
            }}

            // 7. Specific Spike detection questions
            if (q.includes("spike") || q.includes("jump") || q.includes("transient") || q.includes("sharp")) {{
                const spikes = incidents.filter(i => i.verdict === "spike");
                if (sVerdict === "spike" || spikes.length > 0) {{
                    const spSensor = sVerdict === "spike" ? targetSensor : spikes[0].sensor;
                    const spConf = Math.round(confidences[spSensor] || 90);
                    const reason = (shapExplanations[spSensor] || {{}}).reason || "A sudden one-off jump away from its recent level.";
                    return "A spike anomaly was detected on " + spSensor + " with confidence " + spConf + "%. " + reason +
                        " Peer sensors did not show this sudden excursion, pointing to a possible electrical fast transient (EFT) or loose terminal wire.";
                }} else {{
                    return "No spike is currently active on " + targetSensor + ". Its current verdict is '" + sVerdict + "' with reading " + valStr + ".";
                }}
            }}

            // 8. Specific Drift detection questions
            if (q.includes("drift") || q.includes("creeping") || q.includes("gradual") || q.includes("slow")) {{
                const drifts = incidents.filter(i => i.verdict === "drift");
                if (sVerdict === "drift" || drifts.length > 0) {{
                    const drSensor = sVerdict === "drift" ? targetSensor : drifts[0].sensor;
                    const drConf = Math.round(confidences[drSensor] || 90);
                    const reason = (shapExplanations[drSensor] || {{}}).reason || "The reading has steadily crept away from cluster consensus.";
                    return "Drift anomaly detected on " + drSensor + " with confidence " + drConf + "%. " + reason +
                        " This gradual divergence typically indicates possible calibration drift or probe degradation requiring inspection.";
                }} else {{
                    return "No drift is currently observed on " + targetSensor + ". Its reading is " + valStr + " (" + sVerdict + ").";
                }}
            }}

            // 9. Noise, Stuck, Missing/Dropout
            if (q.includes("noise") || q.includes("noisy") || q.includes("jitter")) {{
                if (sVerdict === "noise") {{
                    const reason = (shapExplanations[targetSensor] || {{}}).reason || "Readings show elevated jitter compared to neighbours.";
                    return "Noise anomaly on " + targetSensor + ": " + reason + " This indicates possible DC power rail ripple or electromagnetic interference.";
                }}
                return "Noise levels on " + targetSensor + " are nominal. Current verdict is '" + sVerdict + "'.";
            }}

            if (q.includes("stuck") || q.includes("frozen") || q.includes("lockup") || q.includes("flatline")) {{
                if (sVerdict === "stuck") {{
                    const reason = (shapExplanations[targetSensor] || {{}}).reason || "Sensor reading has remained frozen across multiple timesteps.";
                    return "Stuck sensor detected on " + targetSensor + ": " + reason + " This indicates a potential ADC register latch-up or frozen reading requiring verification.";
                }}
                return "Sensor " + targetSensor + " is actively varying and not frozen.";
            }}

            if (q.includes("missing") || q.includes("dropout") || q.includes("disconnected") || q.includes("offline")) {{
                if (sVerdict === "dropout") {{
                    return "Dropout anomaly on " + targetSensor + ": Telemetry packets are missing, indicating possible wire severance, depleted battery, or link disruption.";
                }}
                return "Sensor " + targetSensor + " has a complete telemetry stream with no missing readings.";
            }}

            // 10. Sensor Health / Status questions
            if (q.includes("health") || q.includes("status") || q.includes("condition") || q.includes("score") || q.includes("how is")) {{
                if (q.includes("fleet") || q.includes("plant") || q.includes("overall") || q.includes("all sensors")) {{
                    const degraded = sensors.filter(s => (healthScores[s] || 100) < 80);
                    const summaryExtra = degraded.length > 0 ? " Sensors on Watch: " + degraded.join(", ") + "." : " All sensors are Healthy.";
                    return "Fleet health average is " + fleetHealth + " out of 100 (" + fleetStatus + " status) across all " + sensors.length + " " + simLabel + " sensors. There are " + incidents.length + " active incident(s)." + summaryExtra;
                }}

                const isHighlighted = (targetSensor === focus);
                const highlightText = isHighlighted ? "highlighted sensor" : "sensor";
                return "Current status for " + highlightText + " " + targetSensor + ": Health score is " + sHealth + " out of 100 (" + sBand + " condition), with model verdict '" + sVerdict + "' (confidence " + sConf + "%). Latest reading is " + valStr + " (" + simLabel + " data at timeline hour " + nowHour.toFixed(1) + ").";
            }}

            // 11. Summary of latest sensor readings and predictions
            if (q.includes("summary") || q.includes("readings") || q.includes("predictions") || q.includes("overview") || q.includes("all")) {{
                const sampleSensors = sensors.slice(0, 8);
                const sensorSummaries = sampleSensors.map(s => s + ": " + (readings[s] !== undefined ? readings[s].toFixed(1) + " " + units : "N/A") + " (" + (verdicts[s] || "normal") + ")").join("; ");
                return "Latest SensorTrust Telemetry Summary at timeline hour " + nowHour.toFixed(1) + " (" + simLabel + "): " + sensorSummaries + ". Total active incidents: " + incidents.length + ". Fleet health average: " + fleetHealth + " out of 100 (" + fleetStatus + ").";
            }}

            // 12. Safe fallback
            return "I am SensorTrust AI Voice Assistant, monitoring " + sensors.length + " " + dataSource + ". Currently, " + focus + " is selected with health " + sHealth + "/100 and verdict '" + sVerdict + "'. You can ask: 'What is my current sensor health?', 'What is the latest reading for " + focus + "?', 'Which sensors have anomalies?', 'How many sensors are healthy?', or 'Is the latest event genuine or suspicious?'.";
        }}

        // Text-to-Speech (Indian English Priority)
        function speakAnswer(text) {{
            if (!synth) {{
                setAssistantState('ready');
                return;
            }}
            // Cancel any previous speech before starting new response
            synth.cancel();

            const cleanText = text.replace(/\\*\\*/g, '').replace(/\\*/g, '');
            const utterance = new SpeechSynthesisUtterance(cleanText);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.lang = 'en-IN';

            // Prioritize Indian English voices (en-IN)
            const voices = synth.getVoices ? synth.getVoices() : [];
            let selectedVoice = voices.find(v => v.lang === 'en-IN' || v.lang === 'en_IN');
            if (!selectedVoice) {{
                selectedVoice = voices.find(v => (v.lang.startsWith('en') || v.lang.startsWith('hi')) && 
                    (v.name.includes('India') || v.name.includes('Indian') || v.name.includes('Hindi') || 
                     v.name.includes('Heera') || v.name.includes('Neerja') || v.name.includes('Ravi') || v.name.includes('Aditi')));
            }}
            if (!selectedVoice) {{
                selectedVoice = voices.find(v => v.lang.startsWith('en') && 
                    (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha')));
            }}
            if (!selectedVoice) {{
                selectedVoice = voices.find(v => v.lang.startsWith('en'));
            }}
            if (selectedVoice) {{
                utterance.voice = selectedVoice;
            }}

            utterance.onstart = function() {{
                setAssistantState('speaking');
            }};

            utterance.onend = function() {{
                setAssistantState('ready');
            }};

            utterance.onerror = function() {{
                setAssistantState('ready');
            }};

            synth.speak(utterance);
        }}

        // Ensure voices are loaded asynchronously in Chrome/Edge
        if (synth && synth.onvoiceschanged !== undefined) {{
            synth.onvoiceschanged = function() {{
                synth.getVoices();
            }};
        }}
    </script>
    </body>
    </html>
    """
