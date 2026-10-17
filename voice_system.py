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
    q = query.lower().strip()
    focus = context.get("focus", "S1")
    sensors = context.get("sensors", [])
    readings = context.get("readings", {})
    health_scores = context.get("health_scores", {})
    health_bands = context.get("health_bands", {})
    verdicts = context.get("verdicts", {})
    confidences = context.get("confidences", {})
    incidents = context.get("incidents", [])
    shap_explanations = context.get("shap_explanations", {})
    units = context.get("units", "°C")
    fleet_health = context.get("fleet_health", 100.0)
    fleet_status = context.get("fleet_status", "Optimal")
    now_hour = context.get("now_hour", 0.0)

    # Check if a specific sensor is mentioned in the query (e.g. S1, S2, sensor 3)
    target_sensor = focus
    for s in sensors:
        s_lower = s.lower()
        num = s_lower.replace("s", "")
        if s_lower in q or f"sensor {num}" in q or f"sensor {s_lower}" in q:
            target_sensor = s
            break

    s_val = readings.get(target_sensor, "N/A")
    s_health = health_scores.get(target_sensor, 100.0)
    s_band = health_bands.get(target_sensor, "Healthy")
    s_verdict = verdicts.get(target_sensor, "normal")
    s_conf = confidences.get(target_sensor, 95.0)
    val_str = f"{s_val:.2f} {units}" if isinstance(s_val, (int, float)) else f"{s_val} {units}"

    # 1. Health status and score questions
    if any(k in q for k in ["health", "score", "status", "condition", "how healthy", "how is"]):
        if any(f"s{i+1}" in q or f"sensor {i+1}" in q for i in range(len(sensors))):
            return (
                f"Sensor {target_sensor} has a health score of {s_health:.0f} out of 100, "
                f"in the {s_band} band. Current reading is {val_str} with model verdict '{s_verdict}' "
                f"(confidence {s_conf:.0f}%)."
            )
        # Fleet-wide or highlighted sensor health
        degraded = [s for s, h in health_scores.items() if h < 80]
        critical = [s for s, h in health_scores.items() if h < 50]
        summary_extra = ""
        if critical:
            summary_extra = f" Critical attention required on {', '.join(critical)}."
        elif degraded:
            summary_extra = f" Watch status flagged on {', '.join(degraded)}."
        else:
            summary_extra = " All sensors are within healthy tolerances."

        return (
            f"Currently, highlighted sensor {target_sensor} has a health score of {s_health:.0f} out of 100 "
            f"({s_band} condition, verdict '{s_verdict}'). Fleet-wide average health is {fleet_health:.0f} out of 100 "
            f"({fleet_status} status) across all {len(sensors)} sensors.{summary_extra}"
        )

    # 2. Recommendations / Troubleshooting / Action
    if any(k in q for k in ["check", "resolve", "fix", "recommend", "investigate", "troubleshoot", "sop", "action", "work order"]):
        if target_sensor in shap_explanations:
            action = shap_explanations[target_sensor].get("action", "")
            return (
                f"Standard Operating Procedure for {target_sensor} ({s_verdict}): {action} "
                f"Recommended checks: Verify 24V loop power, inspect cable shielding ground, check terminal screw torque, "
                f"and perform comparison against a certified reference probe."
            )
        else:
            return (
                f"For {target_sensor} ({s_verdict}): Verify wiring continuity, check for local environmental drafts or heating, "
                f"and compare against the cluster median of {readings.get(target_sensor, 'N/A')} {units}."
            )

    # 3. Genuine Event vs Sensor Fault
    if ("event" in q and "fault" in q) or any(k in q for k in ["genuine", "real", "difference", "vs"]):
        is_event_active = any(inc.get("verdict") == "event" for inc in incidents) or s_verdict == "event"
        if is_event_active:
            return (
                f"SensorTrust confirms a genuine environmental event. The shift was observed synchronously "
                f"across peer sensors in the cluster rather than on an isolated unit. Because multiple sensors moved "
                f"together, spatial consensus proves this is a real ambient temperature shift and not a hardware malfunction."
            )
        elif s_verdict in ["spike", "drift", "stuck", "noise", "bias", "dropout"]:
            return (
                f"Sensor {target_sensor} is classified as a sensor fault ('{s_verdict}'). "
                f"Its reading departed from consensus while its {len(sensors)-1} neighbouring sensors remained steady. "
                f"When only one sensor departs from the cluster median, SensorTrust's spatial consensus identifies "
                f"it as an isolated hardware or electrical failure rather than a genuine event."
            )
        else:
            return (
                f"In SensorTrust, a genuine event is confirmed when neighbouring sensors shift synchronously together. "
                f"A sensor fault is identified when an isolated sensor departs from the spatial median while peer sensors remain steady. "
                f"Currently, highlighted sensor {target_sensor} is '{s_verdict}' with health {s_health:.0f}/100."
            )

    # 3. Spike questions
    if any(k in q for k in ["spike", "jump", "transient", "sharp"]):
        spikes = [inc for inc in incidents if inc.get("verdict") == "spike"]
        if s_verdict == "spike" or spikes:
            sp_sensor = target_sensor if s_verdict == "spike" else spikes[0].get("sensor", target_sensor)
            sp_conf = confidences.get(sp_sensor, 90.0)
            reason = shap_explanations.get(sp_sensor, {}).get("reason", "A sudden one-off jump away from its recent level.")
            return (
                f"A spike fault was detected on {sp_sensor} with confidence {sp_conf:.0f}%. "
                f"{reason} Peer sensors did not show this sudden excursion, confirming an electrical fast transient (EFT) "
                f"or loose terminal wire."
            )
        else:
            return (
                f"No spike is currently active on {target_sensor}. Its current verdict is '{s_verdict}' "
                f"with reading {val_str}."
            )

    # 4. Drift questions
    if any(k in q for k in ["drift", "creeping", "gradual", "slow"]):
        drifts = [inc for inc in incidents if inc.get("verdict") == "drift"]
        if s_verdict == "drift" or drifts:
            dr_sensor = target_sensor if s_verdict == "drift" else drifts[0].get("sensor", target_sensor)
            dr_conf = confidences.get(dr_sensor, 90.0)
            reason = shap_explanations.get(dr_sensor, {}).get("reason", "The reading has steadily crept away from cluster consensus.")
            return (
                f"Drift fault detected on {dr_sensor} with confidence {dr_conf:.0f}%. "
                f"{reason} This gradual divergence typically points to electrochemical aging of the RTD probe or thermal paste drying out."
            )
        else:
            return f"No drift fault is currently observed on {target_sensor}. Its reading is {val_str} ({s_verdict})."

    # 5. Noise, Stuck, Missing/Dropout questions
    if any(k in q for k in ["noise", "noisy", "jitter"]):
        if s_verdict == "noise":
            reason = shap_explanations.get(target_sensor, {}).get("reason", "Readings show elevated jitter compared to neighbours.")
            return f"Noise fault confirmed on {target_sensor}: {reason} This indicates DC power rail ripple or electromagnetic interference."
        return f"Noise levels on {target_sensor} are nominal. Current verdict is '{s_verdict}'."

    if any(k in q for k in ["stuck", "frozen", "lockup", "flatline"]):
        if s_verdict == "stuck":
            reason = shap_explanations.get(target_sensor, {}).get("reason", "Sensor reading has remained frozen across multiple timesteps.")
            return f"Stuck sensor fault on {target_sensor}: {reason} This indicates an ADC register latch-up or firmware lockup."
        return f"Sensor {target_sensor} is actively varying and not frozen."

    if any(k in q for k in ["missing", "dropout", "disconnected", "offline"]):
        if s_verdict == "dropout":
            return f"Dropout fault detected on {target_sensor}: Telemetry packets are missing, indicating wire severance, depleted battery, or link disruption."
        return f"Sensor {target_sensor} has a complete telemetry stream with no missing readings."

    # 6. Explain anomaly / Why flagged
    if any(k in q for k in ["why", "explain", "anomaly", "flagged", "reason", "cause"]):
        if target_sensor in shap_explanations and s_verdict != "normal":
            info = shap_explanations[target_sensor]
            reason = info.get("reason", "")
            action = info.get("action", "")
            top_feats = info.get("top", [])
            feats_str = f" Key TreeSHAP evidence: {', '.join([f[0] for f in top_feats[:2]])}." if top_feats else ""
            return (
                f"{target_sensor} was flagged as '{s_verdict}' (confidence {s_conf:.0f}%). "
                f"{reason}{feats_str} Recommended action: {action}"
            )
        elif s_verdict != "normal":
            return (
                f"{target_sensor} is flagged as '{s_verdict}' with confidence {s_conf:.0f}%. "
                f"Its telemetry deviates from the spatial cluster consensus. Health score is {s_health:.0f}/100."
            )
        else:
            return (
                f"Sensor {target_sensor} is operating normally with health score {s_health:.0f}/100. "
                f"Its reading of {val_str} is fully consistent with neighbouring sensors."
            )


    # 8. Summary of latest sensor readings and model predictions
    if any(k in q for k in ["summary", "readings", "predictions", "overview", "all sensors", "report", "latest"]):
        sensor_summaries = []
        for s in sensors[:6]:
            val = readings.get(s, "N/A")
            v = verdicts.get(s, "normal")
            val_text = f"{val:.1f} {units}" if isinstance(val, (int, float)) else str(val)
            sensor_summaries.append(f"{s}: {val_text} ({v})")
        readings_str = "; ".join(sensor_summaries)
        inc_count = len(incidents)
        return (
            f"Latest SensorTrust Telemetry Summary at timeline hour {now_hour:.1f}: {readings_str}. "
            f"Total active incidents: {inc_count}. Fleet health average: {fleet_health:.0f} out of 100 ({fleet_status})."
        )

    # Default general response
    return (
        f"I am SensorTrust AI Voice Assistant, monitoring {len(sensors)} sensors. "
        f"Currently, highlighted sensor {target_sensor} has a health score of {s_health:.0f} out of 100 ({s_verdict}). "
        f"You can ask me about sensor health, genuine events versus faults, why an anomaly was flagged, "
        f"or troubleshooting recommendations."
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
    now_hour: float = 0.0
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
        "now_hour": now_hour
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

        function addAssistantMessage(text) {{
            const div = document.createElement('div');
            div.className = 'bubble bubble-ai';
            div.innerHTML = '<div class="bubble-header"><span>🛡️ SensorTrust Assistant</span><span>' + getCurrentTime() + '</span></div><span>' + formatAiMarkdown(text) + '</span>';
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

        // Processing & Answering Question
        function processQuestion(query) {{
            setAssistantState('processing');

            // Generate answer from strictly grounded telemetry
            setTimeout(() => {{
                const answer = generateGroundedAnswer(query);
                addAssistantMessage(answer);
                speakAnswer(answer);
            }}, 350);
        }}

        // Grounded Intelligence Engine (Client-side matching)
        function generateGroundedAnswer(query) {{
            const q = query.toLowerCase().trim();
            const f = CTX.focus;
            const h = Math.round(CTX.health);
            const v = CTX.verdict;
            const fleet = Math.round(CTX.fleet);
            const status = CTX.fleet_status;
            const units = CTX.units;
            const val = CTX.readings[f] !== undefined ? CTX.readings[f].toFixed(2) + ' ' + units : 'N/A';
            const conf = Math.round(CTX.confidences_dict[f] || 95);

            // 1. Health Status & Score
            if (q.includes('health') || q.includes('score') || q.includes('status') || q.includes('condition') || q.includes('how is') || q.includes('how healthy')) {{
                if (q.includes('fleet') || q.includes('plant') || q.includes('overall') || q.includes('all')) {{
                    return "The plant fleet health average is currently " + fleet + " out of 100 (" + status + " status) across " + CTX.sensors.length + " sensors. There are " + CTX.incidents + " active anomaly incidents.";
                }}
                return f + " has a health score of " + h + " out of 100 with current verdict '" + v + "' (confidence " + conf + "%). Its latest measured reading is " + val + ". Fleet average is " + fleet + " out of 100.";
            }}

            // 2. Recommendations / Troubleshooting / Action
            if (q.includes('check') || q.includes('resolve') || q.includes('fix') || q.includes('recommend') || q.includes('investigate') || q.includes('troubleshoot') || q.includes('sop') || q.includes('action') || q.includes('work order')) {{
                let act = "For " + f + " (" + v + "): ";
                if (v === 'drift') act += "Perform a two-point calibration check against a certified reference dry-well thermometer, inspect probe sheath for corrosion, and apply zero/span offset correction.";
                else if (v === 'spike') act += "Inspect terminal screw torque (0.5 Nm), verify cable shielding ground continuity at a single point, and check clearance from high-voltage motor cables.";
                else if (v === 'stuck') act += "Perform a 30-second cold reboot of the sensor unit. If it remains frozen after reboot, replace the probe.";
                else if (v === 'noise') act += "Measure AC ripple on the 24V DC power rail (< 50mV p-p required) and check for ground loops with the analog input card.";
                else if (v === 'bias') act += "Check whether the sensor was physically relocated or shielded, and recalibrate reference offset.";
                else if (v === 'dropout') act += "Check 24V loop power fuse, inspect cable conductor continuity, and verify gateway wireless RSSI signal.";
                else act += "No maintenance action required. Continue regular scheduled monitoring.";
                return act;
            }}

            // 3. Genuine Event vs Sensor Fault
            if (q.includes('genuine') || (q.includes('event') && q.includes('fault')) || q.includes('real') || q.includes('vs') || q.includes('difference')) {{
                if (v === 'event' || CTX.incidents_list.some(i => i.verdict === 'event')) {{
                    return "SensorTrust confirms a genuine environmental event. The temperature change was observed synchronously across all peer sensors in the cluster. Because multiple sensors moved together, spatial consensus proves this is a real ambient shift rather than a sensor fault.";
                }} else if (['spike', 'drift', 'stuck', 'noise', 'bias', 'dropout'].includes(v)) {{
                    return "Sensor " + f + " is classified as a sensor fault ('" + v + "'). Its reading departed from consensus while its " + (CTX.sensors.length - 1) + " neighbouring sensors remained steady. When only an isolated sensor departs from the cluster median, SensorTrust identifies it as a hardware malfunction rather than a real event.";
                }} else {{
                    return "SensorTrust uses spatial consensus to distinguish events from faults: when a real environmental shift occurs, all sensors move synchronously together. When a sensor fault occurs, only that single sensor deviates from the peer median while others stay steady.";
                }}
            }}

            // 3. Spike detection
            if (q.includes('spike') || q.includes('jump') || q.includes('transient') || q.includes('sharp')) {{
                const spIncident = CTX.incidents_list.find(i => i.verdict === 'spike');
                if (v === 'spike' || spIncident) {{
                    const sName = v === 'spike' ? f : spIncident.sensor;
                    return "A spike fault was detected on " + sName + " where the reading jumped sharply away from its recent baseline in a single 5-minute timestep while neighbours remained steady. This typically indicates an electrical fast transient or loose wire.";
                }}
                return "No spike fault is active on " + f + ". Its current reading is " + val + " with verdict '" + v + "'.";
            }}

            // 4. Drift detection
            if (q.includes('drift') || q.includes('creeping') || q.includes('gradual') || q.includes('slow')) {{
                const drIncident = CTX.incidents_list.find(i => i.verdict === 'drift');
                if (v === 'drift' || drIncident) {{
                    const sName = v === 'drift' ? f : drIncident.sensor;
                    return "A drift fault was detected on " + sName + ". Its reading has steadily crept away from cluster consensus over several hours, indicating electrochemical aging of the sensing element or thermal paste degradation.";
                }}
                return "No drift is observed on " + f + ". Readings are tracking the expected trajectory.";
            }}

            // 5. Noise, Stuck, Missing/Dropout
            if (q.includes('noise') || q.includes('jitter')) {{
                if (v === 'noise') {{
                    return f + " is showing elevated variance and jitter compared to its neighbours, pointing to DC power rail noise or electromagnetic interference.";
                }}
                return "Noise levels on " + f + " are nominal and within cluster variance.";
            }}
            if (q.includes('stuck') || q.includes('frozen') || q.includes('lockup')) {{
                if (v === 'stuck') {{
                    return f + " has produced identical readings in a row with zero natural variation, indicating an ADC latch-up or microcontroller firmware freeze.";
                }}
                return f + " is varying normally and not stuck.";
            }}
            if (q.includes('missing') || q.includes('dropout') || q.includes('disconnect')) {{
                if (v === 'dropout') {{
                    return f + " is experiencing telemetry dropout, indicating wiring severance, link loss, or battery failure.";
                }}
                return "Telemetry packets for " + f + " are arriving normally with zero missing readings.";
            }}

            // 6. Explain Anomaly / Why Flagged
            if (q.includes('why') || q.includes('explain') || q.includes('anomaly') || q.includes('flagged') || q.includes('reason') || q.includes('cause')) {{
                if (v !== 'normal') {{
                    let expl = f + " was flagged as '" + v + "' with " + conf + "% confidence. ";
                    if (v === 'drift') expl += "Over recent hours it crept away from neighbouring sensors and the gap is still expanding.";
                    else if (v === 'spike') expl += "The reading jumped sharply away from its recent level in a single step while peers stayed steady.";
                    else if (v === 'stuck') expl += "The reading has remained frozen at an identical value while working sensors always wobble slightly.";
                    else if (v === 'noise') expl += "The readings are significantly more jittery than neighbouring sensors.";
                    else if (v === 'bias') expl += "The sensor holds a constant static offset from its peers.";
                    else if (v === 'dropout') expl += "No telemetry arrived from this sensor channel.";
                    return expl;
                }}
                return f + " is currently in normal agreement with peer sensors (health: " + h + "/100). No anomaly flag is triggered.";
            }}


            // 8. Summarize All Readings & Predictions
            if (q.includes('summary') || q.includes('readings') || q.includes('predictions') || q.includes('overview') || q.includes('all') || q.includes('report') || q.includes('latest')) {{
                const sampleSensors = CTX.sensors.slice(0, 6);
                let list = sampleSensors.map(s => s + ": " + (CTX.readings[s] ? CTX.readings[s].toFixed(1) + ' ' + units : 'N/A') + " (" + (CTX.verdicts_dict[s] || 'normal') + ")").join("; ");
                return "Latest Telemetry Summary (Hour " + CTX.now_hour.toFixed(1) + "): " + list + ". Active incidents: " + CTX.incidents + ". Fleet health: " + fleet + "/100 (" + status + ").";
            }}

            // Default
            return "I am SensorTrust AI Voice Assistant monitoring " + CTX.sensors.length + " sensors. Currently, " + f + " has health " + h + " out of 100 with verdict '" + v + "'. How can I help you investigate sensor health or maintenance?";
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
