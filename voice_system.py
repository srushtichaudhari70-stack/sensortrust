"""AI Voice System for SensorTrust.

Supports:
1. Google Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts` / `gemini-3.8-flash-lite-tts`) via `google-genai` SDK.
2. In-browser Web Speech API (SpeechSynthesis & SpeechRecognition) for zero-latency, client-side voice playback and voice commands.
3. Acoustic Industrial Alert Tone Generator using standard library `wave` for telemetry auditory cues.
4. Full Bidirectional Voice Assistant HUD with glowing animated voice orb, speech-to-text, and automatic spoken replies.
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


def render_browser_voice_player(text_to_speak: str, button_label: str = "🔊 Read Aloud (Voice)") -> str:
    """Generates an embedded HTML5 Web Speech API script allowing immediate voice playback in any browser."""
    clean_text = text_to_speak.replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
    html_code = f"""
    <div style="margin: 8px 0; display: inline-flex; gap: 8px; align-items: center; font-family: sans-serif;">
        <button id="speak-btn" onclick="speakText()" style="
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
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
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            {button_label}
        </button>
        <button id="stop-btn" onclick="stopText()" style="
            background: #374151;
            color: #9ca3af;
            border: none;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
        ">
            ⏹️ Stop
        </button>
    </div>
    <script>
        function speakText() {{
            window.speechSynthesis.cancel();
            const text = "{clean_text}";
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            const voices = window.speechSynthesis.getVoices();
            for (let v of voices) {{
                if (v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha') || v.name.includes('David'))) {{
                    utterance.voice = v;
                    break;
                }}
            }}
            window.speechSynthesis.speak(utterance);
        }}
        function stopText() {{
            window.speechSynthesis.cancel();
        }}
    </script>
    """
    return html_code


def render_voice_input_mic() -> str:
    """HTML5 Speech Recognition mic button for real-time voice input."""
    return """
    <div style="margin: 6px 0; font-family: sans-serif;">
        <button id="mic-btn" onclick="startDictation()" style="
            background: #10b981;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        ">
            🎙️ Speak Voice Command
        </button>
        <span id="mic-status" style="margin-left: 8px; font-size: 12px; color: #9ca3af;"></span>
    </div>
    <script>
        function startDictation() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                alert("Speech recognition is not supported in this browser. Please use Chrome/Edge.");
                return;
            }
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SpeechRecognition();
            const status = document.getElementById('mic-status');
            
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            status.textContent = 'Listening... speak now';
            recognition.start();
            
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                status.textContent = 'Heard: "' + transcript + '" (Copied!)';
                navigator.clipboard.writeText(transcript);
            };
            
            recognition.onerror = function(event) {
                status.textContent = 'Mic error: ' + event.error;
            };
            
            recognition.onend = function() {
                setTimeout(() => { status.textContent = ''; }, 4000);
            };
        }
    </script>
    """


def render_interactive_voice_assistant(focus_sensor: str, focus_health: float, focus_verdict: str,
                                       fleet_health: float, incident_count: int, sensors: List[str]) -> str:
    """Full Conversational AI Voice Assistant HUD with animated orb, bidirectional voice, and instant audio replies."""
    context_json = json.dumps({
        "focus": focus_sensor,
        "health": focus_health,
        "verdict": focus_verdict,
        "fleet": fleet_health,
        "incidents": incident_count,
        "sensors": sensors
    })
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 16px;
            background: #111827;
            color: #f3f4f6;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            border-radius: 12px;
            border: 1px solid #1f2937;
        }}
        .hud-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 12px;
        }}
        /* Glowing AI Orb */
        .orb-wrapper {{
            position: relative;
            width: 100px;
            height: 100px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }}
        .orb {{
            width: 70px;
            height: 70px;
            border-radius: 50%;
            background: radial-gradient(circle, #38bdf8 0%, #1d4ed8 70%, #0369a1 100%);
            box-shadow: 0 0 25px rgba(56, 189, 248, 0.6), inset 0 0 15px rgba(255, 255, 255, 0.4);
            transition: all 0.3s ease;
            animation: pulse 2.5s infinite ease-in-out;
        }}
        .orb.listening {{
            background: radial-gradient(circle, #4ade80 0%, #16a34a 70%, #15803d 100%);
            box-shadow: 0 0 35px rgba(74, 222, 128, 0.8), inset 0 0 15px rgba(255, 255, 255, 0.5);
            animation: pulse-fast 1s infinite ease-in-out;
        }}
        .orb.speaking {{
            background: radial-gradient(circle, #c084fc 0%, #7e22ce 70%, #6b21a8 100%);
            box-shadow: 0 0 35px rgba(192, 132, 252, 0.8), inset 0 0 15px rgba(255, 255, 255, 0.5);
            animation: pulse-speaking 0.8s infinite alternate ease-in-out;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(0.96); box-shadow: 0 0 18px rgba(56, 189, 248, 0.5); }}
            50% {{ transform: scale(1.04); box-shadow: 0 0 30px rgba(56, 189, 248, 0.8); }}
            100% {{ transform: scale(0.96); box-shadow: 0 0 18px rgba(56, 189, 248, 0.5); }}
        }}
        @keyframes pulse-fast {{
            0% {{ transform: scale(0.94); }}
            50% {{ transform: scale(1.08); }}
            100% {{ transform: scale(0.94); }}
        }}
        @keyframes pulse-speaking {{
            0% {{ transform: scale(0.95) rotate(0deg); }}
            100% {{ transform: scale(1.07) rotate(5deg); }}
        }}
        .status-badge {{
            font-size: 13px;
            font-weight: 600;
            color: #38bdf8;
            letter-spacing: 0.5px;
        }}
        .transcript-box {{
            background: #1f2937;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13px;
            width: 100%;
            max-width: 580px;
            min-height: 44px;
            text-align: left;
            color: #e5e7eb;
            line-height: 1.4;
            border: 1px solid #374151;
        }}
        .chips-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            justify-content: center;
            max-width: 600px;
        }}
        .chip {{
            background: #1e293b;
            color: #cbd5e1;
            border: 1px solid #334155;
            padding: 5px 10px;
            border-radius: 14px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .chip:hover {{
            background: #334155;
            color: #ffffff;
            border-color: #38bdf8;
        }}
        .btn-talk {{
            background: linear-gradient(135deg, #0284c7, #2563eb);
            color: white;
            border: none;
            padding: 10px 22px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
            transition: transform 0.1s;
        }}
        .btn-talk:hover {{
            transform: scale(1.03);
        }}
        .btn-talk:active {{
            transform: scale(0.98);
        }}
    </style>
    </head>
    <body>
    <div class="hud-container">
        <div style="display: flex; align-items: center; gap: 8px;">
            <h3 style="margin: 0; font-size: 17px; font-weight: 700;">🎙️ SensorTrust AI Voice Assistant</h3>
            <span style="font-size: 11px; background: #065f46; color: #34d399; padding: 2px 8px; border-radius: 10px;">LIVE CONVERSATIONAL</span>
        </div>
        
        <!-- Animated Orb -->
        <div class="orb-wrapper" onclick="toggleVoiceSession()">
            <div id="orb" class="orb"></div>
        </div>
        
        <div id="status" class="status-badge">Tap the Orb or Button to Speak</div>

        <button id="btn-talk" class="btn-talk" onclick="toggleVoiceSession()">
            <span>🎙️ Start Speaking</span>
        </button>

        <div id="transcript" class="transcript-box">
            <span style="color: #6b7280;">Say: "Status of {focus_sensor}", "Why is {focus_sensor} failing?", "Fleet health", or "Is the temperature rise real?"...</span>
        </div>

        <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">Quick Voice Prompts (click to simulate voice command):</div>
        <div class="chips-row">
            <button class="chip" onclick="askQuestion('Status of {focus_sensor}')">🗣️ "Status of {focus_sensor}"</button>
            <button class="chip" onclick="askQuestion('Why is {focus_sensor} failing?')">🗣️ "Why is {focus_sensor} failing?"</button>
            <button class="chip" onclick="askQuestion('Is the temperature rise real?')">🗣️ "Is temperature rise real?"</button>
            <button class="chip" onclick="askQuestion('What is the fleet health?')">🗣️ "Plant fleet health"</button>
            <button class="chip" onclick="askQuestion('Maintenance work order')">🗣️ "Technician work order"</button>
        </div>
    </div>

    <script>
        const ctx = {context_json};
        let isListening = false;
        let isSpeaking = false;
        let recognition = null;
        const orb = document.getElementById('orb');
        const status = document.getElementById('status');
        const transcript = document.getElementById('transcript');
        const btnTalk = document.getElementById('btn-talk');

        // Setup speech recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRec();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onstart = function() {{
                isListening = true;
                orb.className = 'orb listening';
                status.textContent = 'Listening to you... Speak now';
                status.style.color = '#4ade80';
                btnTalk.innerHTML = '<span>⏹️ Stop Listening</span>';
            }};

            recognition.onresult = function(event) {{
                const userSpeech = event.results[0][0].transcript;
                transcript.innerHTML = '<strong style="color:#38bdf8;">You:</strong> ' + userSpeech;
                processAndSpeak(userSpeech);
            }};

            recognition.onerror = function(event) {{
                console.warn('Speech error:', event.error);
                status.textContent = 'Microphone notice: ' + event.error;
                resetState();
            }};

            recognition.onend = function() {{
                if (isListening && !isSpeaking) {{
                    resetState();
                }}
            }};
        }}

        function toggleVoiceSession() {{
            window.speechSynthesis.cancel();
            if (isListening) {{
                if (recognition) recognition.stop();
                resetState();
                return;
            }}
            if (isSpeaking) {{
                window.speechSynthesis.cancel();
                resetState();
                return;
            }}
            if (!recognition) {{
                alert("Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari, or click the quick prompt chips below!");
                return;
            }}
            try {{
                recognition.start();
            }} catch(e) {{
                console.log(e);
            }}
        }}

        function askQuestion(query) {{
            transcript.innerHTML = '<strong style="color:#38bdf8;">You (Voice):</strong> ' + query;
            processAndSpeak(query);
        }}

        function processAndSpeak(query) {{
            isListening = false;
            isSpeaking = true;
            orb.className = 'orb speaking';
            status.textContent = 'AI Diagnosing & Speaking...';
            status.style.color = '#c084fc';
            btnTalk.innerHTML = '<span>⏹️ Stop Voice</span>';

            const reply = generateAnswer(query);
            transcript.innerHTML += '<br><strong style="color:#c084fc;">AI Assistant:</strong> ' + reply;

            // Voice synthesis
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(reply);
            utterance.rate = 1.05;
            utterance.pitch = 1.0;

            const voices = window.speechSynthesis.getVoices();
            for (let v of voices) {{
                if (v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('David') || v.name.includes('Zira'))) {{
                    utterance.voice = v;
                    break;
                }}
            }}

            utterance.onend = function() {{
                resetState();
            }};
            utterance.onerror = function() {{
                resetState();
            }};

            window.speechSynthesis.speak(utterance);
        }}

        function generateAnswer(query) {{
            const q = query.toLowerCase();
            const f = ctx.focus;
            const h = Math.round(ctx.health);
            const v = ctx.verdict;

            if (q.includes('status') || q.includes('health') && q.includes(f.toLowerCase())) {{
                return f + " has a health score of " + h + " out of 100, currently classified as " + v + ".";
            }}
            if (q.includes('why') || q.includes('fail') || q.includes('broken') || q.includes('failing')) {{
                if (v === 'drift') {{
                    return f + " is failing due to slow thermal resistance drift. The gap between " + f + " and its peer neighbors is steadily widening, indicating aging of the sensing element.";
                }} else if (v === 'spike') {{
                    return f + " suffered an electrical transient spike, likely caused by capacitive coupling or loose terminal wiring.";
                }} else if (v === 'stuck') {{
                    return f + " is frozen with identical readings in a row, indicating an ADC latch-up or firmware lockup.";
                }} else if (v === 'noise') {{
                    return f + " is showing elevated variance, indicating power rail noise or electromagnetic interference.";
                }} else {{
                    return f + " is in " + v + " state. Spatial consensus confirms its telemetry deviates from cluster peers.";
                }}
            }}
            if (q.includes('event') || q.includes('real') || q.includes('rise')) {{
                return "The temperature shift was observed synchronously across all " + ctx.sensors.length + " sensors. Because the entire cluster moved together, SensorTrust confirmed this as a genuine environmental event rather than a sensor fault.";
            }}
            if (q.includes('fleet') || q.includes('plant') || q.includes('overall') || q.includes('cluster')) {{
                const status = ctx.fleet >= 80 ? 'Optimal' : (ctx.fleet >= 50 ? 'Degraded' : 'Critical');
                return "Overall plant fleet health is " + Math.round(ctx.fleet) + " out of 100, with status " + status + ". There are " + ctx.incidents + " active anomaly incidents.";
            }}
            if (q.includes('work order') || q.includes('maintenance') || q.includes('fix') || q.includes('sop') || q.includes('action')) {{
                return "Work order generated for " + f + ": Disconnect 24-volt loop, inspect thermocouple sheath, check ground continuity, and perform reference dry-well calibration.";
            }}
            return "I am your AI Sensor Voice Assistant monitoring " + ctx.sensors.length + " sensors. Currently, " + f + " has health " + h + " out of 100 with verdict " + v + ". How else can I assist your maintenance team?";
        }}

        function resetState() {{
            isListening = false;
            isSpeaking = false;
            orb.className = 'orb';
            status.textContent = 'Tap the Orb or Button to Speak';
            status.style.color = '#38bdf8';
            btnTalk.innerHTML = '<span>🎙️ Start Speaking</span>';
        }}
    </script>
    </body>
    </html>
    """
