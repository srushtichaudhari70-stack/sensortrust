"""AI Voice System for SensorTrust.

Supports:
1. Google Gemini 3.8 Flash TTS (`gemini-3.8-flash-tts` / `gemini-3.8-flash-lite-tts`) via `google-genai` SDK.
2. In-browser Web Speech API (SpeechSynthesis & SpeechRecognition) for zero-latency, client-side voice playback and voice commands.
3. Acoustic Industrial Alert Tone Generator using standard library `wave` for telemetry auditory cues.
"""
from __future__ import annotations

import io
import math
import os
import struct
import wave
from typing import Optional
from ai_agent import load_gemini_api_key


def synthesize_gemini_tts(text: str, api_key: Optional[str] = None, voice_name: str = "Kore") -> Optional[bytes]:
    """Generate high-fidelity AI speech using Google Gemini 3.8 Flash TTS."""
    key = load_gemini_api_key(api_key)
    if not key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        # Call Gemini TTS model
        interaction = client.interactions.create(
            model="gemini-3.8-flash-tts",
            input=text[:1500],  # Concise voice summary
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
    except Exception as e:
        # If API key does not have TTS access or fails, fallback safely
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
                # Envelope attack-decay
                env = math.sin(math.pi * i / total_samples)
                val = int(32767.0 * 0.4 * env * math.sin(2.0 * math.pi * freq * t))
                frames.extend(struct.pack("<h", val))
            wf.writeframes(frames)
            
    return buf.getvalue()


def render_browser_voice_player(text_to_speak: str, button_label: str = "🔊 Read Aloud (Voice)") -> str:
    """Generates an embedded HTML5 Web Speech API script allowing immediate voice playback in any browser."""
    clean_text = text_to_speak.replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
    html_code = f"""
    <div style="margin: 8px 0; display: inline-flex; gap: 8px; align-items: center;">
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
            
            // Choose a clear English voice if available
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
    <div style="margin: 6px 0;">
        <button id="mic-btn" onclick="startDictation()" style="
            background: #10b981;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
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
            
            status.textContent = 'Listening...';
            recognition.start();
            
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                status.textContent = 'Heard: "' + transcript + '"';
                
                // Copy to clipboard or trigger streamlit input if possible
                navigator.clipboard.writeText(transcript);
                status.textContent += ' (Copied to clipboard - paste into chat!)';
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
