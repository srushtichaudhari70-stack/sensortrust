"""Comprehensive Verification Suite for SensorTrust Dynamic Voice Assistant (Tests A-F).

Tests:
Test A: Ask for current sensor health and compare response against displayed sensor health.
Test B: Change selected sensor and ask for its health again (reflects newly selected sensor).
Test C: Change number of sensors and verify reported sensor count updates correctly.
Test D: Change scenario or random seed and check whether assistant reflects updated sensor results.
Test E: Ask about a sensor or value that does not exist and confirm assistant does NOT invent a result.
Test F: Verify Web Speech API (STT en-IN & TTS en-IN), iframe permission observer, and audio synthesis.
"""

import sys
import json
import re
import core
import voice_system

H = core.PER_HOUR
DAYS = 2

PRESETS = {
    "Mixed: six faults and two real events": dict(
        events=((16 * H, 3.5), (44 * H, -3.0)),
        faults=((3, "spike", 12 * H, 1, 5.0), (1, "drift", 20 * H, 8 * H, 3.0),
                (2, "dropout", 26 * H, H, 0.0), (4, "stuck", 30 * H, 3 * H, 0.0),
                (6, "noise", 34 * H, 3 * H, 7.0), (5, "bias", 37 * H, 4 * H, -2.0))),
    "Real events only (no sensor is broken)": dict(events=((16 * H, 4.0), (34 * H, -3.0)), faults=()),
    "Faults only (nothing really happened)": dict(
        events=(),
        faults=((1, "drift", 14 * H, 8 * H, -3.0), (3, "spike", 22 * H, 1, -4.0),
                (5, "bias", 26 * H, 6 * H, 2.5), (6, "stuck", 36 * H, 3 * H, 0.0))),
    "Clean data": dict(events=(), faults=()),
}

_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = core.load_or_train_model()
    return _MODEL

def run_pipeline(n_sensors=6, seed=42, preset_name="Mixed: six faults and two real events", focus_sensor="S1", now_h=None):
    model = get_model()
    scen = PRESETS.get(preset_name, PRESETS["Clean data"])
    events = list(scen["events"])
    faults = [dict(sensor=s, kind=k, start=a, dur=d, mag=m) for s, k, a, d, m in scen["faults"] if s < n_sensors]
    
    df, _ = core.simulate(n_sensors=n_sensors, days=DAYS, seed=seed, events=events, faults=faults)
    res, unit = core.analyse(df, model)
    
    sensors = list(df.columns)
    total_steps = len(df)
    last_h = (total_steps - 1) / H
    if now_h is None:
        now_h = float(last_h)
        
    now = min(total_steps - 1, int(now_h * H))
    view = res[res["pos"] <= now]
    inc = core.incidents(res, upto=now)
    latest = view[view["pos"] == now].set_index("sensor")
    
    health_dict = {s: float(latest.loc[s, "health"]) for s in sensors}
    verdicts_dict = {s: str(latest.loc[s, "verdict"]) for s in sensors}
    confidences_dict = {s: float(latest.loc[s, "confidence"]) * 100 for s in sensors}
    latest_readings = {s: float(df.iloc[now][s]) for s in sensors}
    fleet_health = float(latest["health"].mean())
    fleet_status = "Optimal" if fleet_health >= 80 else ("Watch" if fleet_health >= 50 else "Critical")
    
    incidents_list = [
        {
            "sensor": r.sensor,
            "readings": int(r.readings),
            "verdict": str(r.verdict),
            "confidence": float(r.confidence),
            "row": int(r.row)
        }
        for r in inc.itertuples()
    ]
    
    shap_dict = {}
    for r in inc.itertuples():
        try:
            r_text, r_top, r_act = core.explain(res, model, unit, int(r.row), len(sensors), "°C")
            shap_dict[r.sensor] = {"reason": r_text, "top": r_top, "action": r_act}
        except Exception:
            pass
            
    context = {
        "sensors": sensors,
        "readings": latest_readings,
        "health_scores": health_dict,
        "health_dict": health_dict,
        "verdicts": verdicts_dict,
        "verdicts_dict": verdicts_dict,
        "confidences": confidences_dict,
        "confidences_dict": confidences_dict,
        "incidents": incidents_list,
        "incidents_list": incidents_list,
        "shap_explanations": shap_dict,
        "shap": shap_dict,
        "units": "°C",
        "fleet_health": fleet_health,
        "fleet_status": fleet_status,
        "now_hour": float(now_h),
        "data_source": "simulated temperature sensors",
        "is_simulated": True,
        "focus": focus_sensor,
        "scenario": preset_name,
        "seed": seed
    }
    
    hud_html = voice_system.render_interactive_voice_assistant(
        focus_sensor=focus_sensor,
        focus_health=health_dict.get(focus_sensor, 100.0),
        focus_verdict=verdicts_dict.get(focus_sensor, "normal"),
        fleet_health=fleet_health,
        incident_count=len(incidents_list),
        sensors=sensors,
        latest_readings=latest_readings,
        health_dict=health_dict,
        verdicts_dict=verdicts_dict,
        confidences_dict=confidences_dict,
        incidents_list=incidents_list,
        shap_dict=shap_dict,
        units="°C",
        now_hour=float(now_h),
        data_source="simulated temperature sensors",
        is_simulated=True,
        scenario=preset_name,
        seed=seed
    )
    
    return context, hud_html, latest, inc


def main():
    print("=" * 70)
    print("STARTING SENSORTRUST VOICE ASSISTANT TEST SUITE (TESTS A - F)")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # TEST A: Current Sensor Health vs Displayed Telemetry
    # -------------------------------------------------------------
    print("\n--- TEST A: Current sensor health matching displayed state ---")
    ctx_a, html_a, latest_a, _ = run_pipeline(n_sensors=6, seed=42, preset_name="Mixed: six faults and two real events", focus_sensor="S1", now_h=20.0)
    expected_s1_health = latest_a.loc["S1", "health"]
    expected_s1_verdict = latest_a.loc["S1", "verdict"]
    expected_s1_val = ctx_a["readings"]["S1"]
    
    resp_a = voice_system.generate_grounded_voice_response("What is my current sensor health?", ctx_a)
    print(f"Query: 'What is my current sensor health?'")
    print(f"Displayed: S1 Health={expected_s1_health:.1f}, Verdict={expected_s1_verdict}, Reading={expected_s1_val:.2f}°C")
    print(f"Response: {resp_a}")
    
    assert f"{expected_s1_health:.0f} out of 100" in resp_a or f"{expected_s1_health:.0f}" in resp_a, "Health score mismatch!"
    assert expected_s1_verdict in resp_a, "Verdict mismatch!"
    assert f"{expected_s1_val:.2f}" in resp_a or f"{expected_s1_val:.1f}" in resp_a, "Reading mismatch!"
    assert "simulated" in resp_a, "Missing simulation data provenance tag!"
    print(">>> TEST A PASSED: Assistant response perfectly matches displayed sensor health & reading.")

    # -------------------------------------------------------------
    # TEST B: Change Selected Sensor
    # -------------------------------------------------------------
    print("\n--- TEST B: Change selected sensor from S1 to S2 (anomalous) ---")
    ctx_b, html_b, latest_b, _ = run_pipeline(n_sensors=6, seed=42, preset_name="Mixed: six faults and two real events", focus_sensor="S2", now_h=20.0)
    expected_s2_health = latest_b.loc["S2", "health"]
    expected_s2_verdict = latest_b.loc["S2", "verdict"]
    expected_s2_val = ctx_b["readings"]["S2"]
    
    resp_b = voice_system.generate_grounded_voice_response("What is my current sensor health?", ctx_b)
    print(f"Query: 'What is my current sensor health?' (focus switched to S2)")
    print(f"Displayed: S2 Health={expected_s2_health:.1f}, Verdict={expected_s2_verdict}, Reading={expected_s2_val:.2f}°C")
    print(f"Response: {resp_b}")
    
    assert "S2" in resp_b, "Response did not reflect S2!"
    assert f"{expected_s2_health:.0f}" in resp_b, "S2 Health score mismatch!"
    assert f"{expected_s2_val:.2f}" in resp_b or f"{expected_s2_val:.1f}" in resp_b, "S2 reading mismatch!"
    
    # Also test specific query "What is the latest reading for the selected sensor?"
    resp_b_reading = voice_system.generate_grounded_voice_response("What is the latest reading for the selected sensor?", ctx_b)
    print(f"Query: 'What is the latest reading for the selected sensor?'")
    print(f"Response: {resp_b_reading}")
    assert "S2" in resp_b_reading and f"{expected_s2_val:.2f}" in resp_b_reading, "Latest reading mismatch!"
    print(">>> TEST B PASSED: Switching selected sensor dynamically updates target sensor, health, and reading.")

    # -------------------------------------------------------------
    # TEST C: Change Number of Sensors
    # -------------------------------------------------------------
    print("\n--- TEST C: Change number of sensors (4 vs 8) ---")
    ctx_c4, _, _, _ = run_pipeline(n_sensors=4, seed=42, now_h=20.0)
    ctx_c8, _, _, _ = run_pipeline(n_sensors=8, seed=42, now_h=20.0)
    
    resp_c4 = voice_system.generate_grounded_voice_response("How many sensors are healthy?", ctx_c4)
    resp_c8 = voice_system.generate_grounded_voice_response("How many sensors are healthy?", ctx_c8)
    
    print(f"4-Sensor cluster response: {resp_c4}")
    print(f"8-Sensor cluster response: {resp_c8}")
    
    assert "4 sensors" in resp_c4, "Expected '4 sensors' in 4-sensor cluster response!"
    assert "8 sensors" in resp_c8, "Expected '8 sensors' in 8-sensor cluster response!"
    
    # Also verify fleet question
    resp_fleet_c8 = voice_system.generate_grounded_voice_response("What is overall fleet health?", ctx_c8)
    assert "across all 8 simulated sensors" in resp_fleet_c8, "Fleet response did not reflect 8 sensors!"
    print(">>> TEST C PASSED: Sensor count dynamically updates across single and aggregate questions.")

    # -------------------------------------------------------------
    # TEST D: Change Scenario or Random Seed
    # -------------------------------------------------------------
    print("\n--- TEST D: Change scenario (Real Events vs Faults Only at hour 20) and seed (42 vs 999) ---")
    ctx_d_events, _, _, _ = run_pipeline(n_sensors=6, seed=42, preset_name="Real events only (no sensor is broken)", now_h=16.5)
    ctx_d_faults, _, _, _ = run_pipeline(n_sensors=6, seed=42, preset_name="Faults only (nothing really happened)", now_h=20.0)
    ctx_d_seed999, _, _, _ = run_pipeline(n_sensors=6, seed=999, preset_name="Faults only (nothing really happened)", now_h=20.0)
    
    resp_events = voice_system.generate_grounded_voice_response("Is the latest event genuine or suspicious?", ctx_d_events)
    resp_faults = voice_system.generate_grounded_voice_response("Is the latest event genuine or suspicious?", ctx_d_faults)
    resp_faults_anom = voice_system.generate_grounded_voice_response("Which sensors have anomalies?", ctx_d_faults)
    
    print(f"Real Events scenario query ('Is the latest event genuine or suspicious?'):\n-> {resp_events}")
    print(f"\nFaults Only scenario query ('Is the latest event genuine or suspicious?'):\n-> {resp_faults}")
    print(f"\nFaults Only scenario query ('Which sensors have anomalies?'):\n-> {resp_faults_anom}")
    
    assert "GENUINE" in resp_events, "Real events scenario should detect a GENUINE event!"
    assert "SUSPICIOUS" in resp_faults or "fault" in resp_faults, "Faults scenario should detect SUSPICIOUS fault!"
    assert "S2" in resp_faults_anom and "drift" in resp_faults_anom, "S2 drift should be listed in anomalies!"
    
    # Verify seed change generates different readings
    s1_seed42 = ctx_d_faults["readings"]["S1"]
    s1_seed999 = ctx_d_seed999["readings"]["S1"]
    print(f"Seed 42 S1 reading: {s1_seed42:.3f}°C | Seed 999 S1 reading: {s1_seed999:.3f}°C")
    assert s1_seed42 != s1_seed999, "Different random seed must produce different telemetry!"
    
    resp_seed42 = voice_system.generate_grounded_voice_response("What is the latest reading for selected sensor S1?", ctx_d_faults)
    resp_seed999 = voice_system.generate_grounded_voice_response("What is the latest reading for selected sensor S1?", ctx_d_seed999)
    print(f"Seed 42 S1 Voice Reading: {resp_seed42}")
    print(f"Seed 999 S1 Voice Reading: {resp_seed999}")
    assert f"{s1_seed42:.2f}" in resp_seed42
    assert f"{s1_seed999:.2f}" in resp_seed999
    print(">>> TEST D PASSED: Scenario changes and random seeds correctly regenerate telemetry and assistant answers.")

    # -------------------------------------------------------------
    # TEST E: Non-Existent Sensor Validation (No Inventions)
    # -------------------------------------------------------------
    print("\n--- TEST E: Ask about non-existent sensors (e.g. S9 or S15 in 6-sensor cluster) ---")
    resp_e9 = voice_system.generate_grounded_voice_response("What is the current status of sensor S9?", ctx_a)
    resp_e15 = voice_system.generate_grounded_voice_response("What is the health of sensor S15?", ctx_a)
    
    print(f"Query: 'What is the current status of sensor S9?'")
    print(f"Response: {resp_e9}")
    assert "Sensor S9 does not exist in the active cluster" in resp_e9, "Failed to reject non-existent S9!"
    assert "S1, S2, S3, S4, S5, S6" in resp_e9, "Failed to list valid cluster sensors!"
    
    # Also verify non-existing data handling
    empty_ctx = {"sensors": [], "health_scores": {}}
    resp_empty = voice_system.generate_grounded_voice_response("What is my current sensor health?", empty_ctx)
    print(f"Empty data query response: {resp_empty}")
    assert "unavailable" in resp_empty.lower(), "Empty context should report data unavailable!"
    print(">>> TEST E PASSED: Assistant safely refuses to invent data for non-existent sensors or missing telemetry.")

    # -------------------------------------------------------------
    # TEST F: Web Speech API & Indian English TTS Verification
    # -------------------------------------------------------------
    print("\n--- TEST F: Web Speech API, en-IN voice preference, Stop button & Audio tones ---")
    # Verify rendered HUD contains SpeechRecognition, en-IN, Stop button, Equalizer, and iframe permissions
    assert "recognition.lang = 'en-IN'" in html_a, "Speech recognition missing en-IN preference!"
    assert "utterance.lang = 'en-IN'" in html_a, "Speech synthesis missing en-IN preference!"
    assert "Stop Speaking" in html_a, "Stop Speaking button missing from UI!"
    assert "Stop Listening" in html_a, "Stop Listening button missing from UI!"
    assert "generateGroundedAnswer" in html_a, "generateGroundedAnswer missing in client JS!"
    assert "status-listening" in html_a and "status-speaking" in html_a and "status-processing" in html_a, "States missing in CSS/JS!"
    
    # Verify acoustic tone generation works
    tone_wav = voice_system.generate_acoustic_alert_tone("spike")
    assert len(tone_wav) > 1000 and tone_wav.startswith(b"RIFF"), "Acoustic WAV alert generation failed!"
    print(f"Generated {len(tone_wav)} bytes valid RIFF WAV alert tone for spike anomaly.")
    
    # Verify index.html iframe microphone permissions observer
    with open("index.html", "r", encoding="utf-8") as f:
        idx_content = f.read()
    assert "microphone *" in idx_content and "speech-recognition *" in idx_content, "index.html missing microphone iframe permissions!"
    print("Verified index.html microphone permissions observer.")
    print(">>> TEST F PASSED: Voice assistant components, Web Speech API configuration, and Indian English TTS intact.")

    print("\n" + "=" * 70)
    print("ALL TESTS (A, B, C, D, E, F) EXECUTED AND PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
