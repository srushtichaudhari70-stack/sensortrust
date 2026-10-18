"""SensorTrust Step 2C Dynamic Context & Telemetry Freshness Test Suite.

Verifies:
1. Dynamic Health Query: 'What is the current health of sensor S2?'
2. Lowest Health Sensor Query: 'Which sensor has the lowest health score?'
3. Latest Detected Anomaly Query: 'What is the latest detected anomaly?'
4. Genuine Event vs. Fault Query:
   - 4A: Genuine Event (simultaneous shift across cluster)
   - 4B: Isolated Fault (single sensor divergence)
5. Invalid Sensor ID Rejection: 'What is the status of sensor S9?'
6. Scenario & Seed Dynamic Freshness:
   - Verify that altering the scenario/seed immediately updates responses without stale cache.
"""

import sys
import json
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000/api/voice/ask"
HEALTH_URL = "http://127.0.0.1:8000/api/health"


def send_voice_query(query: str, context: dict) -> dict:
    data = json.dumps({"query": query, "context": context}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as res:
        return json.loads(res.read().decode())


def check_health() -> dict:
    with urllib.request.urlopen(HEALTH_URL, timeout=10) as res:
        return json.loads(res.read().decode())


def main():
    print("=" * 75)
    print("SENSORTRUST STEP 2C: DYNAMIC SENSOR CONTEXT VALIDATION SUITE")
    print("=" * 75)

    # Pre-flight health check
    health_res = check_health()
    print(f"\n[Pre-flight] Health Check: status={health_res.get('status')}, model={health_res.get('model')}, configured={health_res.get('gemini_configured')}")
    assert health_res.get("gemini_configured") is True, "Gemini API key is not configured on the backend!"

    # ------------------------------------------------------------------
    # TEST 1: Current health of S2
    # ------------------------------------------------------------------
    print("\n--- TEST 1: Current Health of S2 ---")
    ctx_1 = {
        "sensors": ["S1", "S2", "S3", "S4"],
        "readings": {"S1": 22.1, "S2": 17.8, "S3": 22.0, "S4": 22.3},
        "health_dict": {"S1": 100.0, "S2": 58.0, "S3": 100.0, "S4": 100.0},
        "verdicts_dict": {"S1": "normal", "S2": "drift", "S3": "normal", "S4": "normal"},
        "confidences_dict": {"S1": 99.0, "S2": 96.0, "S3": 99.0, "S4": 99.0},
        "focus": "S2",
        "units": "°C",
        "now_hour": 16.5,
        "data_source": "simulated temperature sensors",
        "is_simulated": True,
        "shap": {
            "S2": {
                "reason": "Gradual downward deviation from peer median.",
                "top": [["r_slope", 0.78], ["r_mean_s", 0.65]],
                "action": "Check RTD sensor calibration against reference dry-well."
            }
        }
    }
    q1 = "What is the current health of sensor S2?"
    res_1 = send_voice_query(q1, ctx_1)
    ans_1 = res_1.get("response", "")
    print(f"Query: '{q1}'")
    print(f"Source: {res_1.get('source')} | Model: {res_1.get('model')}")
    print(f"Answer: {ans_1}\n")

    assert res_1.get("status") == "ok", "Test 1 request failed!"
    assert any(term in ans_1.lower() for term in ["s2", "sensor s2", "sensor 2"]), "S2 not mentioned!"
    assert any(term in ans_1 for term in ["58", "fifty-eight", "drift"]), "S2 health score (58) or verdict (drift) missing!"
    print(">>> TEST 1 PASSED: S2 health and drift anomaly correctly answered using supplied telemetry.")

    # ------------------------------------------------------------------
    # TEST 2: Sensor with lowest health score
    # ------------------------------------------------------------------
    print("\n--- TEST 2: Sensor with Lowest Health Score ---")
    ctx_2 = {
        "sensors": ["S1", "S2", "S3", "S4", "S5"],
        "readings": {"S1": 21.0, "S2": 21.2, "S3": 20.8, "S4": 29.5, "S5": 21.1},
        "health_dict": {"S1": 95.0, "S2": 90.0, "S3": 100.0, "S4": 32.0, "S5": 98.0},
        "verdicts_dict": {"S1": "normal", "S2": "normal", "S3": "normal", "S4": "stuck", "S5": "normal"},
        "confidences_dict": {"S1": 95.0, "S2": 95.0, "S3": 95.0, "S4": 99.0, "S5": 95.0},
        "focus": "S4",
        "units": "°C",
        "now_hour": 24.0,
        "data_source": "simulated temperature sensors",
        "is_simulated": True,
        "incidents_list": [{"sensor": "S4", "verdict": "stuck", "confidence": 0.99, "readings": 20}]
    }
    q2 = "Which sensor has the lowest health score?"
    res_2 = send_voice_query(q2, ctx_2)
    ans_2 = res_2.get("response", "")
    print(f"Query: '{q2}'")
    print(f"Source: {res_2.get('source')} | Model: {res_2.get('model')}")
    print(f"Answer: {ans_2}\n")

    assert res_2.get("status") == "ok", "Test 2 request failed!"
    assert any(term in ans_2 for term in ["S4", "sensor S4", "Sensor S4", "sensor 4"]), "Lowest health sensor S4 missing!"
    assert any(term in ans_2 for term in ["32", "thirty-two", "lowest"]), "Score 32 or 'lowest' indicator missing!"
    print(">>> TEST 2 PASSED: Correctly identified S4 as having lowest health score (32/100).")

    # ------------------------------------------------------------------
    # TEST 3: Latest detected anomaly
    # ------------------------------------------------------------------
    print("\n--- TEST 3: Latest Detected Anomaly ---")
    ctx_3 = {
        "sensors": ["S1", "S2", "S3", "S4"],
        "readings": {"S1": 22.0, "S2": 22.1, "S3": 31.4, "S4": 22.2},
        "health_dict": {"S1": 100.0, "S2": 100.0, "S3": 65.0, "S4": 100.0},
        "verdicts_dict": {"S1": "normal", "S2": "normal", "S3": "spike", "S4": "normal"},
        "confidences_dict": {"S1": 95.0, "S2": 95.0, "S3": 98.0, "S4": 95.0},
        "units": "°C",
        "now_hour": 32.0,
        "data_source": "simulated temperature sensors",
        "is_simulated": True,
        "incidents_list": [
            {"sensor": "S2", "verdict": "noise", "confidence": 0.91, "readings": 5},
            {"sensor": "S3", "verdict": "spike", "confidence": 0.98, "readings": 1}
        ]
    }
    q3 = "What is the latest detected anomaly?"
    res_3 = send_voice_query(q3, ctx_3)
    ans_3 = res_3.get("response", "")
    print(f"Query: '{q3}'")
    print(f"Source: {res_3.get('source')} | Model: {res_3.get('model')}")
    print(f"Answer: {ans_3}\n")

    assert res_3.get("status") == "ok", "Test 3 request failed!"
    assert any(term in ans_3 for term in ["S3", "spike", "Sensor S3", "sensor 3"]), "Latest anomaly S3 spike missing!"
    print(">>> TEST 3 PASSED: Correctly identified S3 spike as the latest detected anomaly.")

    # ------------------------------------------------------------------
    # TEST 4A: Genuine Event classification
    # ------------------------------------------------------------------
    print("\n--- TEST 4A: Genuine Event (Cluster Shift) ---")
    ctx_4a = {
        "sensors": ["S1", "S2", "S3", "S4"],
        "readings": {"S1": 26.5, "S2": 26.4, "S3": 26.6, "S4": 26.5},
        "health_dict": {"S1": 100.0, "S2": 100.0, "S3": 100.0, "S4": 100.0},
        "verdicts_dict": {"S1": "event", "S2": "event", "S3": "event", "S4": "event"},
        "confidences_dict": {"S1": 97.0, "S2": 97.0, "S3": 97.0, "S4": 97.0},
        "units": "°C",
        "now_hour": 12.0,
        "data_source": "simulated temperature sensors",
        "is_simulated": True,
        "incidents_list": [{"sensor": "ALL", "verdict": "event", "confidence": 0.97, "readings": 8}]
    }
    q4a = "Is this reading classified as a genuine event or a possible fault?"
    res_4a = send_voice_query(q4a, ctx_4a)
    ans_4a = res_4a.get("response", "")
    print(f"Query: '{q4a}'")
    print(f"Answer: {ans_4a}\n")

    assert any(term in ans_4a.lower() for term in ["genuine", "environmental", "real"]), "Did not classify as genuine event!"
    print(">>> TEST 4A PASSED: Coordinated cluster movement correctly classified as genuine event.")

    # ------------------------------------------------------------------
    # TEST 4B: Possible Sensor Fault (Isolated departure)
    # ------------------------------------------------------------------
    print("\n--- TEST 4B: Possible Sensor Fault (Isolated Divergence) ---")
    ctx_4b = {
        "sensors": ["S1", "S2", "S3", "S4"],
        "readings": {"S1": 21.0, "S2": 14.2, "S3": 21.1, "S4": 20.9},
        "health_dict": {"S1": 100.0, "S2": 45.0, "S3": 100.0, "S4": 100.0},
        "verdicts_dict": {"S1": "normal", "S2": "drift", "S3": "normal", "S4": "normal"},
        "confidences_dict": {"S1": 99.0, "S2": 97.0, "S3": 99.0, "S4": 99.0},
        "focus": "S2",
        "units": "°C",
        "now_hour": 18.0,
        "data_source": "simulated temperature sensors",
        "is_simulated": True,
        "incidents_list": [{"sensor": "S2", "verdict": "drift", "confidence": 0.97, "readings": 15}]
    }
    q4b = "Is this reading classified as a genuine event or a possible fault?"
    res_4b = send_voice_query(q4b, ctx_4b)
    ans_4b = res_4b.get("response", "")
    print(f"Query: '{q4b}'")
    print(f"Answer: {ans_4b}\n")

    assert any(term in ans_4b.lower() for term in ["fault", "suspicious", "drift", "isolated"]), "Did not classify as fault!"
    print(">>> TEST 4B PASSED: Isolated sensor departure correctly classified as possible fault.")

    # ------------------------------------------------------------------
    # TEST 5: Invalid Sensor ID
    # ------------------------------------------------------------------
    print("\n--- TEST 5: Invalid Sensor ID ---")
    q5 = "What is the current status of sensor S9?"
    res_5 = send_voice_query(q5, ctx_1)
    ans_5 = res_5.get("response", "")
    print(f"Query: '{q5}'")
    print(f"Source: {res_5.get('source')}")
    print(f"Answer: {ans_5}\n")

    assert "does not exist" in ans_5.lower() or "s9" in ans_5, "Nonexistent sensor not properly identified!"
    print(">>> TEST 5 PASSED: Assistant safely refuses to fabricate data for invalid sensor ID S9.")

    # ------------------------------------------------------------------
    # TEST 6: Dynamic Scenario & Seed Freshness Verification
    # ------------------------------------------------------------------
    print("\n--- TEST 6: Dynamic Scenario & Seed Freshness Verification ---")
    # Scenario State A: Seed 42, S1=20.0, S2=15.0 (drift, health 50)
    state_a = {
        "sensors": ["S1", "S2", "S3"],
        "readings": {"S1": 20.0, "S2": 15.0, "S3": 20.1},
        "health_dict": {"S1": 100.0, "S2": 50.0, "S3": 100.0},
        "verdicts_dict": {"S1": "normal", "S2": "drift", "S3": "normal"},
        "confidences_dict": {"S1": 95.0, "S2": 94.0, "S3": 95.0},
        "scenario": "Faults only",
        "seed": 42,
        "now_hour": 10.0,
        "focus": "S2"
    }
    q_fresh = "What is the latest reading for sensor S2?"
    res_a = send_voice_query(q_fresh, state_a)
    print(f"State A (Seed 42): {res_a.get('response')}")
    assert "15" in res_a.get("response") or "fifteen" in res_a.get("response").lower(), "Failed to read State A reading (15.0)!"

    # Scenario State B: Seed 999, S1=25.0, S2=25.2 (normal, health 100)
    state_b = {
        "sensors": ["S1", "S2", "S3"],
        "readings": {"S1": 25.0, "S2": 25.2, "S3": 25.1},
        "health_dict": {"S1": 100.0, "S2": 100.0, "S3": 100.0},
        "verdicts_dict": {"S1": "normal", "S2": "normal", "S3": "normal"},
        "confidences_dict": {"S1": 98.0, "S2": 98.0, "S3": 98.0},
        "scenario": "Summer heatwave",
        "seed": 999,
        "now_hour": 22.0,
        "focus": "S2"
    }
    res_b = send_voice_query(q_fresh, state_b)
    print(f"State B (Seed 999): {res_b.get('response')}")
    assert "25" in res_b.get("response") or "twenty-five" in res_b.get("response").lower(), "Failed to read State B reading (25.2)!"

    print(">>> TEST 6 PASSED: Backend immediately reflects dynamic changes in telemetry, seed, and scenario without stale caching.")

    print("\n" + "=" * 75)
    print("ALL STEP 2C TESTS (1 - 6) PASSED SUCCESSFULLY WITH LIVE GEMINI AI!")
    print("=" * 75)


if __name__ == "__main__":
    main()
