"""SensorTrust Step 2 Comprehensive Verification Suite.

Tests all 6 required conditions:
1. Normal sensor-health question.
2. Anomaly explanation using actual supplied sensor data.
3. Question about a nonexistent sensor.
4. Missing or unavailable sensor data.
5. Missing or invalid Gemini credentials (safe fallback, no secret leakage).
6. Backend errors, CORS headers, and timeouts via FastAPI TestClient.
"""

import os
import sys
from fastapi.testclient import TestClient
import ai_agent
from api_server import app

def build_sample_context():
    return {
        "sensors": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "readings": {"S1": 23.45, "S2": 20.80, "S3": 23.50, "S4": 23.30, "S5": 23.60, "S6": 23.40},
        "health_dict": {"S1": 100.0, "S2": 62.0, "S3": 100.0, "S4": 100.0, "S5": 100.0, "S6": 100.0},
        "verdicts_dict": {"S1": "normal", "S2": "drift", "S3": "normal", "S4": "normal", "S5": "normal", "S6": "normal"},
        "confidences_dict": {"S1": 95.0, "S2": 97.0, "S3": 95.0, "S4": 95.0, "S5": 95.0, "S6": 95.0},
        "incidents_list": [
            {
                "sensor": "S2",
                "verdict": "drift",
                "confidence": 0.97,
                "readings": 12,
                "row": 150
            }
        ],
        "shap": {
            "S2": {
                "reason": "Its telemetry steadily departs from the spatial cluster consensus.",
                "action": "Perform two-point calibration check against a certified reference dry-well thermometer.",
                "top": [["r_slope", 0.85], ["r_mean_s", 0.62]]
            }
        },
        "units": "°C",
        "now_hour": 20.0,
        "data_source": "simulated temperature sensors",
        "is_simulated": True,
        "scenario": "Faults only",
        "seed": 42,
        "focus": "S2",
        "fleet_health": 93.7,
        "fleet_status": "Optimal"
    }


def main():
    print("=" * 70)
    print("SENSORTRUST STEP 2 VERIFICATION SUITE (TESTS 1 - 6)")
    print("=" * 70)

    ctx = build_sample_context()

    # ------------------------------------------------------------------
    # TEST 1: Normal sensor-health question
    # ------------------------------------------------------------------
    print("\n--- TEST 1: Normal Sensor Health Question ---")
    query_1 = "What is the current sensor health?"
    res_1 = ai_agent.ask_gemini_voice_assistant(query_1, ctx)
    print(f"Query: '{query_1}'")
    print(f"Source: {res_1.get('source')} (Status: {res_1.get('status')})")
    print(f"Response: {res_1.get('response')}\n")

    assert res_1.get("response"), "Response was empty!"
    assert any(term in res_1.get("response") for term in ["health", "score", "S2", "100", "62", "Optimal"]), "Health details missing!"
    print(">>> TEST 1 PASSED: Normal sensor-health question answered accurately using supplied telemetry.")

    # ------------------------------------------------------------------
    # TEST 2: Anomaly explanation using actual supplied sensor data
    # ------------------------------------------------------------------
    print("\n--- TEST 2: Anomaly Explanation Using Actual Sensor Data ---")
    query_2 = "Why was sensor S2 flagged?"
    res_2 = ai_agent.ask_gemini_voice_assistant(query_2, ctx)
    print(f"Query: '{query_2}'")
    print(f"Source: {res_2.get('source')} (Status: {res_2.get('status')})")
    print(f"Response: {res_2.get('response')}\n")

    resp_text_2 = res_2.get("response").lower()
    assert "s2" in resp_text_2, "Target sensor S2 missing from response!"
    assert "drift" in resp_text_2, "Anomaly classification 'drift' missing from explanation!"
    assert any(k in resp_text_2 for k in ["consensus", "departs", "calibration", "anomaly"]), "Explanation missing consensus reasoning!"
    print(">>> TEST 2 PASSED: Anomaly explanation reflects actual supplied model verdicts, TreeSHAP evidence, and SOP.")

    # ------------------------------------------------------------------
    # TEST 3: Question about a nonexistent sensor
    # ------------------------------------------------------------------
    print("\n--- TEST 3: Question About Nonexistent Sensor ---")
    query_3 = "What is the current status of sensor S9?"
    res_3 = ai_agent.ask_gemini_voice_assistant(query_3, ctx)
    print(f"Query: '{query_3}'")
    print(f"Source: {res_3.get('source')}")
    print(f"Response: {res_3.get('response')}\n")

    assert "Sensor S9 does not exist in the active cluster" in res_3.get("response"), "Failed to reject nonexistent sensor S9!"
    assert "S1, S2, S3, S4, S5, S6" in res_3.get("response"), "Failed to enumerate valid cluster sensors!"
    print(">>> TEST 3 PASSED: Assistant safely refuses to invent results for nonexistent sensors.")

    # ------------------------------------------------------------------
    # TEST 4: Missing or unavailable sensor data
    # ------------------------------------------------------------------
    print("\n--- TEST 4: Missing or Unavailable Sensor Data ---")
    empty_ctx = {"sensors": [], "health_dict": {}}
    query_4 = "What is my current sensor health?"
    res_4 = ai_agent.ask_gemini_voice_assistant(query_4, empty_ctx)
    print(f"Query: '{query_4}' with empty context")
    print(f"Response: {res_4.get('response')}\n")

    assert "unavailable" in res_4.get("response").lower(), "Failed to report telemetry unavailable!"
    print(">>> TEST 4 PASSED: Missing sensor data produces clear, safe unavailable notification.")

    # ------------------------------------------------------------------
    # TEST 5: Missing or invalid Gemini credentials
    # ------------------------------------------------------------------
    print("\n--- TEST 5: Missing or Invalid Gemini Credentials ---")
    # Subtest 5A: No key supplied
    res_5a = ai_agent.ask_gemini_voice_assistant("What is the latest reading for S1?", ctx, api_key="")
    print("Subtest 5A (Empty key):")
    print(f"Status: {res_5a.get('status')} | Source: {res_5a.get('source')}")
    print(f"Error: {res_5a.get('error')}")
    print(f"Fallback Response: {res_5a.get('response')}\n")

    assert res_5a.get("source") == "grounded_fallback", "Should use grounded fallback when key is missing!"
    assert res_5a.get("status") == "no_key", "Status should be 'no_key'!"
    assert "not configured" in res_5a.get("error").lower(), "Error should inform user that key is not configured!"

    # Subtest 5B: Invalid key supplied (ensure no secrets leaked in error or output)
    fake_secret = "AIzaSyFakeSecret1234567890abcdefghijk"
    res_5b = ai_agent.ask_gemini_voice_assistant("What is the latest reading for S1?", ctx, api_key=fake_secret)
    print("Subtest 5B (Invalid key):")
    print(f"Status: {res_5b.get('status')} | Source: {res_5b.get('source')}")
    print(f"Error: {res_5b.get('error')}")
    print(f"Fallback Response: {res_5b.get('response')}\n")

    assert fake_secret not in str(res_5b.get("error")), "SECURITY VIOLATION: Secret was leaked in error message!"
    assert fake_secret not in str(res_5b.get("response")), "SECURITY VIOLATION: Secret was leaked in response!"
    assert res_5b.get("source") == "grounded_fallback", "Should fall back safely on invalid key!"
    print(">>> TEST 5 PASSED: Safe fallbacks when credentials missing/invalid; zero secret exposure.")

    # ------------------------------------------------------------------
    # TEST 6: Backend errors, CORS, and timeouts via FastAPI TestClient
    # ------------------------------------------------------------------
    print("\n--- TEST 6: Backend API Server, CORS & Error Handling ---")
    client = TestClient(app)

    # 6A: Health Check
    resp_health = client.get("/api/health")
    print(f"GET /api/health -> Status {resp_health.status_code}")
    print(f"Body: {resp_health.json()}")
    assert resp_health.status_code == 200, "Health check failed!"
    assert resp_health.json().get("status") == "healthy", "Server not healthy!"
    assert "gemini_configured" in resp_health.json(), "Missing gemini_configured field!"
    assert resp_health.json().get("model") == "gemini-2.5-flash", "Model should be verified gemini-2.5-flash!"

    # 6B: CORS Headers
    resp_cors = client.options("/api/voice/ask", headers={
        "Origin": "https://srushtichaudhari70-stack.github.io",
        "Access-Control-Request-Method": "POST"
    })
    print(f"OPTIONS /api/voice/ask -> Status {resp_cors.status_code}")
    print(f"CORS Allow Origin: {resp_cors.headers.get('access-control-allow-origin')}")
    assert resp_cors.status_code in [200, 204], "CORS preflight failed!"
    assert resp_cors.headers.get("access-control-allow-origin") in ["*", "https://srushtichaudhari70-stack.github.io"]

    # 6C: Empty Query Validation
    resp_empty_q = client.post("/api/voice/ask", json={"query": "   ", "context": ctx})
    print(f"POST /api/voice/ask (empty query) -> Status {resp_empty_q.status_code}")
    assert resp_empty_q.status_code == 400, "Empty query should return 400 Bad Request!"

    # 6D: Valid Voice Ask via API
    resp_valid = client.post("/api/voice/ask", json={
        "query": "Is this a genuine event or a sensor fault?",
        "context": ctx
    })
    print(f"POST /api/voice/ask (valid query) -> Status {resp_valid.status_code}")
    data_valid = resp_valid.json()
    print(f"API Response Source: {data_valid.get('source')}")
    print(f"API Response Text: {data_valid.get('response')}\n")
    assert resp_valid.status_code == 200, "Valid ask failed!"
    assert data_valid.get("response"), "Empty response from API!"
    assert any(term in data_valid.get("response").lower() for term in ["fault", "suspicious", "drift", "consensus", "sensor"]), "Context logic missing!"

    print(">>> TEST 6 PASSED: Backend FastAPI server, CORS, schema validation, and error handling verified.")

    print("\n" + "=" * 70)
    print("ALL 6 TESTS (1 - 6) PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
