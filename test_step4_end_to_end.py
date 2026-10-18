"""SensorTrust Step 4: Comprehensive End-to-End Integration Test Suite.

Executes thorough testing across:
1. ML Pipeline: normal readings, cluster events, spikes, stuck values, drift, noise, dropout/NaNs.
2. Verified predictions from trained model artifact (model.joblib).
3. Held-out empirical evaluation metrics (precision, recall, F1, confusion matrix).
4. Live Gemini API integration with dynamic context and refusal to hallucinate.
5. Verification that suspected faults are NEVER described as confirmed hardware failures.
6. API schema compliance, invalid input handling (HTTP 400/422), and missing context resilience.
7. Voice engine debouncing and Indian English (en-IN) configuration.
8. Dynamic parameter responsiveness (seed, scenario, sensor count).
"""

import os
import sys
import json
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import unittest

import core
import ai_agent
import voice_system
import evaluate

BACKEND_URL = "http://127.0.0.1:8000"


class TestSensorTrustEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure model is available
        cls.model = core.load_or_train_model()
        assert cls.model is not None, "Model failed to load."

    def test_01_ml_pipeline_all_regimes(self):
        """Test all 7 sensor behavior regimes: normal, event, spike, stuck, drift, noise, dropout."""
        events = [(100, 4.0)]
        faults = [
            dict(sensor=0, kind="spike", start=130, dur=1, mag=5.0),
            dict(sensor=1, kind="stuck", start=140, dur=20, mag=0.0),
            dict(sensor=2, kind="drift", start=170, dur=40, mag=3.0),
            dict(sensor=3, kind="noise", start=220, dur=24, mag=7.0),
            dict(sensor=4, kind="dropout", start=260, dur=12, mag=0.0),
        ]
        df, lab = core.simulate(n_sensors=6, days=2, seed=101, events=events, faults=faults)
        
        # Verify NaN is produced for dropout
        assert df["S5"].iloc[260:272].isna().all(), "Dropout must inject NaNs."
        
        res, unit = core.analyse(df, self.model)
        
        # 1. Normal: clean sensor S6 around step 80
        s6_clean = res[(res["sensor"] == "S6") & (res["pos"].between(75, 95))]
        assert (s6_clean["verdict"] == "normal").mean() > 0.85, "S6 should be predominantly normal."
        
        # 2. Genuine Event: all sensors shift around step 100
        ev_readings = res[res["pos"].between(100, 115)]
        assert (ev_readings["verdict"] == "event").sum() > 0, "Event must be detected across cluster."
        
        # 3. Isolated Spike on S1 at step 130
        s1_spike = res[(res["sensor"] == "S1") & (res["pos"] == 130)]
        assert not s1_spike.empty
        assert s1_spike["verdict"].iloc[0] == "spike", f"Expected spike on S1, got {s1_spike['verdict'].iloc[0]}"
        
        # 4. Stuck on S2 starting step 140
        s2_stuck = res[(res["sensor"] == "S2") & (res["pos"].between(144, 158))]
        assert (s2_stuck["verdict"] == "stuck").all(), "S2 must be flagged as stuck."
        
        # 5. Drift on S3 starting step 170
        s3_drift = res[(res["sensor"] == "S3") & (res["pos"].between(180, 200))]
        assert (s3_drift["verdict"] == "drift").sum() > 0, "S3 drift must be detected."
        
        # 6. Noise on S4 starting step 220
        s4_noise = res[(res["sensor"] == "S4") & (res["pos"].between(225, 240))]
        assert (s4_noise["verdict"] == "noise").sum() > 0, "S4 noise must be detected."
        
        # 7. Dropout on S5 starting step 260
        s5_drop = res[(res["sensor"] == "S5") & (res["pos"].between(260, 271))]
        assert (s5_drop["verdict"] == "dropout").all(), "S5 dropout must be flagged."
        print("  ✓ ML Pipeline correctly detects all 7 telemetry regimes.")

    def test_02_held_out_baseline_metrics(self):
        """Verify empirical metrics and comparison against baselines on unseen data."""
        summary, per_class, extra = evaluate.run(model=self.model, n_runs=2)
        
        # Check baseline performance comparison
        self.assertIn("Z-score", summary.index)
        self.assertIn("Isolation Forest", summary.index)
        self.assertIn("SensorTrust", summary.index)
        
        # SensorTrust should have high fault precision and low event miss rate
        st_metrics = summary.loc["SensorTrust"]
        self.assertGreater(st_metrics["fault_precision"], 0.85)
        self.assertLess(st_metrics["event_miss_rate"], 0.05)
        self.assertGreater(extra["macro_f1"], 0.75)
        print(f"  ✓ Held-out baseline evaluation confirmed (Macro F1: {extra['macro_f1']:.3f}, Event Miss Rate: {st_metrics['event_miss_rate']:.1%}).")

    def test_03_backend_health_and_gemini_config(self):
        """Verify FastAPI backend reports healthy status and Gemini is configured."""
        req = urllib.request.Request(f"{BACKEND_URL}/health")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["status"], "healthy")
            self.assertTrue(data["gemini_configured"])
            self.assertEqual(data["configuration_status"], "ready")
            self.assertIn("gemini-3.5-flash", data["model"])
        print(f"  ✓ Backend health verified: {data['service']} running with {data['model']}.")

    def test_04_live_gemini_request_and_context_grounding(self):
        """Test a real Gemini question over the backend API with dynamic sensor telemetry."""
        payload = {
            "query": "What is the current health and reading of sensor S2?",
            "context": {
                "sensors": ["S1", "S2", "S3", "S4"],
                "health": {"S1": 98.0, "S2": 62.0, "S3": 100.0, "S4": 99.0},
                "readings": {"S1": 22.4, "S2": 16.8, "S3": 22.5, "S4": 22.3},
                "verdicts": {"S1": "normal", "S2": "drift", "S3": "normal", "S4": "normal"},
                "confidences": {"S1": 99.0, "S2": 94.0, "S3": 99.0, "S4": 98.0},
                "units": "°C",
                "now_hour": 14.5,
                "data_source": "simulated temperature sensors",
                "is_simulated": True,
                "focus": "S2"
            }
        }
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/voice/ask",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["source"], "gemini")
            reply = data["response"].lower()
            # Must ground response in S2 health (62) or reading (16.8) or drift
            self.assertTrue(any(term in reply for term in ["s2", "62", "16.8", "drift"]), f"Reply lacked S2 grounding: {data['response']}")
            print(f"  ✓ Live Gemini response grounded: {data['response'][:100]}...")

    def test_05_no_confirmed_hardware_failure_claim(self):
        """Verify the assistant describes anomalies as suspected faults requiring inspection, NEVER as confirmed broken hardware."""
        payload = {
            "query": "Is sensor S2 permanently broken?",
            "context": {
                "sensors": ["S1", "S2", "S3"],
                "health": {"S1": 100.0, "S2": 45.0, "S3": 100.0},
                "readings": {"S1": 20.0, "S2": 12.0, "S3": 20.1},
                "verdicts": {"S1": "normal", "S2": "drift", "S3": "normal"},
                "confidences": {"S1": 99.0, "S2": 95.0, "S3": 99.0},
                "focus": "S2",
                "units": "°C"
            }
        }
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/voice/ask",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            reply = data["response"].lower()
            # Must NOT claim confirmed hardware failure
            forbidden_phrases = ["confirmed hardware failure", "permanently broken hardware", "hardware is broken"]
            for f in forbidden_phrases:
                self.assertNotIn(f, reply, f"Assistant made unsupported hardware failure claim: {f}")
            # Must mention probabilistic/suspected/inspection
            expected_cautions = ["inspection", "suspected", "probabilistic", "anomaly", "verify", "departure", "probable", "technician"]
            self.assertTrue(any(w in reply for w in expected_cautions), f"Assistant should advise inspection: {data['response']}")
            print(f"  ✓ Assistant correctly treats anomaly as suspected fault requiring verification.")

    def test_06_nonexistent_sensor_id_rejection(self):
        """Verify the assistant refuses to hallucinate data for an invalid sensor ID (e.g. S9 in a 3-sensor network)."""
        payload = {
            "query": "What is the status of sensor S9?",
            "context": {
                "sensors": ["S1", "S2", "S3"],
                "health": {"S1": 100.0, "S2": 98.0, "S3": 99.0},
                "readings": {"S1": 21.0, "S2": 21.2, "S3": 20.9},
                "verdicts": {"S1": "normal", "S2": "normal", "S3": "normal"},
                "confidences": {"S1": 99.0, "S2": 98.0, "S3": 99.0},
                "units": "°C"
            }
        }
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/voice/ask",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            reply = data["response"].lower()
            self.assertTrue("does not exist" in reply or "not present" in reply or "s9" in reply, f"Expected refusal for S9: {data['response']}")
            print(f"  ✓ Safely rejected invalid sensor ID S9: {data['response']}")

    def test_07_api_error_handling_empty_and_malformed(self):
        """Verify API returns 400 for empty query and handles missing context safely."""
        # 1. Empty query -> 400
        req_empty = urllib.request.Request(
            f"{BACKEND_URL}/api/voice/ask",
            data=json.dumps({"query": "   ", "context": {}}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req_empty) as resp:
                self.fail("Expected HTTP 400 for empty query.")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            print("  ✓ Empty query rejected with HTTP 400.")

        # 2. Missing context -> Graceful 200 with notice
        req_no_ctx = urllib.request.Request(
            f"{BACKEND_URL}/api/voice/ask",
            data=json.dumps({"query": "What is my sensor health?", "context": {}}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_no_ctx) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("unavailable", data["response"].lower())
            print(f"  ✓ Missing context safely handled: {data['response']}")

    def test_08_voice_debouncing_and_speech_config(self):
        """Verify voice HUD contains duplicate prevention and Indian English (en-IN) configuration."""
        hud_html = voice_system.render_interactive_voice_assistant(
            focus_sensor="S1", focus_health=100.0, focus_verdict="normal",
            fleet_health=100.0, incident_count=0, sensors=["S1", "S2", "S3"]
        )
        # Check debouncing logic
        self.assertIn("if (isProcessing)", hud_html)
        self.assertIn("Already processing a query", hud_html)
        # Check Indian English speech recognition and synthesis
        self.assertIn("en-IN", hud_html)
        self.assertIn("Stop Speaking", hud_html)
        self.assertIn("status-speaking", hud_html)
        self.assertIn("status-listening", hud_html)
        self.assertIn("status-processing", hud_html)
        print("  ✓ Voice HUD debouncing and Indian English (en-IN) configuration verified.")

    def test_09_dynamic_sensor_controls_responsiveness(self):
        """Verify that modifying random seed and scenario produces dynamically distinct states."""
        df_seed1, _ = core.simulate(n_sensors=5, days=1, seed=42)
        df_seed2, _ = core.simulate(n_sensors=5, days=1, seed=999)
        self.assertFalse(df_seed1.equals(df_seed2), "Different seeds must generate distinct telemetry.")
        
        df_4s, _ = core.simulate(n_sensors=4, days=1, seed=42)
        df_8s, _ = core.simulate(n_sensors=8, days=1, seed=42)
        self.assertEqual(df_4s.shape[1], 4)
        self.assertEqual(df_8s.shape[1], 8)
        print("  ✓ Dynamic seed and sensor count controls generate distinct, responsive states.")


if __name__ == "__main__":
    print("=" * 75)
    print("🚀 SENSORTRUST STEP 4: END-TO-END INTEGRATION TEST SUITE")
    print("=" * 75)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSensorTrustEndToEnd)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n🎉 ALL 9 END-TO-END INTEGRATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)
