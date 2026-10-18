"""SensorTrust Step 7 Integration Test Suite: Backend Connectivity & Production Alignment.

Verifies:
1. Backend health endpoint (/health and /api/health) returns valid status.
2. Production CORS headers allow 'https://srushtichaudhari70-stack.github.io' for GET and OPTIONS.
3. Backend /api/telemetry returns dynamic sensor snapshot with ML verdicts and health scores.
4. Backend /api/voice/ask processes dynamic sensor context safely.
5. Voice assistant HUD handles production origin checks, parent URL propagation, and storage fallbacks.
6. Downstream data consistency across scenarios, sensor counts, and seeds.
"""
import os
import json
import unittest
import urllib.request
import urllib.error


BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
PROD_ORIGIN = "https://srushtichaudhari70-stack.github.io"


class TestStep7BackendIntegration(unittest.TestCase):
    def test_01_backend_health_endpoint(self):
        """Verify the backend health check indicates ready status."""
        req = urllib.request.Request(f"{BACKEND_URL}/health")
        with urllib.request.urlopen(req, timeout=10) as res:
            self.assertEqual(res.status, 200)
            data = json.loads(res.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "healthy")
            self.assertEqual(data.get("service"), "SensorTrust Gemini AI Voice Backend")
            self.assertTrue(data.get("gemini_configured"))
            self.assertIn("gemini", data.get("model", "").lower())

    def test_02_production_cors_headers(self):
        """Verify CORS headers specifically permit the GitHub Pages production origin."""
        # Test GET CORS
        req_get = urllib.request.Request(
            f"{BACKEND_URL}/health",
            headers={"Origin": PROD_ORIGIN}
        )
        with urllib.request.urlopen(req_get, timeout=10) as res:
            self.assertEqual(res.status, 200)
            self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), PROD_ORIGIN)
            self.assertEqual(res.headers.get("Access-Control-Allow-Credentials"), "true")

        # Test Pre-flight OPTIONS CORS
        req_options = urllib.request.Request(
            f"{BACKEND_URL}/api/voice/ask",
            headers={
                "Origin": PROD_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            },
            method="OPTIONS"
        )
        with urllib.request.urlopen(req_options, timeout=10) as res:
            self.assertEqual(res.status, 200)
            self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), PROD_ORIGIN)
            methods = res.headers.get("Access-Control-Allow-Methods", "")
            self.assertIn("POST", methods)
            self.assertIn("OPTIONS", methods)

    def test_03_backend_telemetry_endpoint(self):
        """Verify the /api/telemetry endpoint returns live readings, verdicts, and health scores."""
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/telemetry?n_sensors=5&seed=42&days=1",
            headers={"Origin": PROD_ORIGIN}
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            self.assertEqual(res.status, 200)
            data = json.loads(res.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "ok")
            self.assertEqual(data.get("n_sensors"), 5)
            self.assertEqual(len(data.get("sensors", [])), 5)
            latest = data.get("latest", {})
            for s in data["sensors"]:
                self.assertIn(s, latest)
                self.assertIn("reading", latest[s])
                self.assertIn("health", latest[s])
                self.assertIn("verdict", latest[s])
                self.assertIn("confidence", latest[s])

    def test_04_live_voice_ask_with_telemetry_context(self):
        """Verify the backend answers a question using dynamic sensor context."""
        payload = {
            "query": "Which sensor has the lowest health score right now?",
            "context": {
                "sensors": ["S1", "S2", "S3"],
                "health_dict": {"S1": 98.0, "S2": 42.0, "S3": 95.0},
                "verdicts_dict": {"S1": "normal", "S2": "drift", "S3": "normal"},
                "focus_sensor": "S2",
                "focus_health": 42.0,
                "fleet_health": 78.3
            }
        }
        req = urllib.request.Request(
            f"{BACKEND_URL}/api/voice/ask",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Origin": PROD_ORIGIN},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            self.assertEqual(res.status, 200)
            data = json.loads(res.read().decode("utf-8"))
            self.assertIn("response", data)
            self.assertEqual(data.get("status"), "ok")
            # The answer must correctly reference sensor S2 as having the lowest health
            self.assertIn("S2", data["response"])

    def test_05_voice_system_parent_propagation(self):
        """Verify voice_system.py includes parent frame detection and try-catch storage handling."""
        import voice_system
        hud = voice_system.render_interactive_voice_assistant(
            focus_sensor="S1", focus_health=95.0, focus_verdict="normal",
            fleet_health=95.0, incident_count=0, sensors=["S1", "S2"],
            latest_readings={"S1": 21.0, "S2": 21.5}, health_dict={"S1": 95.0, "S2": 95.0},
            verdicts_dict={"S1": "normal", "S2": "normal"}, confidences_dict={"S1": 99.0, "S2": 99.0},
            incidents_list=[], shap_dict={}, units="°C", now_hour=10.0,
            data_source="simulated", is_simulated=True, scenario="Clean", seed=42,
            backend_url=None
        )
        self.assertIn("window.parent.SENSORTRUST_BACKEND_URL", hud)
        self.assertIn("try { localStorage.setItem('sensortrust_backend_url'", hud.replace("\n", " "))

    def test_06_downstream_data_consistency(self):
        """Verify that varying seed and sensor count produces distinct, consistent downstream states."""
        import core
        model = core.load_or_train_model()

        # State A
        df_a, _ = core.simulate(n_sensors=4, seed=10, days=1)
        res_a, _ = core.analyse(df_a, model)

        # State B
        df_b, _ = core.simulate(n_sensors=6, seed=20, days=1)
        res_b, _ = core.analyse(df_b, model)

        self.assertNotEqual(len(df_a.columns), len(df_b.columns))
        self.assertNotEqual(float(df_a["S1"].iloc[-1]), float(df_b["S1"].iloc[-1]))


if __name__ == "__main__":
    unittest.main()
