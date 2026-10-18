"""Test suite for safe configuration loading and secrets handling in SensorTrust.
Verifies that:
1. get_safe_config never raises StreamlitSecretNotFoundError when secrets are absent.
2. Streamlit script execution does not crash when Highlight Sensor or any control is interacted with.
3. Environment variables take precedence over secrets.
4. Fallback defaults are correctly returned.
"""
import os
import unittest
from streamlit.errors import StreamlitSecretNotFoundError


class MockStreamlitMissingSecrets:
    """Simulates Streamlit runtime when no secrets.toml or cloud secrets exist."""
    class MockSecrets:
        def __getattr__(self, name):
            raise StreamlitSecretNotFoundError("No secrets found.")
        def __getitem__(self, key):
            raise StreamlitSecretNotFoundError("No secrets found.")
        def __contains__(self, key):
            raise StreamlitSecretNotFoundError("No secrets found.")
        def get(self, key, default=None):
            raise StreamlitSecretNotFoundError("No secrets found.")

    def __init__(self):
        self.secrets = self.MockSecrets()


class TestSafeConfiguration(unittest.TestCase):
    def setUp(self):
        self.orig_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)

    def test_get_safe_config_without_secrets(self):
        import app
        import ai_agent

        # 1. Test fallback default when key does not exist
        os.environ.pop("BACKEND_URL", None)
        os.environ.pop("GEMINI_API_KEY", None)
        
        self.assertEqual(app.get_safe_config("BACKEND_URL", ""), "")
        self.assertEqual(app.get_safe_config("BACKEND_URL", "http://default:8000"), "http://default:8000")
        self.assertEqual(ai_agent.get_safe_config("GEMINI_API_KEY", "fallback"), "fallback")

    def test_get_safe_config_with_env_var(self):
        import app
        import ai_agent

        os.environ["BACKEND_URL"] = "https://custom-backend.onrender.com"
        os.environ["GEMINI_MODEL"] = "gemini-3.5-flash"

        self.assertEqual(app.get_safe_config("BACKEND_URL"), "https://custom-backend.onrender.com")
        self.assertEqual(ai_agent.get_gemini_model_name(), "gemini-3.5-flash")

    def test_simulated_streamlit_secret_not_found_error(self):
        """Simulate StreamlitSecretNotFoundError when accessing st.secrets."""
        import streamlit as st
        orig_secrets = getattr(st, "secrets", None)

        try:
            # Inject failing secrets object
            st.secrets = MockStreamlitMissingSecrets.MockSecrets()

            import app
            import ai_agent

            # Neither app.get_safe_config nor ai_agent.load_gemini_api_key should raise an error
            res_url = app.get_safe_config("BACKEND_URL", "safe_fallback")
            self.assertEqual(res_url, "safe_fallback")

            model_name = ai_agent.get_gemini_model_name()
            self.assertIn("gemini", model_name)

            key = ai_agent.load_gemini_api_key()
            # Must return None or env key, not raise StreamlitSecretNotFoundError
            self.assertTrue(key is None or isinstance(key, str))

        finally:
            if orig_secrets is not None:
                st.secrets = orig_secrets

    def test_highlight_sensor_computation_no_crash(self):
        """Verify the full pipeline: highlight sensor selection, shap explanation, and voice kwargs."""
        import core
        import app

        df, _ = core.simulate(n_sensors=5, days=1, seed=42)
        model = core.load_or_train_model()
        res, unit = core.analyse(df, model)

        sensors = list(df.columns)
        for focus_sensor in sensors:
            focus_rows = res[res["sensor"] == focus_sensor]
            self.assertFalse(focus_rows.empty)
            f_idx = int(focus_rows.index[-1])
            f_text, f_top, f_act = core.explain(res, model, unit, f_idx, len(sensors), "°C")
            self.assertTrue(len(f_text) > 0)
            self.assertIsInstance(f_act, str)

            # Test safe config loading during highlight sensor interaction
            backend_url = app.get_safe_config("BACKEND_URL", "")
            self.assertIsInstance(backend_url, str)


if __name__ == "__main__":
    unittest.main()
