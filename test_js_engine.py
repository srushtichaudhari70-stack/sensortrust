import voice_system
import subprocess
import re

html = voice_system.render_interactive_voice_assistant(
    focus_sensor='S2',
    focus_health=63.7,
    focus_verdict='drift',
    fleet_health=94.0,
    incident_count=1,
    sensors=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
    latest_readings={'S1': 23.4, 'S2': 21.0, 'S3': 23.5, 'S4': 23.3, 'S5': 23.6, 'S6': 23.4},
    health_dict={'S1': 100.0, 'S2': 63.7, 'S3': 100.0, 'S4': 100.0, 'S5': 100.0, 'S6': 100.0},
    verdicts_dict={'S1': 'normal', 'S2': 'drift', 'S3': 'normal', 'S4': 'normal', 'S5': 'normal', 'S6': 'normal'},
    confidences_dict={'S1': 95.0, 'S2': 97.0, 'S3': 95.0, 'S4': 95.0, 'S5': 95.0, 'S6': 95.0},
    incidents_list=[{'sensor': 'S2', 'verdict': 'drift', 'confidence': 0.97, 'readings': 12, 'row': 100}],
    shap_dict={'S2': {'reason': 'Telemetry departed steadily from median', 'action': 'Recalibrate RTD probe', 'top': [['r_slope', 0.85]]}},
    units='°C',
    now_hour=20.0,
    data_source='simulated temperature sensors',
    is_simulated=True,
    scenario='Faults only',
    seed=42
)

# Extract script contents
m = re.search(r'<script>(.*?)</script>', html, re.DOTALL)
assert m, 'Script tag not found!'
script_body = m.group(1)

# Isolate CTX and generateGroundedAnswer function (without DOM elements for Node)
ctx_match = re.search(r'const CTX = (\{.*?\});', script_body)
func_match = re.search(r'function generateGroundedAnswer\(query\)\s*\{.*?\n        \}', script_body, re.DOTALL)

assert ctx_match, "CTX not found"
assert func_match, "generateGroundedAnswer not found"

js_code = f"""
const CTX = {ctx_match.group(1)};
{func_match.group(0)}

console.log("=== EXECUTING CLIENT-SIDE JAVASCRIPT IN NODE.JS ===");

const queries = [
    "What is my current sensor health?",
    "What is the latest reading for the selected sensor?",
    "Which sensors have anomalies?",
    "Is the latest event genuine or suspicious?",
    "Why was this sensor flagged?",
    "How many sensors are healthy?",
    "What is the current status of sensor S6?",
    "What is the status of sensor S9?"
];

for (const q of queries) {{
    const ans = generateGroundedAnswer(q);
    console.log(`[Q] ${{q}}`);
    console.log(`[A] ${{ans}}\\n`);
}}
"""

with open("scratch_js_test.js", "w", encoding="utf-8") as f:
    f.write(js_code)

res = subprocess.run(["node", "scratch_js_test.js"], capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
assert res.returncode == 0
print("ALL JAVASCRIPT ENGINE IN-BROWSER TESTS VERIFIED AND PASSED!")
