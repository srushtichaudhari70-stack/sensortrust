"""SensorTrust dashboard with AI Copilot & Root-Cause Analysis.
Run with: streamlit run app.py
"""
import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core
import ai_agent

st.set_page_config(page_title="SensorTrust · AI Sensor Intelligence", page_icon="🛡️", layout="wide")
H = core.PER_HOUR
DAYS = 2
N = DAYS * core.PER_DAY

try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False
INK = "#ffffff" if DARK else "#0b0b0b"
INK2 = "#c3c2b7" if DARK else "#52514e"
MUTED = "#898781"
GRID = "#2c2c2a" if DARK else "#e1e0d9"
OTHER_LINE = "#4a4a47" if DARK else "#c3c2b7"

VERDICT_STYLE = {
    "event":   ("#199e70" if DARK else "#1baf7a", "circle"),
    "spike":   ("#e66767" if DARK else "#e34948", "diamond"),
    "drift":   ("#d95926" if DARK else "#eb6834", "triangle-up"),
    "stuck":   ("#9085e9" if DARK else "#4a3aa7", "square"),
    "noise":   ("#c98500" if DARK else "#eda100", "x"),
    "bias":    ("#3987e5" if DARK else "#2a78d6", "triangle-down"),
    "dropout": ("#d55181" if DARK else "#e87ba4", "cross"),
    core.UNCERTAIN: (MUTED, "circle-open"),
}
VERDICT_LABEL = {
    "event": "genuine event", "spike": "spike (fault)", "drift": "drift (fault)",
    "stuck": "stuck (fault)", "noise": "noise (fault)", "bias": "bias (fault)",
    "dropout": "dropout (fault)", core.UNCERTAIN: "needs review"
}
BAND_COLOR = {"Healthy": "green", "Watch": "orange", "Service it": "red"}

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


@st.cache_resource(show_spinner="Training LightGBM model on simulated data (about 15 seconds, once)...")
def get_model():
    return core.train_model()


@st.cache_data(show_spinner="Analysing readings...")
def run_simulated(n_sensors, seed, events, faults):
    fl = [dict(sensor=s, kind=k, start=a, dur=d, mag=m) for s, k, a, d, m in faults if s < n_sensors]
    df, _ = core.simulate(n_sensors=n_sensors, days=DAYS, seed=seed, events=list(events), faults=fl)
    res, unit = core.analyse(df, get_model())
    return df, res, unit


@st.cache_data(show_spinner="Analysing your file...")
def run_uploaded(raw: bytes):
    df = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    res, unit = core.analyse(df, get_model())
    return df, res, unit


@st.cache_data(show_spinner="Testing on held-out simulated data (about 40 seconds)...")
def run_evaluation():
    import evaluate
    return evaluate.run(get_model())


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛡️ SensorTrust")
st.sidebar.caption("AI-Powered Sensor Fault Intelligence & Root-Cause Diagnostics.")

# AI Integration settings
st.sidebar.subheader("🤖 AI Engine")
gemini_key = st.sidebar.text_input(
    "Gemini API Key (Optional)",
    type="password",
    value=os.environ.get("GEMINI_API_KEY", ""),
    help="Enter Google Gemini API key to activate Gemini 3.8 Flash online reasoning. Leave blank for built-in expert AI."
)
if gemini_key:
    st.sidebar.success("✨ Gemini 3.8 Flash: Connected")
else:
    st.sidebar.info("💡 Mode: Built-in Expert AI (Zero-key)")

preset = st.sidebar.selectbox("Scenario", list(PRESETS))
n_sensors = st.sidebar.slider("Number of sensors", 4, 10, 8)
seed = st.sidebar.number_input("Random seed", 0, 9999, 7, help="Changes the simulated weather and noise.")

st.sidebar.subheader("Break something yourself")
if "extra" not in st.session_state:
    st.session_state.extra = []
kind = st.sidebar.selectbox("What to add", ["genuine event (all sensors)"] + core.FAULTS)
target = st.sidebar.selectbox("Sensor", [f"S{i + 1}" for i in range(n_sensors)],
                              disabled=kind.startswith("genuine"))
start_h = st.sidebar.slider("Starts at hour", 8, 44, 24)
size = st.sidebar.slider("Size (°C, or noise multiplier)", -6.0, 8.0, 3.0, 0.5,
                         help="Ignored for stuck and dropout.")
dur_h = st.sidebar.slider("Lasts (hours)", 0.5, 8.0, 3.0, 0.5, help="Spikes always last one reading.")
c1, c2 = st.sidebar.columns(2)
if c1.button("Add", width="stretch"):
    st.session_state.extra.append((kind, int(target[1:]) - 1, start_h * H, size, int(dur_h * H)))
if c2.button("Clear", width="stretch"):
    st.session_state.extra = []
for k_, s_, a_, m_, d_ in st.session_state.extra:
    st.sidebar.caption(f"+ {k_} at hour {a_ // H}" + ("" if k_.startswith("genuine") else f" on S{s_ + 1}"))

upload = st.sidebar.file_uploader("Or analyse your own CSV", type="csv",
                                  help="First column: timestamp. Other columns: one per sensor (>= 3 sensors).")

# ------------------------------------------------------------------ data
if upload is not None:
    df, res, unit = run_uploaded(upload.getvalue())
    units, source = "units", f"your file ({upload.name})"
    if df.shape[1] < 3:
        st.error("The file needs at least 3 sensor columns so sensors can be compared with each other.")
        st.stop()
else:
    events = list(PRESETS[preset]["events"])
    faults = list(PRESETS[preset]["faults"])
    for k_, s_, a_, m_, d_ in st.session_state.extra:
        if k_.startswith("genuine"):
            events.append((a_, m_ if m_ else 3.0))
        else:
            faults.append((s_, k_, a_, 1 if k_ == "spike" else d_, abs(m_) if k_ == "noise" else m_))
    df, res, unit = run_simulated(n_sensors, int(seed), tuple(events), tuple(faults))
    units, source = "°C", "simulated temperature sensors"
sensors = list(df.columns)
n = len(df)

st.title("🛡️ SensorTrust: AI-Integrated Sensor Intelligence")
st.caption(f"Sensor health & anomaly diagnostics · Data source: {source} · Readings every {core.STEP_MIN} min")

last_h = (n - 1) / H
now_h = st.slider("⏱️ Timeline Replay (Hours)", float(core.WARMUP // H), float(round(last_h, 1)),
                  float(round(last_h, 1)), 0.5)
now = min(n - 1, int(now_h * H))
view = res[res["pos"] <= now]
inc = core.incidents(res, upto=now)

# ------------------------------------------------------------------ health cards
latest = view[view["pos"] == now].set_index("sensor")
cols = st.columns(len(sensors))
health_dict = {}
for col, s in zip(cols, sensors):
    score = float(latest.loc[s, "health"])
    health_dict[s] = score
    band = core.health_band(score)
    v = latest.loc[s, "verdict"]
    with col:
        st.metric(s, f"{score:.0f}", border=True, help="Health score: 0 (failed) to 100 (healthy)")
        st.markdown(f":{BAND_COLOR[band]}[**{band}**]  \n`{VERDICT_LABEL.get(v, v)}`")

focus = st.selectbox("🎯 Highlight Sensor", sensors,
                     index=int(np.argmin([latest.loc[s, "health"] for s in sensors])))


def base_layout(title, ytitle, height=360):
    return dict(title=dict(text=title, font=dict(size=15, color=INK)), height=height,
                margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=12), hovermode="closest",
                legend=dict(orientation="h", y=-0.18, x=0),
                xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
                yaxis=dict(title=ytitle, gridcolor=GRID, linecolor=GRID, zeroline=False))


# ------------------------------------------------------------------ navigation tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Telemetry & Health",
    "🤖 AI Root-Cause Analysis & SOP",
    "💬 AI Sensor Copilot (Chat)",
    "🔬 Model Benchmarks"
])

# ------------------------------------------------------------------ TAB 1: Telemetry & Health
with tab1:
    shown = df.iloc[:now + 1]
    fig = go.Figure()
    for s in sensors:
        if s != focus:
            fig.add_trace(go.Scatter(x=shown.index, y=shown[s], mode="lines", name="other sensors",
                                     legendgroup="others", showlegend=(s == [x for x in sensors if x != focus][0]),
                                     line=dict(color=OTHER_LINE, width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=shown.index, y=shown[focus], mode="lines", name=focus,
                             line=dict(color=INK, width=2.5),
                             hovertemplate=f"{focus}: %{{y:.2f}} {units}<br>%{{x|%d %b %H:%M}}<extra></extra>"))
    filled = df.ffill().bfill()
    flag = view[view["verdict"] != "normal"].copy()
    flag["y"] = [filled.iloc[p][s] for p, s in zip(flag["pos"], flag["sensor"])]
    for v, (color, symbol) in VERDICT_STYLE.items():
        g = flag[flag["verdict"] == v]
        if len(g):
            fig.add_trace(go.Scatter(
                x=g["time"], y=g["y"], mode="markers", name=VERDICT_LABEL[v],
                marker=dict(color=color, symbol=symbol, size=8, line=dict(width=1, color=color)),
                customdata=np.c_[g["sensor"], (g["confidence"] * 100).round()],
                hovertemplate="%{customdata[0]} · " + VERDICT_LABEL[v] +
                              "<br>confidence %{customdata[1]:.0f}%<br>%{x|%d %b %H:%M}<extra></extra>"))
    fig.update_layout(**base_layout("Cluster Readings: Multi-sensor cross-validation & flagged anomalies", units))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Detected Incidents")
        if inc.empty:
            st.info("Nominal condition: All sensors agree with neighbor consensus.")
        else:
            table = inc.assign(verdict=inc["verdict"].map(VERDICT_LABEL),
                               duration=(inc["readings"] * core.STEP_MIN).astype(str) + " min",
                               confidence=(inc["confidence"] * 100).round().astype(int).astype(str) + "%",
                               start=inc["start"].dt.strftime("%d %b %H:%M"))
            st.dataframe(table[["sensor", "start", "duration", "verdict", "confidence"]],
                         width="stretch", hide_index=True, height=270)
    with right:
        st.subheader("TreeSHAP Explainability")
        if inc.empty:
            st.write("Pick a scenario with faults, or inject an anomaly via the sidebar.")
        else:
            labels = [f"{r.sensor} · {VERDICT_LABEL[r.verdict]} · {r.start:%d %b %H:%M}" for r in inc.itertuples()]
            default = next((i for i, r in enumerate(inc.itertuples()) if r.sensor == focus), 0)
            pick = st.selectbox("Inspect Incident", range(len(labels)), index=default, format_func=lambda i: labels[i])
            row = inc.iloc[pick]
            reason, top, action = core.explain(res, get_model(), unit, int(row["row"]), len(sensors), units)
            st.markdown(f"**{row['sensor']}: {VERDICT_LABEL[row['verdict']]}**")
            st.write(reason)
            st.markdown(f"**Recommended action:** {action}")
            if top and not res.loc[int(row["row"]), "rule"]:
                bar = go.Figure(go.Bar(x=[t[1] for t in top][::-1], y=[t[0] for t in top][::-1], orientation="h",
                                       marker=dict(color="#3987e5" if DARK else "#2a78d6"), width=0.5,
                                       hovertemplate="%{y}<br>weight %{x:.2f}<extra></extra>"))
                lay = base_layout("Feature Importance (TreeSHAP Attribution)", "", height=210)
                lay["xaxis"]["title"] = "Weight of evidence (SHAP)"
                lay["yaxis"]["title"] = None
                bar.update_layout(**lay)
                st.plotly_chart(bar, width="stretch")

    # Health trajectory
    hfig = go.Figure()
    for s in sensors:
        g = view[view["sensor"] == s]
        if s != focus:
            hfig.add_trace(go.Scatter(x=g["time"], y=g["health"], mode="lines", name="other sensors",
                                      legendgroup="others", showlegend=(s == [x for x in sensors if x != focus][0]),
                                      line=dict(color=OTHER_LINE, width=1), hovertemplate=f"{s}: %{{y:.0f}}<extra></extra>"))
    g = view[view["sensor"] == focus]
    hfig.add_trace(go.Scatter(x=g["time"], y=g["health"], mode="lines", name=focus, line=dict(color=INK, width=2.5),
                              hovertemplate=f"{focus}: %{{y:.0f}}<br>%{{x|%d %b %H:%M}}<extra></extra>"))
    for level, text in [(80, "below 80: watch"), (50, "below 50: service it")]:
        hfig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dot"), annotation_text=text,
                       annotation_position="bottom right", annotation_font=dict(color=MUTED, size=11))
    lay = base_layout("Sensor Health Trajectory Over Time (0 - 100)", "Health Score", height=300)
    lay["yaxis"]["range"] = [0, 104]
    hfig.update_layout(**lay)
    st.plotly_chart(hfig, width="stretch")


# ------------------------------------------------------------------ TAB 2: AI Root-Cause Analysis
with tab2:
    st.subheader("🤖 AI Root-Cause Analysis & Maintenance Work Orders")
    st.markdown("Generative AI synthesis combining machine learning classifications, spatial neighbour residuals, and physical failure physics.")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("📑 Generate Executive Reliability Briefing", width="stretch"):
            with st.spinner("Synthesizing Executive Reliability Briefing..."):
                brief = ai_agent.generate_executive_briefing(health_dict, inc.to_dict(orient="records"), api_key=gemini_key)
                st.session_state["exec_briefing"] = brief
        if "exec_briefing" in st.session_state:
            st.markdown(st.session_state["exec_briefing"])

    with col_b:
        if not inc.empty:
            incident_labels = [f"{r.sensor} · {VERDICT_LABEL[r.verdict]} ({r.start:%d %b %H:%M})" for r in inc.itertuples()]
            target_idx = st.selectbox("Select Incident for Deep-Dive RCA", range(len(incident_labels)), format_func=lambda i: incident_labels[i])
            sel_row = inc.iloc[target_idx]
            
            if st.button(f"🔍 Run AI Diagnostic on {sel_row['sensor']}", width="stretch"):
                with st.spinner(f"Analyzing physical root causes for {sel_row['sensor']}..."):
                    reason, top, _ = core.explain(res, get_model(), unit, int(sel_row["row"]), len(sensors), units)
                    incident_info = dict(
                        sensor=sel_row['sensor'],
                        verdict=sel_row['verdict'],
                        duration=f"{sel_row['readings'] * core.STEP_MIN} min",
                        confidence=f"{sel_row['confidence'] * 100:.0f}%",
                        reason=reason
                    )
                    rca = ai_agent.generate_incident_rca(incident_info, top, unit, units, api_key=gemini_key)
                    st.session_state[f"rca_{target_idx}"] = rca
            
            if f"rca_{target_idx}" in st.session_state:
                st.markdown(st.session_state[f"rca_{target_idx}"])
        else:
            st.info("No active incidents to analyze in the current scenario.")


# ------------------------------------------------------------------ TAB 3: AI Sensor Copilot
with tab3:
    st.subheader("💬 AI Sensor Intelligence Copilot")
    st.caption("Ask questions about any sensor, physical root cause, neighbor consensus, or technician procedures.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": f"Hello! I am your AI Sensor Copilot monitoring {len(sensors)} sensors in real-time. Currently, **{focus}** is highlighted. What would you like to investigate?"}
        ]

    # Quick prompt chips
    chips = st.columns(3)
    with chips[0]:
        if st.button(f"❓ Why is {focus} failing?", width="stretch"):
            st.session_state.chat_prompt = f"Why is {focus} failing, and what is the physical root cause?"
    with chips[1]:
        if st.button("🌡️ Real event or sensor fault?", width="stretch"):
            st.session_state.chat_prompt = "How can we prove whether the temperature shift was a real event or a sensor fault?"
    with chips[2]:
        if st.button(f"🛠️ Work order for {focus}", width="stretch"):
            st.session_state.chat_prompt = f"Generate a step-by-step maintenance work order for {focus}."

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask the AI Sensor Copilot...")
    if "chat_prompt" in st.session_state and st.session_state.chat_prompt:
        user_query = st.session_state.chat_prompt
        del st.session_state.chat_prompt

    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        context_data = {
            "sensors": sensors,
            "health": health_dict,
            "incident_count": len(inc),
            "focus": focus,
            "focus_verdict": latest.loc[focus, "verdict"]
        }
        with st.spinner("AI Copilot analyzing sensor network..."):
            reply = ai_agent.ask_sensor_copilot(user_query, context_data, st.session_state.chat_messages, api_key=gemini_key)
        
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


# ------------------------------------------------------------------ TAB 4: Benchmarks
with tab4:
    st.subheader("🔬 Benchmark Evaluation vs Standard Baselines")
    st.write("Held-out testing across 50,688 unseen readings with randomized fault scales outside training range.")
    if st.button("Run Evaluation Benchmark"):
        summary, per_class, extra = run_evaluation()
        st.markdown("**Fault Detection Performance (vs Simple Baselines)**  \n*Event miss rate = share of genuine events wrongly called a fault (lower is better)*")
        st.dataframe(summary, width="stretch")
        st.markdown(f"**SensorTrust Class Breakdown** · Macro F1: **{extra['macro_f1']:.3f}** · Needs Review: **{extra['needs_review_share']:.1%}** · Test Readings: **{extra['test_readings']:,}**")
        st.dataframe(per_class, width="stretch")
