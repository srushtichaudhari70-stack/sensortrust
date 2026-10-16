"""SensorTrust dashboard.   Run with:   streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="SensorTrust", layout="wide")
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
# one fixed colour and one fixed marker shape per verdict, so identity never rests on colour alone
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
VERDICT_LABEL = {"event": "genuine event", "spike": "spike (fault)", "drift": "drift (fault)",
                 "stuck": "stuck (fault)", "noise": "noise (fault)", "bias": "bias (fault)",
                 "dropout": "dropout (fault)", core.UNCERTAIN: "needs review"}
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


@st.cache_resource(show_spinner="Training the model on simulated data (about 15 seconds, once)...")
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
    import io
    df = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    res, unit = core.analyse(df, get_model())
    return df, res, unit


@st.cache_data(show_spinner="Testing on held-out simulated data (about 40 seconds)...")
def run_evaluation():
    import evaluate
    return evaluate.run(get_model())


# ------------------------------------------------------------------ sidebar
st.sidebar.title("SensorTrust")
st.sidebar.caption("Tells a broken sensor from a real event, and explains why.")
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

upload = st.sidebar.file_uploader("Or analyse your own CSV (experimental)", type="csv",
                                  help="First column: timestamp. Other columns: one per sensor, "
                                       "at least 3 sensors measuring the same thing, evenly spaced readings.")

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

st.title("SensorTrust")
st.caption(f"Sensor health dashboard · data source: {source} · one reading every {core.STEP_MIN} minutes · "
           f"first {core.WARMUP // H} hours are used to learn each sensor's normal level")

last_h = (n - 1) / H
now_h = st.slider("Replay: show the data up to this hour", float(core.WARMUP // H), float(round(last_h, 1)),
                  float(round(last_h, 1)), 0.5)
now = min(n - 1, int(now_h * H))
view = res[res["pos"] <= now]
inc = core.incidents(res, upto=now)

# ------------------------------------------------------------------ health cards
latest = view[view["pos"] == now].set_index("sensor")
cols = st.columns(len(sensors))
for col, s in zip(cols, sensors):
    score = float(latest.loc[s, "health"])
    band = core.health_band(score)
    v = latest.loc[s, "verdict"]
    with col:
        st.metric(s, f"{score:.0f}", border=True, help="Health score, 0 to 100")
        st.markdown(f":{BAND_COLOR[band]}[**{band}**]  \nnow: {VERDICT_LABEL.get(v, v)}")

focus = st.selectbox("Sensor to highlight", sensors,
                     index=int(np.argmin([latest.loc[s, "health"] for s in sensors])))


def base_layout(title, ytitle, height=380):
    return dict(title=dict(text=title, font=dict(size=15, color=INK)), height=height,
                margin=dict(l=10, r=10, t=48, b=10), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=12), hovermode="closest",
                legend=dict(orientation="h", y=-0.18, x=0),
                xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False),
                yaxis=dict(title=ytitle, gridcolor=GRID, linecolor=GRID, zeroline=False))


# ------------------------------------------------------------------ readings chart
shown = df.iloc[:now + 1]
fig = go.Figure()
for s in sensors:
    if s != focus:
        fig.add_trace(go.Scatter(x=shown.index, y=shown[s], mode="lines", name="other sensors",
                                 legendgroup="others", showlegend=(s == [x for x in sensors if x != focus][0]),
                                 line=dict(color=OTHER_LINE, width=1), hoverinfo="skip"))
fig.add_trace(go.Scatter(x=shown.index, y=shown[focus], mode="lines", name=focus,
                         line=dict(color=INK, width=2),
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
fig.update_layout(**base_layout("Readings, with every flagged reading marked by its verdict", units))
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ incidents + why
left, right = st.columns([3, 2])
with left:
    st.subheader("Incidents")
    if inc.empty:
        st.info("Nothing flagged so far. All sensors agree with each other.")
    else:
        table = inc.assign(verdict=inc["verdict"].map(VERDICT_LABEL),
                           duration=(inc["readings"] * core.STEP_MIN).astype(str) + " min",
                           confidence=(inc["confidence"] * 100).round().astype(int).astype(str) + "%",
                           start=inc["start"].dt.strftime("%d %b %H:%M"))
        st.dataframe(table[["sensor", "start", "duration", "verdict", "confidence"]],
                     width="stretch", hide_index=True, height=300)
with right:
    st.subheader("Why?")
    if inc.empty:
        st.write("Pick a scenario with faults, or add one from the sidebar.")
    else:
        labels = [f"{r.sensor} · {VERDICT_LABEL[r.verdict]} · {r.start:%d %b %H:%M}" for r in inc.itertuples()]
        default = next((i for i, r in enumerate(inc.itertuples()) if r.sensor == focus), 0)
        pick = st.selectbox("Incident", range(len(labels)), index=default, format_func=lambda i: labels[i])
        row = inc.iloc[pick]
        reason, top, action = core.explain(res, get_model(), unit, int(row["row"]), len(sensors), units)
        st.markdown(f"**{row['sensor']}: {VERDICT_LABEL[row['verdict']]}**")
        st.write(reason)
        st.markdown(f"**What to do:** {action}")
        if top and not res.loc[int(row["row"]), "rule"]:
            bar = go.Figure(go.Bar(x=[t[1] for t in top][::-1], y=[t[0] for t in top][::-1], orientation="h",
                                   marker=dict(color="#3987e5" if DARK else "#2a78d6"), width=0.5,
                                   hovertemplate="%{y}<br>weight %{x:.2f}<extra></extra>"))
            lay = base_layout("What pushed the model to this verdict", "", height=230)
            lay["xaxis"]["title"] = "weight of evidence (SHAP value)"
            lay["yaxis"]["title"] = None
            bar.update_layout(**lay)
            st.plotly_chart(bar, width="stretch")
        elif res.loc[int(row["row"]), "rule"]:
            st.caption("Caught by a sanity rule, so no model was needed for this one.")

# ------------------------------------------------------------------ health over time
hfig = go.Figure()
for s in sensors:
    g = view[view["sensor"] == s]
    if s != focus:
        hfig.add_trace(go.Scatter(x=g["time"], y=g["health"], mode="lines", name="other sensors",
                                  legendgroup="others", showlegend=(s == [x for x in sensors if x != focus][0]),
                                  line=dict(color=OTHER_LINE, width=1),
                                  hovertemplate=f"{s}: %{{y:.0f}}<extra></extra>"))
g = view[view["sensor"] == focus]
hfig.add_trace(go.Scatter(x=g["time"], y=g["health"], mode="lines", name=focus, line=dict(color=INK, width=2),
                          hovertemplate=f"{focus}: %{{y:.0f}}<br>%{{x|%d %b %H:%M}}<extra></extra>"))
for level, text in [(80, "below 80: watch"), (50, "below 50: service it")]:
    hfig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dot"), annotation_text=text,
                   annotation_position="bottom right", annotation_font=dict(color=MUTED, size=11))
lay = base_layout("Health score over time (a genuine event never lowers it)", "health, 0 to 100", height=320)
lay["yaxis"]["range"] = [0, 104]
hfig.update_layout(**lay)
st.plotly_chart(hfig, width="stretch")

# ------------------------------------------------------------------ evaluation
with st.expander("How well does it work? (test on held-out simulated data)"):
    st.write("The test uses new random runs with fault sizes the model never saw in training "
             "(smaller and larger). Scores are plain point-by-point, with no point adjustment. "
             "This is simulated data, so treat the numbers as a check of the method, not a field result.")
    if st.button("Run the test"):
        summary, per_class, extra = run_evaluation()
        st.markdown("**Fault detection against simple baselines**  \n"
                    "*event miss rate* = share of genuine-event readings wrongly called a fault (lower is better)")
        st.dataframe(summary, width="stretch")
        st.markdown(f"**SensorTrust per class** · macro F1 {extra['macro_f1']:.2f} · "
                    f"{extra['needs_review_share']:.1%} of readings sent to review · "
                    f"{extra['test_readings']:,} test readings")
        st.dataframe(per_class, width="stretch")
