"""SensorTrust core: simulate sensors, inject faults, detect, explain, score health.

Pipeline
  1. sanity rules        -> dropout / frozen readings are caught outright
  2. ask the neighbours  -> each sensor is compared with the median of the others
  3. classifier          -> LightGBM names what happened (normal, event, or a fault type)
  4. explain + score     -> plain-language reason, 0-100 health score per sensor
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

CLASSES = ["normal", "event", "spike", "drift", "stuck", "noise", "bias", "dropout"]
FAULTS = ["spike", "drift", "stuck", "noise", "bias", "dropout"]
C = {name: i for i, name in enumerate(CLASSES)}
UNCERTAIN = "needs review"

STEP_MIN = 5                      # one reading every 5 minutes
PER_HOUR = 60 // STEP_MIN
PER_DAY = 24 * PER_HOUR
WARMUP = 6 * PER_HOUR             # first 6 h are used to learn each sensor's normal offset

# how hard each fault type hits the health score (0 = not at all, 1 = fully)
SEVERITY = {"spike": 0.5, "drift": 1.0, "stuck": 1.0, "noise": 0.7, "bias": 0.9, "dropout": 0.8}
FORGET = 0.985                    # health memory: older evidence fades (~5.5 h time constant)
MIN_CONFIDENCE = 0.55             # below this the system says "needs review"
FROZEN_RUN = 4                    # this many identical readings in a row means the sensor is frozen

ACTIONS = {
    "event": "No sensor action. The environment really changed, so act on the reading.",
    "spike": "Ignore this reading. If spikes repeat, check wiring and power.",
    "drift": "Recalibrate the sensor against a reference.",
    "stuck": "Restart the sensor. If it stays frozen, replace it.",
    "noise": "Check the battery, power supply and connectors.",
    "bias": "Recalibrate, and check whether the sensor was moved or covered.",
    "dropout": "Check power and network connection.",
    UNCERTAIN: "Not confident either way. Have someone take a look.",
}

FEATURES = ["r", "dr", "spike_dev", "fwd2", "r_mean_s", "r_mean_l", "r_std_s", "r_slope", "std_ratio",
            "run_same", "own_delta", "neigh_delta", "neigh_dev3", "neigh_yday", "neigh_frac_moved",
            "missing_frac"]
BUFFER = 2                        # readings of look-ahead before a verdict is final (10 minutes)
FEATURE_TEXT = {
    "r": "gap between this sensor and its neighbours",
    "dr": "sudden change in that gap",
    "spike_dev": "one-off jump away from its own recent level",
    "r_mean_s": "average gap to neighbours over the last hour",
    "r_mean_l": "average gap to neighbours over the last 6 hours",
    "r_std_s": "how jittery the gap to neighbours is",
    "r_slope": "how fast the gap to neighbours is growing",
    "std_ratio": "jitter compared with neighbouring sensors",
    "run_same": "number of identical readings in a row",
    "own_delta": "this sensor's change over the last hour",
    "neigh_delta": "neighbours' change over the last hour",
    "fwd2": "whether the jump was still there two readings later",
    "neigh_dev3": "neighbours' change over the last 3 hours",
    "neigh_yday": "neighbours compared with the same time yesterday",
    "neigh_frac_moved": "share of neighbours that also changed",
    "missing_frac": "share of readings missing",
}


# --------------------------------------------------------------------------- simulation
def simulate(n_sensors=8, days=3, seed=0, events=None, faults=None, random_faults=False,
             mag_scale=1.0, start="2026-10-01"):
    """Simulated co-located temperature sensors.

    events: list of (start_index, magnitude_degC)          -> seen by every sensor
    faults: list of dict(sensor, kind, start, dur, mag)    -> hits one sensor only
    Returns (readings DataFrame, labels array [time, sensor] of class indices).
    """
    rng = np.random.default_rng(seed)
    n = days * PER_DAY
    t = np.arange(n)
    weather = np.convolve(np.cumsum(rng.normal(0, 0.035, n + 200)), np.ones(40) / 40, "same")[100:100 + n]
    shared = 4.0 * np.sin(2 * np.pi * (t / PER_DAY - 0.3)) + weather

    if events is None:
        events = []
        if random_faults:
            for _ in range(int(rng.integers(2, 5))):
                events.append((int(rng.integers(WARMUP, n - 4 * PER_HOUR)),
                               float(rng.choice([-1, 1]) * rng.uniform(2.0, 5.0))))

    offset = rng.normal(0, 0.4, n_sensors)
    gain = rng.normal(1.0, 0.03, n_sensors)
    noise_sd = rng.uniform(0.08, 0.15, n_sensors)
    X = np.empty((n, n_sensors))
    labels = np.zeros((n, n_sensors), dtype=int)
    for j in range(n_sensors):
        sig = np.roll(shared, int(rng.integers(0, 3)))
        ev = np.zeros(n)
        for s0, mag in events:
            # fast rise, slow decay; each sensor feels it a little differently and a little later
            rise, decay = rng.uniform(2, 4), rng.uniform(14, 36)
            k = np.arange(n) - (s0 + int(rng.integers(0, 4)))
            shape = np.where(k >= 0, (1 - np.exp(-k / rise)) * np.exp(-k / decay), 0.0)
            ev += mag * rng.uniform(0.6, 1.3) * shape / shape.max()
        X[:, j] = 24.0 + offset[j] + gain[j] * sig + ev + rng.normal(0, noise_sd[j], n)
        labels[np.abs(ev) > 0.5, j] = C["event"]

    if faults is None:
        faults = _random_faults(rng, n, n_sensors) if random_faults else []
    for f in faults:
        _inject(X, labels, rng, noise_sd, mag_scale=mag_scale, **f)

    idx = pd.date_range(start, periods=n, freq=f"{STEP_MIN}min")
    cols = [f"S{j + 1}" for j in range(n_sensors)]
    return pd.DataFrame(np.round(X, 2), index=idx, columns=cols), labels


def _random_faults(rng, n, n_sensors):
    out = []
    for j in range(n_sensors):
        slots = np.linspace(WARMUP, n - 12 * PER_HOUR, 4).astype(int)
        for a, b in zip(slots[:-1], slots[1:]):
            if rng.random() < 0.75:
                kind = str(rng.choice(FAULTS))
                out.append(dict(sensor=j, kind=kind, start=int(rng.integers(a, max(a + 1, b - 10 * PER_HOUR))),
                                dur=None, mag=None))
    return out


def _inject(X, labels, rng, noise_sd, sensor, kind, start, dur=None, mag=None, mag_scale=1.0):
    n = len(X)
    sign = rng.choice([-1, 1])
    defaults = {  # (duration in samples, magnitude in degC)
        "spike": (int(rng.integers(1, 3)), sign * rng.uniform(2.0, 6.0)),
        "drift": (int(rng.integers(4, 10) * PER_HOUR), sign * rng.uniform(1.5, 4.0)),
        "stuck": (int(rng.integers(12, 48)), 0.0),
        "noise": (int(rng.integers(12, 48)), rng.uniform(5.0, 9.0)),
        "bias": (int(rng.integers(24, 96)), sign * rng.uniform(1.0, 3.0)),
        "dropout": (int(rng.integers(6, 24)), 0.0),
    }[kind]
    dur = int(dur) if dur else defaults[0]
    mag = float(mag) if mag is not None else defaults[1] * mag_scale
    s, e = int(start), min(n, int(start) + dur)
    if s >= e:
        return
    seg = slice(s, e)
    hit = np.ones(e - s, dtype=bool)
    if kind == "spike":
        X[seg, sensor] += mag
    elif kind == "drift":
        ramp = mag * np.linspace(0, 1, e - s)
        X[seg, sensor] += ramp
        hit = np.abs(ramp) > 0.5
    elif kind == "stuck":
        X[seg, sensor] = X[s, sensor]
    elif kind == "noise":
        X[seg, sensor] += rng.normal(0, noise_sd[sensor] * abs(mag), e - s)
    elif kind == "bias":
        X[seg, sensor] += mag
    elif kind == "dropout":
        X[seg, sensor] = np.nan
    labels[s:e, sensor][hit] = C[kind]


# --------------------------------------------------------------------------- features
def compute_features(df: pd.DataFrame):
    """One row of features per (time, sensor). Everything is expressed in units of sensor noise,
    so the same model works whatever the physical unit of the readings is."""
    V = df.astype(float)
    miss = V.isna()
    Vf = V.ffill().bfill()
    d1 = Vf.diff().fillna(0.0)
    mad = np.nanmedian(np.abs(d1.values - np.nanmedian(d1.values, axis=0)), axis=0)
    unit = max(float(np.median(1.4826 * mad)) / np.sqrt(2), 1e-6)

    cal = max(6, min(WARMUP, len(V) // 4))
    off = Vf.iloc[:cal].sub(Vf.iloc[:cal].median(axis=1), axis=0).median()
    A = (Vf - off).mask(miss)                      # aligned readings, gaps kept as gaps
    moved = (Vf - Vf.shift(6)).abs() > 5 * unit
    std_s = d1.rolling(PER_HOUR, min_periods=3).std()

    frames = []
    for j, col in enumerate(V.columns):
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nm = np.nanmedian(A.drop(columns=col).values, axis=1)
                nstd = np.nanmedian(std_s.drop(columns=col).values, axis=1)
        nm = pd.Series(nm, index=V.index).ffill().bfill()
        own = Vf[col] - off[col]
        r = (own - nm) / unit
        r_mean_s = r.rolling(PER_HOUR, min_periods=1).mean()
        zero = (d1[col] == 0) & (np.arange(len(V)) > 0)
        run = zero.groupby((~zero).cumsum()).cumsum().clip(upper=24)
        f = pd.DataFrame({
            "r": r,
            "dr": r.diff(),
            "spike_dev": r - r.shift(1).rolling(6, min_periods=1).median(),
            "fwd2": r.shift(-BUFFER) - r,
            "r_mean_s": r_mean_s,
            "r_mean_l": r.rolling(6 * PER_HOUR, min_periods=1).mean(),
            "r_std_s": r.rolling(PER_HOUR, min_periods=3).std(),
            "r_slope": r_mean_s - r_mean_s.shift(3 * PER_HOUR),
            "std_ratio": (std_s[col] / unit) / (pd.Series(nstd, index=V.index) / unit + 0.2),
            "run_same": run.astype(float),
            "own_delta": (Vf[col] - Vf[col].shift(PER_HOUR)) / unit,
            "neigh_delta": (nm - nm.shift(PER_HOUR)) / unit,
            "neigh_dev3": (nm - nm.shift(3 * PER_HOUR)) / unit,
            "neigh_yday": (nm - nm.shift(PER_DAY)) / unit,
            "neigh_frac_moved": (moved.sum(axis=1) - moved[col]) / max(1, V.shape[1] - 1),
            "missing_frac": miss[col].rolling(6, min_periods=1).mean(),
        }).fillna(0.0)
        f.insert(0, "sensor", col)
        f.insert(1, "pos", np.arange(len(V)))
        f["is_missing"] = miss[col].values
        frames.append(f)
    return pd.concat(frames), unit


# --------------------------------------------------------------------------- model
def make_training_set(n_runs=12, seed=1000, mag_range=(0.9, 1.3), days=3, n_sensors=8):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for i in range(n_runs):
        df, lab = simulate(n_sensors=n_sensors, days=days, seed=seed + i, random_faults=True,
                           mag_scale=float(rng.uniform(*mag_range)))
        F, _ = compute_features(df)
        y = lab.T.reshape(-1)                      # features are stacked sensor by sensor
        # frozen and missing readings belong to the rule layer, so the classifier never sees them
        ruled = F["is_missing"].values | (F["run_same"].values >= FROZEN_RUN) | np.isin(y, [C["stuck"], C["dropout"]])
        keep = (F["pos"].values >= WARMUP) & ~ruled
        Xs.append(F.loc[keep, FEATURES].values)
        ys.append(y[keep])
    return np.vstack(Xs), np.concatenate(ys)


def train_model(n_runs=20, seed=1000):
    X, y = make_training_set(n_runs=n_runs, seed=seed)
    rng = np.random.default_rng(seed)
    keep = (y != C["normal"]) | (rng.random(len(y)) < 0.5)      # thin out the "normal" majority
    # small trees and large leaves on purpose: fault boundaries are fuzzy, so a model that
    # memorises them becomes confidently wrong on clean data
    model = LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, min_child_samples=150,
                           reg_lambda=5.0, random_state=0, verbose=-1)
    model.fit(pd.DataFrame(X[keep], columns=FEATURES), y[keep])
    return model


# --------------------------------------------------------------------------- analysis
def analyse(df: pd.DataFrame, model):
    """Run the whole pipeline. Returns a long table with one row per (time, sensor)."""
    F, unit = compute_features(df)
    Xf = F[FEATURES]
    proba = model.predict_proba(Xf)
    full = np.zeros((len(F), len(CLASSES)))
    full[:, model.classes_] = proba

    # layer 1: sanity rules override the classifier for the unambiguous cases
    missing = F["is_missing"].values
    full[missing] = 0.0
    full[missing, C["dropout"]] = 1.0
    frozen = (F["run_same"].values >= FROZEN_RUN) & ~missing
    full[frozen] = 0.0
    full[frozen, C["stuck"]] = 1.0

    pred = full.argmax(axis=1)
    conf = full.max(axis=1)
    verdict = np.array(CLASSES, dtype=object)[pred]
    # "needs review" is for doubt about whether a sensor is faulty. If the only doubt is between
    # normal and genuine event, no sensor is at fault either way, so the likelier of the two stands.
    unsure = conf < MIN_CONFIDENCE
    harmless = unsure & ((full[:, C["normal"]] + full[:, C["event"]]) >= 0.8)
    verdict[unsure] = UNCERTAIN
    verdict[harmless] = np.where(full[harmless, C["event"]] > full[harmless, C["normal"]], "event", "normal")
    warm = F["pos"].values < WARMUP
    verdict[warm & ~missing] = "normal"

    slot = {c: i for i, c in enumerate(model.classes_)}

    out = F[["sensor", "pos"] + FEATURES].copy()
    out["time"] = F.index
    out["value"] = df.values.T.reshape(-1)
    out["verdict"] = verdict
    out["confidence"] = conf
    out["rule"] = np.where(missing, "missing reading", np.where(frozen, "frozen reading", ""))
    out["pred_slot"] = [slot.get(p, 0) for p in pred]
    out = out.reset_index(drop=True)
    out["health"] = np.concatenate([health_series(g["verdict"].values, g["confidence"].values)
                                    for _, g in out.groupby("sensor", sort=False)])
    return out, unit


def health_series(verdicts, confidence):
    """0-100 health per sensor. Only confirmed faults count against a sensor; a genuine event,
    a normal reading or an uncertain verdict never lowers the score."""
    a, b = 12.0, 0.0
    h = np.empty(len(verdicts))
    for i, (v, c) in enumerate(zip(verdicts, confidence)):
        sev = SEVERITY.get(v, 0.0) * float(c)
        a = FORGET * a + (1.0 - sev)
        b = FORGET * b + sev
        h[i] = 100.0 * a / (a + b)
    return h


def health_band(score):
    return "Healthy" if score >= 80 else ("Watch" if score >= 50 else "Service it")


def incidents(res: pd.DataFrame, upto=None, min_len=3):
    """Group flagged readings into incidents: one row per continuous episode on one sensor."""
    rows = []
    for sensor, g in res.groupby("sensor", sort=False):
        if upto is not None:
            g = g[g["pos"] <= upto]
        v = g["verdict"].values
        i = 0
        while i < len(v):
            if v[i] == "normal":
                i += 1
                continue
            j = i                                   # extend while the same verdict returns within 6 readings
            while True:
                nxt = [k for k in range(j + 1, min(len(v), j + 7)) if v[k] == v[i]]
                if not nxt:
                    break
                j = nxt[-1]
            seg = g.iloc[i:j + 1]
            seg = seg[seg["verdict"] == v[i]]
            need = {"spike": 1, "dropout": 1, "event": 4, UNCERTAIN: 4}.get(v[i], min_len)
            if len(seg) >= need:
                # the reading that best shows the reason: the onset for an event, the longest
                # run for a frozen sensor, otherwise the most confident reading
                if v[i] == "event":
                    peak = seg.loc[seg["neigh_delta"].abs().idxmax()]
                elif v[i] == "stuck":
                    peak = seg.loc[seg["run_same"].idxmax()]
                else:
                    peak = seg.loc[seg["confidence"].idxmax()]
                rows.append(dict(sensor=sensor, start=seg["time"].iloc[0], end=seg["time"].iloc[-1],
                                 readings=len(seg), verdict=v[i],
                                 confidence=float(seg["confidence"].mean()), row=int(peak.name)))
            i = j + 1
    cols = ["sensor", "start", "end", "readings", "verdict", "confidence", "row"]
    return pd.DataFrame(rows, columns=cols).sort_values("start", ascending=False).reset_index(drop=True)


def explain(res: pd.DataFrame, model, unit, row: int, n_sensors: int, units="°C"):
    """Plain-language reason for one reading, built only from numbers the system measured."""
    x = res.loc[row]
    v = x["verdict"]
    u = lambda val: f"{val * unit:+.1f} {units}"            # noise units -> physical units
    others = n_sensors - 1
    moved = int(round(x["neigh_frac_moved"] * others))
    if x["rule"] == "missing reading":
        text = "No reading arrived from this sensor."
    elif x["rule"] == "frozen reading":
        text = (f"The value has not changed for {int(x['run_same'])} readings in a row, while a "
                f"working sensor always wobbles a little.")
    elif v == "event" and moved < 2:
        text = (f"All the sensors shifted together ({u(x['neigh_dev3'])} over the last 3 hours) and this "
                f"one is still within {abs(x['r'] * unit):.1f} {units} of its neighbours. "
                f"A change everyone sees is real.")
    elif v == "event":
        text = (f"This sensor changed {u(x['own_delta'])} in the last hour and {moved} of its {others} "
                f"neighbours changed too (they moved {u(x['neigh_delta'])}). A change everyone sees is real.")
    elif v == "spike":
        text = (f"The reading jumped {u(x['spike_dev'])} away from its own recent level in one step, "
                f"while the neighbours stayed where they were.")
    elif v == "drift":
        text = (f"Over the last 6 hours this sensor has crept {u(x['r_mean_l'])} away from its "
                f"neighbours, and the gap is still growing ({u(x['r_slope'])} in 3 hours).")
    elif v == "bias":
        text = (f"The sensor reads {u(x['r_mean_s'])} off from its neighbours and has held that "
                f"offset steadily.")
    elif v == "noise":
        text = (f"The readings are {x['std_ratio']:.1f} times more jittery than the neighbouring "
                f"sensors right now.")
    elif v == UNCERTAIN:
        text = (f"The sensor is {u(x['r'])} from its neighbours, but the pattern does not clearly "
                f"match an event or any known fault.")
    else:
        text = "The reading agrees with its neighbours."
    nf = len(FEATURES)
    raw = model.booster_.predict(res.loc[[row], FEATURES], pred_contrib=True)
    c = raw.reshape(-1, nf + 1)[int(x["pred_slot"]), :nf]       # TreeSHAP values for the chosen class
    order = np.argsort(-c)[:4]
    top = [(FEATURE_TEXT[FEATURES[i]], float(c[i])) for i in order if c[i] > 0]
    return text, top, ACTIONS.get(v, "")
