"""Measure SensorTrust against simple baselines on held-out simulated data.

The test runs use different random seeds from training and fault sizes the model never saw
(smaller and larger than the training range). Everything here is simulated data.

    python evaluate.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

import core


def test_runs(n_runs=8, seed=5000):
    rng = np.random.default_rng(seed)
    for i in range(n_runs):
        scale = float(rng.choice([rng.uniform(0.6, 0.85), rng.uniform(1.35, 1.6)]))   # unseen sizes
        yield core.simulate(days=3, seed=seed + i, random_faults=True, mag_scale=scale)


def zscore_flags(df, k=4.0):
    """Baseline 1: robust z-score of each sensor against its own last 6 hours."""
    V = df.ffill().bfill()
    med = V.rolling(6 * core.PER_HOUR, min_periods=6).median()
    mad = (V - med).abs().rolling(6 * core.PER_HOUR, min_periods=6).median() * 1.4826 + 1e-6
    return (((V - med).abs() / mad) > k) | df.isna()


def iforest_flags(df, contamination=0.08):
    """Baseline 2: Isolation Forest on single-sensor features (no neighbour information)."""
    V = df.ffill().bfill()
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)
    for c in df.columns:
        d = V[c].diff().fillna(0)
        feats = np.c_[d, d.rolling(12, min_periods=1).std().fillna(0),
                      V[c] - V[c].rolling(72, min_periods=1).median()]
        flags[c] = IsolationForest(contamination=contamination, random_state=0).fit_predict(feats) == -1
    return flags | df.isna()


def run(model=None, n_runs=8):
    model = model or core.train_model()
    y_true, y_pred = [], []
    base = {"Z-score": [], "Isolation Forest": []}
    for df, lab in test_runs(n_runs):
        res, _ = core.analyse(df, model)
        keep = res["pos"].values >= core.WARMUP
        y_true.append(lab.T.reshape(-1)[keep])
        y_pred.append(res["verdict"].values[keep])
        km = keep.reshape(df.shape[1], -1).T
        base["Z-score"].append(zscore_flags(df).values[km.nonzero()])
        base["Isolation Forest"].append(iforest_flags(df).values[km.nonzero()])
        _order = km.nonzero()
        base.setdefault("_lab", []).append(lab[_order])
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    names = np.array(core.CLASSES, dtype=object)[y_true]

    is_fault = np.isin(names, core.FAULTS)
    is_event = names == "event"
    rows = []
    lab_b = np.concatenate(base.pop("_lab"))
    names_b = np.array(core.CLASSES, dtype=object)[lab_b]
    for name, flags in base.items():
        f = np.concatenate(flags)
        tf = np.isin(names_b, core.FAULTS)
        rows.append(dict(method=name, fault_precision=precision_score(tf, f), fault_recall=recall_score(tf, f),
                         fault_f1=f1_score(tf, f), event_miss_rate=float(f[names_b == "event"].mean())))
    ours = np.isin(y_pred, core.FAULTS)
    rows.append(dict(method="SensorTrust", fault_precision=precision_score(is_fault, ours),
                     fault_recall=recall_score(is_fault, ours), fault_f1=f1_score(is_fault, ours),
                     event_miss_rate=float(ours[is_event].mean())))
    summary = pd.DataFrame(rows).set_index("method").round(3)

    per_class = []
    for c in core.CLASSES:
        t, p = names == c, y_pred == c
        per_class.append(dict(label=c, readings=int(t.sum()), precision=precision_score(t, p, zero_division=0),
                              recall=recall_score(t, p, zero_division=0), f1=f1_score(t, p, zero_division=0)))
    per_class = pd.DataFrame(per_class).set_index("label").round(3)
    extra = dict(macro_f1=float(per_class["f1"].mean()),
                 needs_review_share=float((y_pred == core.UNCERTAIN).mean()),
                 test_readings=int(len(y_true)))
    extra["confusion"] = pd.crosstab(pd.Series(names, name="true"), pd.Series(y_pred, name="predicted"))
    return summary, per_class, extra


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    summary, per_class, extra = run()
    print("\nFault detection (fault vs not-fault) and event-miss rate, simulated held-out data")
    print("event_miss_rate = share of genuine-event readings wrongly flagged as a fault\n")
    print(summary)
    print("\nSensorTrust per class (plain point-wise scores, no point adjustment)\n")
    print(per_class)
    print("\nConfusion matrix (rows = truth)\n")
    print(extra["confusion"])
    print(f"\nmacro F1 = {extra['macro_f1']:.3f}   needs-review share = {extra['needs_review_share']:.3%}"
          f"   test readings = {extra['test_readings']}")
