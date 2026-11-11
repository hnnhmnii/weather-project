"""
analyze.py - verifies Open-Meteo forecasts against observations and writes docs/results.json
(the file Aleeya's dashboard reads).

Usage: python analyze.py [--db weather.db] [--out docs/results.json]

Outputs:
  * MAE / RMSE / bias / correlation per variable, overall and per lead-time bucket and per lead hour
  * Rain-event verification (accuracy, POD, FAR, CSI, F1, Brier score)
  * Persistence baseline + skill score (is the forecast better than 'tomorrow = now'?)
  * Simple bias-corrected forecast (own "forecast" model), trained on first 60% of data, tested on the rest
  * Time series of forecast vs observed at lead 1h / 6h / 12h / 24h, and the latest 48h forecast
"""
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

VARS = ["temperature", "humidity", "rainfall", "wind_speed"]
UNITS = {"temperature": "degC", "humidity": "%", "rainfall": "mm", "wind_speed": "km/h"}
EDGES = [0, 6, 12, 24, 48]
LABELS = ["1-6h", "7-12h", "13-24h", "25-48h"]
RAIN_MM = 0.1       # observed rain threshold
RAIN_PROB = 50      # forecast "rain" if probability >= 50 %


def f(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 3)


def metrics(fc, ob):
    e = fc - ob
    n = len(e)
    if n == 0:
        return {"n": 0}
    r = np.corrcoef(fc, ob)[0, 1] if n > 2 and fc.std() > 0 and ob.std() > 0 else np.nan
    return {"n": n, "mae": f(e.abs().mean()), "rmse": f(np.sqrt((e ** 2).mean())),
            "bias": f(e.mean()), "r": f(r)}


def rain_scores(d):
    d = d.dropna(subset=["rain_prob", "rainfall_obs"])
    if d.empty:
        return {"n": 0}
    o = d["rainfall_obs"] >= RAIN_MM
    p = d["rain_prob"] >= RAIN_PROB
    hit, miss = int((o & p).sum()), int((o & ~p).sum())
    fa, cn = int((~o & p).sum()), int((~o & ~p).sum())
    div = lambda a, b: a / b if b else np.nan
    prec, pod = div(hit, hit + fa), div(hit, hit + miss)
    return {"n": len(d), "hits": hit, "misses": miss, "false_alarms": fa, "correct_negatives": cn,
            "accuracy": f(div(hit + cn, len(d))), "pod": f(pod), "far": f(div(fa, hit + fa)),
            "csi": f(div(hit, hit + miss + fa)),
            "f1": f(div(2 * prec * pod, prec + pod) if not (np.isnan(prec) or np.isnan(pod)) else np.nan),
            "brier": f(((d["rain_prob"] / 100 - o.astype(float)) ** 2).mean())}


def main(db, out):
    con = sqlite3.connect(db)
    obs = pd.read_sql("SELECT * FROM observations", con, parse_dates=["obs_time"])
    fc = pd.read_sql("SELECT * FROM forecasts", con, parse_dates=["issued_at", "target_time"])
    con.close()

    res = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "location": "Seri Iskandar, Perak", "units": UNITS,
           "counts": {"observations": len(obs), "forecasts": len(fc)}}
    if obs.empty or fc.empty:
        res["error"] = "not enough data yet"
    else:
        res["period"] = {"start": str(obs.obs_time.min()), "end": str(obs.obs_time.max()),
                         "days": round((obs.obs_time.max() - obs.obs_time.min()).total_seconds() / 86400, 2)}
        df = fc.merge(obs, left_on="target_time", right_on="obs_time", suffixes=("_fc", "_obs"))
        df["bucket"] = pd.cut(df.lead_hours, EDGES, labels=LABELS)
        res["matched_pairs"] = len(df)
        if df.empty:
            res["error"] = "no forecast has reached its target time yet - wait a few hours and run again"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(res, indent=1))
            print("wrote", out, "(no matched pairs yet)")
            return

        base = obs.set_index("obs_time")
        overall, by_bucket, by_lead, persist, corrected = {}, {}, {}, {}, {}
        cut = df.issued_at.sort_values().iloc[int(len(df) * 0.6)] if len(df) else None
        train, test = df[df.issued_at < cut], df[df.issued_at >= cut]

        for v in VARS:
            a, p = df[f"{v}_obs"], df[f"{v}_fc"]
            ok = a.notna() & p.notna()
            overall[v] = metrics(p[ok], a[ok])
            by_bucket[v] = {b: metrics(g[f"{v}_fc"].dropna(), g.loc[g[f"{v}_fc"].notna(), f"{v}_obs"])
                            for b, g in df[ok].groupby("bucket", observed=True)}
            by_lead[v] = [{"lead": int(l), "mae": metrics(g[f"{v}_fc"], g[f"{v}_obs"]).get("mae")}
                          for l, g in df[ok].groupby("lead_hours")]

            # persistence baseline: "future value = value at issue time"
            pers = df.issued_at.map(base[v])
            m = ok & pers.notna()
            pm = metrics(pers[m], a[m])
            fm = metrics(p[m], a[m])
            persist[v] = {"persistence_mae": pm.get("mae"), "forecast_mae": fm.get("mae"),
                          "skill": f(1 - fm["mae"] / pm["mae"]) if pm.get("mae") else None}

            # bias correction per lead bucket (train on first 60 %, test on last 40 %)
            if len(train) and len(test):
                bias = (train[f"{v}_fc"] - train[f"{v}_obs"]).groupby(train.bucket, observed=True).mean()
                adj = test[f"{v}_fc"] - test.bucket.map(bias).astype(float).fillna(0)
                if v == "rainfall":
                    adj = adj.clip(lower=0)
                if v == "humidity":
                    adj = adj.clip(0, 100)
                t_ok = test[f"{v}_obs"].notna() & adj.notna()
                corrected[v] = {"raw_mae": metrics(test.loc[t_ok, f"{v}_fc"], test.loc[t_ok, f"{v}_obs"]).get("mae"),
                                "corrected_mae": metrics(adj[t_ok], test.loc[t_ok, f"{v}_obs"]).get("mae")}

        res.update(overall=overall, by_lead_bucket=by_bucket, mae_by_lead_hour=by_lead,
                   persistence_baseline=persist, bias_corrected=corrected,
                   rain_events={"threshold_obs_mm": RAIN_MM, "threshold_prob_pct": RAIN_PROB,
                                "overall": rain_scores(df),
                                "by_lead_bucket": {b: rain_scores(g) for b, g in df.groupby("bucket", observed=True)}})

        # time series for charts
        series = {}
        for lead in (1, 6, 12, 24):
            d = df[df.lead_hours == lead].sort_values("target_time")
            series[f"lead_{lead}h"] = [
                {"time": t.strftime("%Y-%m-%dT%H:%M"),
                 **{v: {"forecast": f(r[f"{v}_fc"]), "observed": f(r[f"{v}_obs"])} for v in VARS},
                 "rain_prob": f(r["rain_prob"])}
                for t, r in zip(d.target_time, d.to_dict("records"))]
        res["series"] = series
        res["observations"] = [{"time": t.strftime("%Y-%m-%dT%H:%M"), **{v: f(r[v]) for v in VARS}}
                               for t, r in zip(obs.obs_time, obs.to_dict("records"))]
        last = fc[fc.issued_at == fc.issued_at.max()].sort_values("target_time")
        res["latest_forecast"] = {"issued_at": str(last.issued_at.iloc[0]),
                                  "rows": [{"time": t.strftime("%Y-%m-%dT%H:%M"), "lead": int(r["lead_hours"]),
                                            **{v: f(r[v]) for v in VARS}, "rain_prob": f(r["rain_prob"])}
                                           for t, r in zip(last.target_time, last.to_dict("records"))]}
        # next 6 h rain alert (nice for the dashboard / bonus hardware idea)
        nxt = last.head(6)
        res["rain_alert_next6h"] = bool((nxt.rain_prob >= RAIN_PROB).any() or (nxt.rainfall >= RAIN_MM).any())

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path(__file__).with_name("weather.db")))
    ap.add_argument("--out", default=str(Path(__file__).with_name("docs") / "results.json"))
    a = ap.parse_args()
    main(a.db, a.out)
