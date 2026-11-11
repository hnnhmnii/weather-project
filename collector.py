"""
collector.py - pulls weather data for Seri Iskandar from Open-Meteo and stores it in SQLite.
"""
import argparse
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

LAT, LON = 4.3667, 100.9667            # Seri Iskandar, Perak
HERE = Path(__file__).parent
DB = HERE / "weather.db"
MYT = timezone(timedelta(hours=8))      # Malaysia time (UTC+8)
URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation",
          "wind_speed_10m", "precipitation_probability"]


def fetch():
    params = dict(latitude=LAT, longitude=LON, hourly=",".join(HOURLY),
                  past_hours=6, forecast_hours=48,
                  timezone="Asia/Kuala_Lumpur", wind_speed_unit="kmh")
    with urlopen(URL + "?" + urlencode(params), timeout=30) as r:
        return json.load(r)["hourly"]


def init_db(con):
    schema_path = HERE / "schema.sql"
    if schema_path.exists():
        con.executescript(schema_path.read_text(encoding="utf-8"))


def store(con, h):
    now = datetime.now(MYT).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    issued = now.strftime("%Y-%m-%dT%H:%M")
    n_obs = n_fc = 0
    for i, t in enumerate(h["time"]):
        tt = datetime.fromisoformat(t)
        temp, hum = h["temperature_2m"][i], h["relative_humidity_2m"][i]
        rain, wind = h["precipitation"][i], h["wind_speed_10m"][i]
        prob = h["precipitation_probability"][i]
        if tt <= now:
            con.execute("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?)",
                        (t, temp, hum, rain, wind))
            n_obs += 1
        else:
            lead = int((tt - now).total_seconds() // 3600)
            con.execute("INSERT OR IGNORE INTO forecasts VALUES (?,?,?,?,?,?,?,?)",
                        (issued, t, lead, temp, hum, rain, wind, prob))
            n_fc += 1
    con.commit()
    print(f"[{issued}] stored {n_obs} observation rows, {n_fc} forecast rows")


def run_once():
    con = sqlite3.connect(DB)
    init_db(con)
    store(con, fetch())
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    if not args.loop:
        run_once()
    else:
        while True:
            try:
                run_once()
            except Exception as e:
                print("fetch failed:", e)
            time.sleep(3600)
