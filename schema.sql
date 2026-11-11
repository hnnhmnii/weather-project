CREATE TABLE IF NOT EXISTS observations (
    obs_time     TEXT PRIMARY KEY,   -- local time (Asia/Kuala_Lumpur), 'YYYY-MM-DDTHH:MM'
    temperature  REAL,               -- degC
    humidity     REAL,               -- %
    rainfall     REAL,               -- mm (preceding hour)
    wind_speed   REAL                -- km/h
);

CREATE TABLE IF NOT EXISTS forecasts (
    issued_at    TEXT,               -- when we fetched the forecast (hour)
    target_time  TEXT,               -- hour the forecast is for
    lead_hours   INTEGER,            -- target_time - issued_at, in hours (1..48)
    temperature  REAL,
    humidity     REAL,
    rainfall     REAL,
    wind_speed   REAL,
    rain_prob    REAL,               -- precipitation probability, %
    PRIMARY KEY (issued_at, target_time)
);

CREATE INDEX IF NOT EXISTS idx_fc_target ON forecasts(target_time);
