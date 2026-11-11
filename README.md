# roblox-forecast

Forecaster for make.fun competition `roblox-up-and-coming-forecast-01` (Roblox Up-and-Coming top 10).

- `main.py` (Python 3.13, stdlib only): reads `/input/history.ndjson` + `/input/task.json`,
  writes `/output/forecast.json`.
- Model: empirical per-rank drift/diffusion estimated from the chart history, per-game recent
  momentum (exponentially weighted, shrunk toward the positional drift), age-out hazard around
  the 28-day publication window, mixed uniformly over the 72 hourly horizons of the scored
  window. Probabilities are smoothed (mass reserved for unnamed games) so no game gets p = 0
  and the log score stays finite.
- Deterministic (no randomness, no seeds needed), runs in well under a second.
