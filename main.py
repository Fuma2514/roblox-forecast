#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads hourly chart history from /input/history.ndjson and writes /output/forecast.json.
Model: empirical rank drift-diffusion per chart position, per-game momentum shrinkage,
age-out hazard near the 28-day publication window, mixed uniformly over horizons 1..72h.
Deterministic, stdlib only.
"""
import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

PLACES = 10
MAX_NAMED = 25          # games named per place (cap 100)
EPS = 1e-6              # probability floor for named rows
UNNAMED_RESERVE = 0.10  # mass left unnamed per place (shared over unnamed games)
PUB_WINDOW_DAYS = 28.0
AGEOUT_START = 26.5     # days: begin tapering survival
AGEOUT_END = 29.5

def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def load_history():
    path = os.path.join(INPUT_DIR, "history.ndjson")
    snaps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("ok"):
                continue
            rows = d.get("up_and_coming") or []
            if not rows:
                continue
            rows = [r for r in rows if isinstance(r.get("rank"), int)]
            if not rows:
                continue
            rows.sort(key=lambda r: r["rank"])
            snaps.append({
                "at": parse_ts(d["at"]),
                "chart": [r["universe_id"] for r in rows],
                "playing": {r["universe_id"]: (r.get("playing") or 0) for r in rows},
            })
    snaps.sort(key=lambda s: s["at"])
    return snaps

def game_ages(snaps, now):
    """Age (days) of each current game: from earliest sighting (details.created
    is unavailable in archive lines; first-seen underestimates age)."""
    first_seen = {}
    for s in snaps:
        for g in s["chart"]:
            first_seen.setdefault(g, s["at"])
    ages = {}
    for g, t in first_seen.items():
        ages[g] = max(0.0, (now - t).total_seconds() / 86400.0)
    return ages

def survival(age_days, horizon_h):
    """Probability the game is still on the chart after horizon_h hours,
    given its current age in days (28-day publication window)."""
    age_end = age_days + horizon_h / 24.0
    if age_end <= AGEOUT_START:
        return 1.0
    if age_end >= AGEOUT_END:
        return 0.0
    return (AGEOUT_END - age_end) / (AGEOUT_END - AGEOUT_START)

def drift_stats(snaps, ages):
    """Empirical hourly rank drift and spread per current rank, plus exit rates."""
    # collect (rank, gap_h, rank_change, exited)
    buckets = {}  # rank -> list of (rate_per_hour, gap_h, exited)
    for i in range(len(snaps) - 1):
        a, b = snaps[i], snaps[i + 1]
        gap = (b["at"] - a["at"]).total_seconds() / 3600.0
        if gap <= 0 or gap > 24:
            continue
        pa = {g: r for r, g in enumerate(a["chart"], 1)}
        pb = {g: r for r, g in enumerate(b["chart"], 1)}
        gaph = gap
        for g, r in pa.items():
            if g in pb:
                buckets.setdefault(r, []).append((pb[g] - r) / gaph)
            else:
                buckets.setdefault(r, []).append(None)  # exited within gap
    drift = {}
    sigma = {}
    for r, evs in buckets.items():
        moves = [e for e in evs if e is not None]
        if moves:
            moves.sort()
            n = len(moves)
            med = moves[n // 2] if n % 2 else 0.5 * (moves[n // 2 - 1] + moves[n // 2])
            var = sum((m - med) ** 2 for m in moves) / max(1, n - 1)
            drift[r] = med
            sigma[r] = math.sqrt(max(var, 1e-4))
        else:
            drift[r] = 0.0
            sigma[r] = 0.5
    return drift, sigma

def momentum(snaps, current_chart, half_life_h=18.0):
    """Shrunk recent rank velocity per game (ranks per hour)."""
    pos_hist = []
    for s in snaps[-6:]:
        pos_hist.append((s["at"], {g: r for r, g in enumerate(s["chart"], 1)}))
    vel = {}
    last_t, last_p = pos_hist[-1]
    for g in current_chart:
        num = 0.0
        den = 0.0
        for t, p in pos_hist[:-1]:
            if g in p and g in last_p:
                dt = (last_t - t).total_seconds() / 3600.0
                if dt <= 0:
                    continue
                w = 0.5 ** (dt / half_life_h)
                num += w * (last_p[g] - p[g]) / dt
                den += w
        vel[g] = num / den if den > 1e-9 else 0.0
    return vel

def normal_pdf(x, mu, sig):
    z = (x - mu) / sig
    return math.exp(-0.5 * z * z) / (sig * math.sqrt(2 * math.pi))

def forecast(snaps, task, drift, sigma, vel, ages):
    last = snaps[-1]
    now = last["at"]
    chart = last["chart"]
    n_chart = len(chart)
    pos = {g: r for r, g in enumerate(chart, 1)}
    horizon_max = task["window"]["slots"] * task["window"]["slot_minutes"] / 60.0

    # candidate games: the whole current chart
    cands = chart[:]

    # per-horizon rank distributions -> mixture over horizons (uniform)
    mix = {g: [0.0] * PLACES for g in cands}   # P(game at place k), on-chart mass only
    surv_mix = {g: 0.0 for g in cands}

    rows_by_place = {k: [] for k in range(1, PLACES + 1)}
    for h in range(1, int(horizon_max) + 1):
        for g in cands:
            r0 = pos[g]
            d = drift.get(r0, 0.0)
            s = sigma.get(r0, 0.5)
            v = vel.get(g, 0.0)
            # shrink momentum toward positional drift
            mu_r = r0 + (0.35 * v + 0.65 * d) * h
            sig_h = s * math.sqrt(max(h, 1.0)) * 1.0
            sig_h = max(sig_h, 0.8)
            sv = survival(ages.get(g, 0.0), h)
            surv_mix[g] += sv / horizon_max
            if sv <= 0.0:
                continue
            # proper truncated normal over ranks 1..n_chart
            pdfs = [normal_pdf(k, mu_r, sig_h) for k in range(1, n_chart + 1)]
            z = sum(pdfs)
            if z <= 1e-300:
                continue
            row = mix[g]
            w = sv / horizon_max
            for k in range(PLACES):
                row[k] += w * pdfs[k] / z

    # renormalise each place's named probabilities, reserve unnamed mass
    rows = []
    for k in range(1, PLACES + 1):
        col = [(mix[g][k - 1], g) for g in cands]
        col.sort(reverse=True)
        top = col[:MAX_NAMED]
        z = sum(p for p, _ in top)
        budget = 1.0 - UNNAMED_RESERVE
        if z <= 1e-12:
            continue
        acc = 0.0
        for p, g in top:
            pr = max(EPS, p / z * budget)
            acc += pr
            rows.append({"universe_id": g, "place": k, "p": pr})
        # fix rounding: scale to exactly `budget`
    # ensure per-place sum <= 1 and not over
    from collections import defaultdict
    place_sum = defaultdict(float)
    for r in rows:
        place_sum[r["place"]] += r["p"]
    for k, s in place_sum.items():
        if s > 1.0 - UNNAMED_RESERVE + 1e-9 or True:
            f = (1.0 - UNNAMED_RESERVE) / s
            for r in rows:
                if r["place"] == k:
                    r["p"] *= f
            place_sum[k] = s * f

    # top10 guess: rank games by expected position (mixture-weighted)
    def score_place(g):
        tot = sum(mix[g])
        if tot <= 1e-12:
            return (float(n_chart) + 10.0, -surv_mix[g])
        exp = sum((i + 1) * p for i, p in enumerate(mix[g])) / tot
        return (exp, -surv_mix[g])
    top10 = [g for g, _ in sorted(((g, score_place(g)) for g in cands), key=lambda x: x[1])][:PLACES]
    while len(top10) < PLACES:
        top10.append(0)  # should not happen; chart has >=10 games
    return {"probabilities": rows, "top10": top10}

def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    snaps = load_history()
    if not snaps:
        raise SystemExit("no snapshots")
    last = snaps[-1]
    chart = last["chart"]
    ages = game_ages(snaps, last["at"])
    drift, sigma = drift_stats(snaps, ages)
    vel = momentum(snaps, chart)
    out = forecast(snaps, task, drift, sigma, vel, ages)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w", encoding="utf-8") as f:
        f.write(json.dumps(out, separators=(",", ":")))
        f.write("\n")

if __name__ == "__main__":
    main()
