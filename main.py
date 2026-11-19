"""gta6-trophy-rarity-forecast-01: a counts-decomposition forecaster for GTA VI PS5 trophy earn rates on Exophase.

The program reads one case file at a time from the input directory (JSON, schema
makefun/gta6-trophies-case/v1) and writes one forecast file per case to the output directory:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]} -- five quantiles of each
trophy's earn rate (percent) at the case's target time, non-decreasing, 0..100.

Why the naive forecast fails: Exophase's page only refreshes on its own schedule. In the
practice cases every read before the checkpoint is the same frozen launch-night page (638
players tracked), and the answer is the page after its first refresh (3,868 tracked, every
rate but the 100% one lower). A persistence forecast therefore misses by ~0.7 points and
scores ~2.35 pinball loss. The rates move because (a) new players join and dilute every
rate, and (b) existing players unlock trophies as they progress -- fast for early story
trophies, slow for grind/collectible ones. This program models both, in counts.

Model (per case, stdlib only, deterministic, ~0.1 s):
  1. Dedupe the history into distinct page states (identical consecutive reads are one
     refresh). If there is only one distinct state the page never moved inside the case:
     use the dilution prior (step 4).
  2. Counts decomposition. For each trophy, work with earned counts E = rate * N (or the
     not-earned count M = N - E when the rate is >= 50%, so the saturated trophy "Complete
     Gone Fishing" is modelled on its shrinking complement). Fit ln E and ln N as
     quadratics in u = ln(1 + hours since release) by ridge-weighted least squares
     (recency weights, weights by sqrt(E)). The forecast rate is E(target) / N(target).
     The quadratic's implied slope over the horizon is blended with a shrunk slope and
     clamped to a band around the mean slope of the ten trophies nearest in current rate,
     so a fast trophy keeps its speed and a slow one is not dragged up by the pack.
  3. Spread. The per-trophy logit spread combines the fit residual, the ln N fit residual,
     an extrapolation term growing with the horizon in u, and a refresh-lag term (half the
     median gap between distinct states, capped at 6 h, times the trophy's current rate of
     change). A global multiplier is picked by rolling-origin backtests inside the case's
     own history (origins with >= 6 states, horizons 6/12/24 h), minimising mean pinball
     loss over multipliers 0.5..3.0. Quantiles are the median in logit space plus
     z_tau * spread, clamped to 0.01..99.99 and rounded to 4 decimals (half up).
  4. Frozen-history prior. With no movement to fit, assume the player base grows by a
     factor G = min(30, 1 + 1.5 * horizon_hours) and new players earn each trophy at a
     speed-class share s of its last rate: fast 1.0 (fishing, arcade, chapter 1, ...),
     slow 0.15 (platinum/gold/silver tiers, late chapters, 100% completion, grinds, ...),
     medium 0.5 otherwise. Then E_target = E_last + (N_target - N_last) * s and the rate
     is E_target / N_target, with a wide spread (1.2 logit units).

Backtests (measured while building; see STATUS.md): practice case p01 (14.2 h horizon,
frozen history) -- this model 0.625 pinball loss vs 2.347 for the house baseline and the
house curve baseline (typical miss 0.15 pp vs 0.76 pp). Four simulated final cases
(4.5-day horizon, 21 distinct states, N growing to ~100k-670k, with and without a
post-checkpoint recovery in rates) -- this model 0.149-0.194 vs 0.297-0.328 for the house
curve baseline and 0.314-0.328 for the house baseline.

Safety: any error while forecasting a case falls back to the house baseline (persistence
with a random-walk spread), so a case always gets a valid output file.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Spread of the frozen-history prior, in logit units.
FROZEN_SIGMA = 1.2
# Rate floor/ceiling before the logit, as the scorer clamps.
LO, HI = 0.01, 99.99


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo=LO, hi=HI):
    return min(hi, max(lo, x))


def logit(pct):
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def inv_logit(l):
    if l > 30:
        return 100.0
    if l < -30:
        return 0.0
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation, ~1e-9 accurate)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def solve_ridge(xs, ys, ws, lam):
    """Weighted ridge regression for a small polynomial design; returns coefficients."""
    k = len(xs[0])
    a = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for xi, yi, wi in zip(xs, ys, ws):
        for r in range(k):
            b[r] += wi * xi[r] * yi
            for c in range(k):
                a[r][c] += wi * xi[r] * xi[c]
    for r in range(1, k):
        a[r][r] += lam
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    n = k
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            continue
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col] / pv
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit_curve(us, ys, ws, ridge=1.0, degree=2):
    """Ridge polynomial fit in u; returns (predict_fn, residual_sd, coefficients)."""
    n = len(us)
    deg = min(degree, max(0, n - 1))
    xs = [[u ** d for d in range(deg + 1)] for u in us]
    coef = solve_ridge(xs, ys, ws, ridge)

    def pred(u):
        return sum(c * u ** d for d, c in enumerate(coef))

    res = [y - pred(u) for u, y in zip(us, ys)]
    sd = math.sqrt(sum(r * r for r in res) / max(1, n - deg - 1)) if n > deg + 1 else 0.0
    return pred, sd, coef


def distinct_states(case):
    """Collapse consecutive identical reads into one page state: (time, players, rates)."""
    states = []
    prev = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key == prev:
            continue
        prev = key
        states.append((parse_time(read["at"]), read["players_tracked"], list(read["rates"])))
    return states


FAST_PATTERNS = ("fishing", "arcade game", "chapter 1", "breakfast run",
                 "discover 100 vehicles", "unscheduled departure", "what the hell",
                 "bowling")
SLOW_PATTERNS = ("chapter 2", "chapter 3", "chapter 4", "final mission", "100% completion",
                 "250 stunts", "100 locations", "50 challenges", "gold medal", "8 mission strands",
                 "20 different races", "5 bank vaults", "each gang", "all of leonida",
                 "3 collectible card sets", "drone race at every location", "rhinos",
                 "jewelry store", "unique animal", "cartel heist", "child's play",
                 "headliners", "vcia takedown", "operation savage glory", "strangers")


def speed_class(text, tier):
    """How fast new players earn this trophy, as a share of its last rate: fast 1.0,
    slow 0.15, medium 0.5. Used only by the frozen-history prior."""
    if tier in ("platinum", "gold", "silver"):
        return 0.15
    if any(p in text for p in SLOW_PATTERNS):
        return 0.15
    if any(p in text for p in FAST_PATTERNS):
        return 1.0
    return 0.5


def forecast_medians(case, states, t0, t_last):
    """Per-trophy (median rate, base logit spread) at the case target, from the moving
    history. No spread calibration here -- that step calls this function on sub-cases.
    t_last is the time of the last read in the history (the checkpoint); the states'
    own times are when the page actually changed, which can be earlier when the last
    reads are identical."""
    trophies = case["trophies"]
    t_target = parse_time(case["target"])

    def u_of(t):
        return math.log(1.0 + max(0.0, (t - t0).total_seconds() / 3600.0))

    u_last = u_of(t_last)
    u_tgt = u_of(t_target)
    gaps = [(states[i + 1][0] - states[i][0]).total_seconds() / 3600.0
            for i in range(len(states) - 1)]
    lag_h = min(0.5 * (sorted(gaps)[len(gaps) // 2] if gaps else 4.0), 6.0)

    us = [u_of(t) for t, _, _ in states]
    w_time = [0.5 ** ((t_last - t).total_seconds() / 86400.0) for t, _, _ in states]
    ln_n = [math.log(max(1.0, float(n))) for _, n, _ in states]
    pred_n, sd_n, coef_n = fit_curve(us, ln_n, w_time, ridge=0.5, degree=2)
    ln_n_tgt = pred_n(u_tgt)

    # Per-trophy log-earned series (not-earned counts for rates >= 50%).
    per = []
    for i in range(len(trophies)):
        ys, uu, ww = [], [], []
        for (t, n, rates), u, wt in zip(states, us, w_time):
            r = rates[i]
            if r is None:
                continue
            e = r / 100.0 * n
            if r >= 50.0:
                e = n - e
                if e <= 0:
                    e = 0.5
            if e <= 0:
                e = 0.5
            ys.append(math.log(e))
            uu.append(u)
            ww.append(wt * min(1.0, math.sqrt(e) / 7.0))
        per.append((uu, ys, ww))

    # Linear slopes per trophy, for the neighbourhood-shrinkage target.
    lin_slopes = []
    for uu, ys, ww in per:
        if len(uu) >= 4:
            _, _, coef = fit_curve(uu, ys, ww, ridge=1.0, degree=1)
            lin_slopes.append(coef[1] if len(coef) > 1 else 0.0)
        else:
            lin_slopes.append(None)
    cur_lograte = [math.log(max(0.01, states[-1][2][i] if states[-1][2][i] is not None else 0.01))
                   for i in range(len(trophies))]

    def neighbourhood_slope(i):
        idx = sorted(range(len(trophies)), key=lambda j: abs(cur_lograte[j] - cur_lograte[i]))
        vals = [lin_slopes[j] for j in idx[1:11] if lin_slopes[j] is not None]
        if not vals:
            g = [s for s in lin_slopes if s is not None]
            return sum(g) / len(g) if g else 0.0
        return sum(vals) / len(vals)

    n_slope = coef_n[1] + 2 * coef_n[2] * u_last if len(coef_n) > 2 else coef_n[1]
    du_per_h = 1.0 / (1.0 + (t_last - t0).total_seconds() / 3600.0)

    out = []
    for i in range(len(trophies)):
        uu, ys, ww = per[i]
        r_last = states[-1][2][i]
        complement = bool(r_last is not None and r_last >= 50.0)
        if len(uu) < 2:
            out.append((r_last if r_last is not None else 0.0, 1.2))
            continue
        pred, sd, coef = fit_curve(uu, ys, ww, ridge=1.0, degree=2)
        n_eff = len(uu)
        w_own = n_eff / (n_eff + 2.0)
        slope_tgt = coef[1] + 2 * coef[2] * u_tgt if len(coef) > 2 else coef[1]
        nb = neighbourhood_slope(i)
        slope_blend = w_own * slope_tgt + (1 - w_own) * nb
        val_last = pred(u_last)
        avg_slope = (pred(u_tgt) - val_last) / (u_tgt - u_last) if u_tgt > u_last else slope_blend
        avg_slope = 0.7 * avg_slope + 0.3 * slope_blend
        avg_slope = min(nb + 2.5, max(nb - 1.5, avg_slope))
        if not complement:
            avg_slope = max(0.0, avg_slope)
        ln_e_tgt = val_last + avg_slope * (u_tgt - u_last)
        if complement:
            med = 100.0 * (1.0 - math.exp(ln_e_tgt) / math.exp(ln_n_tgt))
        else:
            med = 100.0 * math.exp(ln_e_tgt - ln_n_tgt)
        med = clamp(med)
        resid = max(sd, 0.05)
        extrap = 0.22 * max(0.0, u_tgt - u_last)
        lag_term = abs(avg_slope - n_slope) * du_per_h * lag_h
        sigma = math.sqrt(resid ** 2 + sd_n ** 2 + extrap ** 2 + lag_term ** 2 + 0.09)
        sigma = min(3.0, max(0.25, sigma))
        out.append((med, sigma))
    return out


def pinball_loss(rows, truth, mult, z_scores):
    """Mean pinball loss on the logit scale of a median+spread forecast, scaled by mult."""
    total = 0.0
    n = 0
    for (med, sig), y in zip(rows, truth):
        if y is None:
            continue
        lg = logit(med)
        sg = sig * mult
        for tau, z in zip(LEVELS, z_scores):
            q = inv_logit(lg + z * sg)
            d = logit(y) - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n if n else None


def calibrate_spread(case, states, t0, z_scores):
    """Rolling-origin backtests inside the case's history: pick the spread multiplier
    (0.5..3.0) minimising mean pinball loss. Uses forecast_medians only, no recursion."""
    if len(states) < 6:
        return 1.0
    mults = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    tot = {m: 0.0 for m in mults}
    cnt = 0
    cand = list(range(5, len(states)))
    if len(cand) > 6:
        step = len(cand) / 6.0
        cand = [cand[int(k * step)] for k in range(6)]
    for oi in cand:
        origin_t = states[oi][0]
        for horizon_h in (6.0, 12.0, 24.0):
            tgt_ts = origin_t.timestamp() + horizon_h * 3600
            j = next((k for k in range(oi + 1, len(states))
                      if states[k][0].timestamp() >= tgt_ts - 1), None)
            if j is None:
                continue
            sub = {"trophies": case["trophies"],
                   "target": states[j][0].isoformat().replace("+00:00", "Z")}
            try:
                meds = forecast_medians(sub, states[:oi + 1], t0, states[oi][0])
            except Exception:
                continue
            for m in mults:
                loss = pinball_loss(meds, states[j][2], m, z_scores)
                if loss is not None:
                    tot[m] += loss
                    cnt += 1
    if cnt == 0:
        return 1.0
    return min(mults, key=lambda m: tot[m])


def frozen_forecast(case, states, t0, t_last):
    """Dilution prior for a history with a single distinct state: the player base grows,
    new players earn each trophy at a speed-class share of its last rate. The horizon is
    measured from the last read (the checkpoint), not from when the page last changed."""
    trophies = case["trophies"]
    t_target = parse_time(case["target"])
    horizon_h = (t_target - t_last).total_seconds() / 3600.0
    n_last = states[-1][1]
    growth = min(30.0, 1.0 + 1.5 * horizon_h)
    n_tgt = n_last * growth
    rows = []
    for i, tr in enumerate(trophies):
        r_last = states[-1][2][i]
        if r_last is None:
            r_last = 0.0
        text = (tr.get("name", "") + " " + tr.get("description", "")).lower()
        if r_last >= 99.99:
            m_last = max(0.0, (100.0 - r_last) / 100.0 * n_last)
            m_tgt = m_last + (n_tgt - n_last) * 0.002
            med = 100.0 * (1.0 - m_tgt / n_tgt)
        else:
            speed = speed_class(text, tr.get("tier", ""))
            share = min(speed * (r_last / 100.0) * (horizon_h / 26.0), 0.5)
            e_last = r_last / 100.0 * n_last
            e_tgt = e_last + (n_tgt - n_last) * share
            med = 100.0 * e_tgt / n_tgt
        rows.append((clamp(med), FROZEN_SIGMA))
    return rows


def house_baseline_rows(case):
    """Persistence with a random-walk spread: the fallback when anything else fails."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        rate = last["rates"][i]
        rows.append((rate if rate is not None else 0.0, spread))
    return rows


def forecast_case(case):
    levels = case.get("levels", LEVELS)
    z_scores = [norm_ppf(t) for t in levels]
    release = (case.get("game") or {}).get("release_date", "2026-11-19")
    t0 = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
    states = distinct_states(case)
    t_last = parse_time(case["history"][-1]["at"])
    if len(states) <= 1:
        meds = frozen_forecast(case, states, t0, t_last)
    else:
        meds = forecast_medians(case, states, t0, t_last)
        mult = calibrate_spread(case, states, t0, z_scores)
        meds = [(med, min(3.0, sig * mult)) for med, sig in meds]
    rows = []
    for i, tr in enumerate(case["trophies"]):
        med, sig = meds[i]
        lg = logit(med)
        rows.append({"id": tr["id"],
                     "q": [round4(inv_logit(lg + z * sig)) for z in z_scores]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        levels = case.get("levels", LEVELS)
        z_scores = [norm_ppf(t) for t in levels]
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case without a valid output: fall back to persistence.
            meds = house_baseline_rows(case)
            rows = []
            for i, tr in enumerate(case["trophies"]):
                med, sig = meds[i]
                lg = logit(med)
                rows.append({"id": tr["id"],
                             "q": [round4(inv_logit(lg + z * sig)) for z in z_scores]})
            forecast = {"trophies": rows}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
