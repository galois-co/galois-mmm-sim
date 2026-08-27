"""Simulate a synthetic DTC fashion business with known ground truth.

Usage
    python simulate.py                 # all regimes in config.json
    python simulate.py --regime C      # one regime
    python simulate.py --seed 7        # override the config seed

Outputs, per regime, in data/regime_<X>/
    observed.csv        what a practitioner would receive (revenue, spend, calendar, discount)
    truth.csv           what nobody observes (baseline, sentiment, contributions, adstock, noise)
    truth.json          parameters, realised ROI and mROI per channel, shares, spend correlations
    decomposition.png   stacked contributions and spend, for the day-one sanity check

Design
- Baseline (no media) = level * trend * seasonality * events * discount elasticity
  * unobserved AR(1) sentiment. It never depends on media.
- Media contribution of channel c in week t = beta_c * Hill(adstock_c,t ; K_c, S_c),
  with normalised geometric adstock so K is in spend units.
- beta_c is calibrated on the planned spend path so that ROI matches roi_target;
  realised ROI on the endogenous path is recorded, not assumed.
- The week loop is sequential because regimes C and D make spend depend on past revenue.
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from transforms import adstock_weights, fourier_terms, hill

WEEKS_PER_YEAR = 52.1775


# --------------------------------------------------------------------------- calendar

def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """weekday: Monday=0 ... Sunday=6."""
    d = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(days=7 * (n - 1))


def build_calendar(start: date, n_weeks: int) -> pd.DataFrame:
    """Weekly event flags for a French fashion retailer. Weeks start on Monday."""
    weeks = [start + timedelta(days=7 * i) for i in range(n_weeks)]
    cal = pd.DataFrame({"week_start": weeks})
    for col in ["black_friday", "black_friday_next", "christmas", "post_christmas",
                "soldes_early", "soldes_late", "rentree", "august"]:
        cal[col] = 0

    def week_index(d: date):
        i = (d - start).days // 7
        return i if 0 <= i < n_weeks else None

    def flag(col, d, span=1):
        i = week_index(d)
        if i is None:
            return
        for j in range(i, min(i + span, n_weeks)):
            cal.loc[j, col] = 1

    years = range(start.year, weeks[-1].year + 2)
    for y in years:
        bf = last_weekday_of_month(y, 11, 4)                 # last Friday of November
        flag("black_friday", bf)
        flag("black_friday_next", bf + timedelta(days=7))
        flag("christmas", date(y, 12, 24) - timedelta(days=7), span=2)
        flag("post_christmas", date(y, 12, 24) + timedelta(days=7))
        wsale = nth_weekday_of_month(y, 1, 2, 2)              # second Wednesday of January
        flag("soldes_early", wsale, span=2)
        flag("soldes_late", wsale + timedelta(days=14), span=2)
        ssale = last_weekday_of_month(y, 6, 2)                # last Wednesday of June
        flag("soldes_early", ssale, span=2)
        flag("soldes_late", ssale + timedelta(days=14), span=2)
        flag("rentree", date(y, 9, 1), span=2)
    cal["august"] = [int(w.month == 8) for w in weeks]
    cal["week_of_year"] = [int(w.isocalendar()[1]) for w in weeks]
    cal["quarter_index"] = np.arange(n_weeks) // 13
    return cal


# --------------------------------------------------------------------------- baseline

def build_baseline(cfg: dict, cal: pd.DataFrame, rng: np.random.Generator, misspec: bool):
    b = cfg["business"]
    n = len(cal)
    t = np.arange(n)

    log_det = np.zeros(n)
    log_det += np.log1p(b["trend_annual"]) * t / WEEKS_PER_YEAR
    F = fourier_terms(n, order=len(b["seasonality_log_coefs"]) // 2)
    log_det += F @ np.array(b["seasonality_log_coefs"])
    for ev, eff in b["event_log_effects"].items():
        log_det += eff * cal[ev].to_numpy()

    d = b["discount"]
    discount = d["base"] + rng.normal(0, d["noise_sd"], n)
    discount = np.where(cal["soldes_early"] == 1, d["soldes_early"], discount)
    discount = np.where(cal["soldes_late"] == 1, d["soldes_late"], discount)
    discount = np.where(cal["black_friday"] == 1, d["black_friday"], discount)
    discount = np.where(cal["black_friday_next"] == 1, d["black_friday_next"], discount)
    discount = np.clip(discount, 0.0, 0.6)
    log_det += np.log1p(d["elasticity"] * (discount - d["base"]))

    competitor = np.zeros(n)
    if misspec:
        m = cfg["misspecification"]["competitor_shock"]
        starts = rng.choice(np.arange(8, n - m["length_weeks"]), size=m["episodes"], replace=False)
        for s0 in starts:
            competitor[s0:s0 + m["length_weeks"]] = m["log_effect"]
        log_det += competitor

    # Scale so that the deterministic baseline averages the target weekly level.
    target_weekly = b["baseline_annual_eur"] / WEEKS_PER_YEAR
    det = np.exp(log_det)
    det *= target_weekly / det.mean()

    s = b["sentiment"]
    sentiment = np.zeros(n)
    innov = rng.normal(0, s["sd"] * np.sqrt(1 - s["phi"] ** 2), n)
    for i in range(1, n):
        sentiment[i] = s["phi"] * sentiment[i - 1] + innov[i]
    baseline = det * np.exp(sentiment)
    return baseline, det, sentiment, discount, competitor


# --------------------------------------------------------------------------- planned spend

def budget_multiplier(cfg: dict, cal: pd.DataFrame, regime: dict, rng: np.random.Generator):
    """Common budget cycle across channels (regimes B, C, D) plus a common shock."""
    n = len(cal)
    mult = np.ones(n)
    if regime["budget_follows_season"]:
        bs = cfg["media"]["budget_season"]
        F = fourier_terms(n, order=len(cfg["business"]["seasonality_log_coefs"]) // 2)
        mult *= np.exp(bs["fourier_amplification"] * (F @ np.array(cfg["business"]["seasonality_log_coefs"])))
        for ev, m in bs["event_multipliers"].items():
            mult *= np.where(cal[ev] == 1, m, 1.0)
    if regime["common_shock_sd"] > 0:
        mult *= np.exp(rng.normal(0, regime["common_shock_sd"], n))
    return mult / mult.mean()


def channel_pattern(name: str, spec: dict, cfg: dict, cal: pd.DataFrame, regime: dict,
                    rng: np.random.Generator) -> np.ndarray:
    n = len(cal)
    pat = np.ones(n)
    p = cfg["media"]["patterns"]
    years = n / WEEKS_PER_YEAR
    if spec["pattern"] == "bursty":
        b = p["bursty"]
        n_bursts = int(round(b["bursts_per_year"] * years))
        starts = rng.choice(np.arange(0, n - b["burst_weeks"]), size=n_bursts, replace=False)
        for s0 in starts:
            pat[s0:s0 + b["burst_weeks"]] *= b["burst_multiplier"]
    elif spec["pattern"] == "flights":
        pat = np.zeros(n)
        if regime["video_flights"] == "random":
            f = p["flights_random"]
            n_flights = int(round(f["flights_per_year"] * years))
            starts = rng.choice(np.arange(0, n - f["flight_weeks"], f["flight_weeks"]),
                                size=n_flights, replace=False)
            for s0 in starts:
                pat[s0:s0 + f["flight_weeks"]] = 1.0
        else:
            on = set(p["flights_calendar_weeks_of_year"])
            pat = np.array([1.0 if w in on else 0.0 for w in cal["week_of_year"]])
    return pat


def planned_spend(cfg: dict, cal: pd.DataFrame, regime: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Spend each channel would have received without any feedback from revenue."""
    n = len(cal)
    weekly_budget = cfg["media"]["annual_budget_eur"] / WEEKS_PER_YEAR
    mult = budget_multiplier(cfg, cal, regime, rng)
    out = {}
    for name, spec in cfg["media"]["channels"].items():
        pat = channel_pattern(name, spec, cfg, cal, regime, rng)
        noise = np.exp(rng.normal(0, regime["channel_noise_sd"], n))
        x = mult * pat * noise
        x *= (weekly_budget * spec["share"]) / x.mean()      # respect the channel's budget share
        out[name] = x
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- media response

def channel_weights(name: str, spec: dict, cfg: dict, misspec: bool) -> np.ndarray:
    if misspec and name == "video":
        w = np.array(cfg["misspecification"]["video_delayed_weights"], dtype=float)
        return w / w.sum()
    return adstock_weights(spec["decay"], cfg["media"]["l_max"], normalize=True)


def adstock_at(x_hist: np.ndarray, t: int, w: np.ndarray) -> float:
    """Adstock of week t from the spend history x_hist[0..t]."""
    lags = min(len(w), t + 1)
    return float(np.dot(w[:lags], x_hist[t - np.arange(lags)]))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, 0.0 when either series is constant."""
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def hill_derivative(x: float, k: float, s: float) -> float:
    return s * x ** (s - 1) * k ** s / (k ** s + x ** s) ** 2 if x > 0 else 0.0


def calibrate_betas(cfg: dict, planned: pd.DataFrame, weights: dict, k_abs: dict) -> dict:
    """Choose beta so that ROI on the planned path equals roi_target."""
    betas = {}
    for name, spec in cfg["media"]["channels"].items():
        x = planned[name].to_numpy()
        a = np.array([adstock_at(x, t, weights[name]) for t in range(len(x))])
        h = hill(a, k_abs[name], spec["s"])
        betas[name] = spec["roi_target"] * x.sum() / h.sum()
    return betas


# --------------------------------------------------------------------------- simulation loop

def simulate_regime(cfg: dict, regime_key: str, seed: int, out_root: Path) -> dict:
    regime = cfg["regimes"][regime_key]
    misspec = regime["misspecification"]
    rng = np.random.default_rng([seed, ord(regime_key)])
    start = date.fromisoformat(cfg["start_date"])
    n = cfg["n_weeks"]
    cal = build_calendar(start, n)
    channels = list(cfg["media"]["channels"].keys())
    specs = cfg["media"]["channels"]
    endo = cfg["endogeneity"]

    baseline, det_baseline, sentiment, discount, competitor = build_baseline(cfg, cal, rng, misspec)
    planned = planned_spend(cfg, cal, regime, rng)

    weights = {c: channel_weights(c, specs[c], cfg, misspec) for c in channels}
    k_abs = {c: specs[c]["k_rel"] * planned[c].mean() for c in channels}
    betas = calibrate_betas(cfg, planned, weights, k_abs)

    # Planned contributions, used for the noise scale and the brand-search demand proxy.
    planned_contrib = np.zeros((n, len(channels)))
    for j, c in enumerate(channels):
        x = planned[c].to_numpy()
        a = np.array([adstock_at(x, t, weights[c]) for t in range(n)])
        planned_contrib[:, j] = betas[c] * hill(a, k_abs[c], specs[c]["s"])
    mean_revenue_planned = baseline.mean() + planned_contrib.sum(axis=1).mean()
    noise_sd = cfg["business"]["noise_sd_share"] * mean_revenue_planned
    noise = rng.normal(0, noise_sd, n)
    others_mean = planned_contrib.sum(axis=1).mean() - planned_contrib[:, channels.index("search_brand")].mean()

    # Anticipatory weeks: one or two weeks before Black Friday, Christmas and the winter/summer sales.
    upcoming = np.zeros(n, dtype=bool)
    if regime["anticipatory_spend"]:
        starts = ((cal["black_friday"] == 1) | (cal["soldes_early"].diff().fillna(cal["soldes_early"]) == 1)
                  | (cal["christmas"].diff().fillna(cal["christmas"]) == 1)).to_numpy()
        for t in np.where(starts)[0]:
            upcoming[max(0, t - 2):t] = True

    spend = np.zeros((n, len(channels)))
    adstock = np.zeros((n, len(channels)))
    contrib = np.zeros((n, len(channels)))
    revenue = np.zeros(n)
    quarter_scale = np.ones(n)
    brand_noise = np.exp(rng.normal(0, endo["brand_search_noise_sd"], n))

    for t in range(n):
        # Quarterly budget feedback: budgets follow revenue growth of the previous quarter.
        if regime["quarterly_feedback"] and t >= 26 and t % 13 == 0:
            last_q = revenue[t - 13:t].mean()
            prev_q = revenue[t - 26:t - 13].mean()
            lo, hi = endo["quarterly_feedback_clip"]
            quarter_scale[t:t + 13] = np.clip((last_q / prev_q) ** endo["quarterly_feedback_exponent"], lo, hi)

        row = planned.iloc[t].to_numpy() * quarter_scale[t]

        if regime["anticipatory_spend"] and upcoming[t]:
            for c in endo["anticipatory_channels"]:
                row[channels.index(c)] *= endo["anticipatory_multiplier"]

        if regime["performance_chasing"] and t >= 2:
            lo, hi = endo["chasing_clip"]
            growth = np.clip((revenue[t - 1] / revenue[t - 2]) ** endo["chasing_exponent"], lo, hi)
            row[channels.index("social")] *= growth

        if regime["endogenous_brand_search"]:
            # Automated bidding on brand queries scales with demand the marketer does not observe
            # (sentiment) and with the awareness created by last week's other channels.
            jb = channels.index("search_brand")
            surprise = (baseline[t] / det_baseline[t]) ** endo["brand_search_demand_exponent"]
            others = contrib[t - 1].sum() - contrib[t - 1, jb] if t > 0 else others_mean
            awareness = (max(others, 1.0) / others_mean) ** endo["brand_search_media_exponent"]
            row[jb] = row[jb] * surprise * awareness * brand_noise[t]

        spend[t] = row
        for j, c in enumerate(channels):
            adstock[t, j] = adstock_at(spend[:, j], t, weights[c])
            contrib[t, j] = betas[c] * hill(adstock[t, j], k_abs[c], specs[c]["s"])
        revenue[t] = baseline[t] + contrib[t].sum() + noise[t]

    # What the practitioner observes.
    observed_spend = spend.copy()
    if misspec:
        m = cfg["misspecification"]
        observed_spend *= np.exp(rng.normal(0, m["spend_measurement_error_sd"], spend.shape))
        observed_spend[:, channels.index("display")] *= (1 + m["display_fee_markup"])

    # ---- outputs
    out = out_root / f"regime_{regime_key}"
    out.mkdir(parents=True, exist_ok=True)

    observed = cal[["week_start", "black_friday", "black_friday_next", "christmas", "post_christmas",
                    "soldes_early", "soldes_late", "rentree", "august"]].copy()
    observed["revenue"] = np.round(revenue, 2)
    observed["discount_depth"] = np.round(discount, 4)
    for j, c in enumerate(channels):
        observed[f"spend_{c}"] = np.round(observed_spend[:, j], 2)
    observed.to_csv(out / "observed.csv", index=False)

    truth = pd.DataFrame({"week_start": cal["week_start"], "baseline": baseline,
                          "baseline_deterministic": det_baseline, "sentiment": sentiment,
                          "competitor_shock": competitor, "noise": noise, "quarter_scale": quarter_scale})
    for j, c in enumerate(channels):
        truth[f"spend_true_{c}"] = spend[:, j]
        truth[f"spend_planned_{c}"] = planned[c].to_numpy()
        truth[f"adstock_{c}"] = adstock[:, j]
        truth[f"contrib_{c}"] = contrib[:, j]
    truth.to_csv(out / "truth.csv", index=False)

    total_contrib = contrib.sum()
    summary = {
        "regime": regime_key, "label": regime["label"], "seed": seed, "n_weeks": n,
        "mean_weekly_revenue": float(revenue.mean()),
        "baseline_share": float(baseline.sum() / revenue.sum()),
        "noise_sd": float(noise_sd),
        "channels": {},
        "spend_correlation": pd.DataFrame(observed_spend, columns=channels).corr().round(3).to_dict(),
    }
    for j, c in enumerate(channels):
        x = spend[:, j]
        summary["channels"][c] = {
            "decay": specs[c]["decay"] if not (misspec and c == "video") else None,
            "adstock_weights": [float(v) for v in weights[c]],
            "k_abs": float(k_abs[c]), "k_rel": specs[c]["k_rel"], "s": specs[c]["s"],
            "beta": float(betas[c]),
            "mean_weekly_spend_true": float(x.mean()),
            "mean_weekly_spend_observed": float(observed_spend[:, j].mean()),
            "roi_target": specs[c]["roi_target"],
            "roi_realised": float(contrib[:, j].sum() / x.sum()),
            "mroi_at_mean_spend": float(betas[c] * hill_derivative(x.mean(), k_abs[c], specs[c]["s"])),
            "contribution_share_of_media": float(contrib[:, j].sum() / total_contrib),
            "contribution_share_of_revenue": float(contrib[:, j].sum() / revenue.sum()),
            "corr_spend_sentiment": float(np.corrcoef(x, sentiment)[0, 1]),
            "corr_spend_deviation_sentiment": safe_corr(
                np.log(np.maximum(x, 1.0) / np.maximum(planned[c].to_numpy(), 1.0)), sentiment),
        }
    (out / "truth.json").write_text(json.dumps(summary, indent=2))
    plot_decomposition(cal, baseline, contrib, revenue, observed_spend, channels, out / "decomposition.png",
                       f"Regime {regime_key}: {regime['label']}")
    return summary


# --------------------------------------------------------------------------- reporting

def plot_decomposition(cal, baseline, contrib, revenue, spend, channels, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = pd.to_datetime(cal["week_start"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].stackplot(x, baseline / 1e3, *(contrib[:, j] / 1e3 for j in range(len(channels))),
                      labels=["baseline"] + channels, alpha=0.85)
    axes[0].plot(x, revenue / 1e3, color="black", lw=1.2, label="revenue (with noise)")
    axes[0].set_ylabel("EUR thousand / week")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left", ncol=4, fontsize=8)
    axes[1].stackplot(x, *(spend[:, j] / 1e3 for j in range(len(channels))), labels=channels, alpha=0.85)
    axes[1].set_ylabel("observed spend, EUR thousand / week")
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_summary(s: dict):
    print(f"\nRegime {s['regime']} ({s['label']}), seed {s['seed']}")
    print(f"  mean weekly revenue  {s['mean_weekly_revenue']:>12,.0f} EUR")
    print(f"  baseline share       {s['baseline_share']:>12.1%}")
    print(f"  noise sd             {s['noise_sd']:>12,.0f} EUR")
    print(f"  {'channel':16s} {'spend/wk':>10s} {'ROI tgt':>8s} {'ROI real':>9s} {'mROI':>7s} {'share':>7s} "
          f"{'corr(spend,sent)':>17s} {'corr(dev,sent)':>15s}")
    for c, v in s["channels"].items():
        print(f"  {c:16s} {v['mean_weekly_spend_true']:>10,.0f} {v['roi_target']:>8.2f} {v['roi_realised']:>9.2f} "
              f"{v['mroi_at_mean_spend']:>7.2f} {v['contribution_share_of_media']:>7.1%} "
              f"{v['corr_spend_sentiment']:>17.2f} {v['corr_spend_deviation_sentiment']:>15.2f}")
    corr = pd.DataFrame(s["spend_correlation"])
    off = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    print(f"  spend correlation, off-diagonal: mean {off.mean():.2f}, min {off.min():.2f}, max {off.max():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--regime", default=None, help="A, B, C or D; default all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else cfg["seed"]
    regimes = [args.regime] if args.regime else list(cfg["regimes"].keys())
    for r in regimes:
        s = simulate_regime(cfg, r, seed, Path(args.out))
        print_summary(s)


if __name__ == "__main__":
    main()
