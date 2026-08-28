"""M2 and M3: Bayesian marketing mix model written directly in PyMC.

The model
    y_t / y_mean = intercept + controls_t . gamma
                   + sum_c beta_c * Hill( Adstock(x_c,t / x_c_mean ; decay_c) ; k_c, s_c ) + eps_t

Spend is scaled by each channel's mean, so k_c is directly the half-saturation point in
multiples of typical weekly spend, and beta_c is the channel's maximum weekly contribution in
multiples of mean weekly revenue. Adstock is geometric, normalised, with a 13-week window.

Priors (all documented here because they are part of the model, not a detail)
    decay_c   Beta, centred by channel class: search ~0.2, social/display/affiliate ~0.4, video ~0.65
    k_c       LogNormal(log 1, 0.5): half-saturation around typical weekly spend
    s_c       LogNormal(log 1, 0.4): Hill slope around 1
    beta_c    HalfNormal with scale set so that the implied ROI prior is centred near 2
    gamma     Normal(0, 0.2) on standardised controls; intercept Normal(0.6, 0.3)
    sigma     HalfNormal(0.1)
--prior-scale multiplies every prior width (0.5 = tighter, 2 = looser) for sensitivity runs.

M3 = M2 plus calibration: the go-dark lift test in lift_test.json enters as an extra
likelihood term on the model-implied incremental revenue over the test window.

Usage
    python fit_bayes.py --regime C                    # M2
    python fit_bayes.py --regime C --calibrate        # M3
    python fit_bayes.py --regime C --prior-scale 2    # sensitivity
"""

import argparse
import json
import time
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from features import (baseline_design, compare_to_truth, load_observed, nrmse, print_comparison, r2,
                      save_json, spend_matrix)
from transforms import hill

CHANNEL_CLASS = {"search_nonbrand": "search", "search_brand": "search", "social": "digital",
                 "display": "digital", "affiliate": "digital", "video": "video"}
DECAY_PRIOR_MEAN = {"search": 0.20, "digital": 0.40, "video": 0.65}
DECAY_PRIOR_CONCENTRATION = 10.0     # Beta(a, b) with a + b = concentration
ROI_PRIOR_CENTRE = 2.0
EPS = 1e-6


def lagged_stack(x: np.ndarray, l_max: int) -> np.ndarray:
    """Shape (l_max, n, C): lag l of x, zero-padded, so adstock = sum_l w_l * stack[l]."""
    n, C = x.shape
    out = np.zeros((l_max, n, C))
    for lag in range(l_max):
        out[lag, lag:, :] = x[: n - lag, :]
    return out


def build_model(y_s, X_ctrl, x_s, spend_scale_ratio, channels, l_max, prior_scale, lift=None, x_cf_s=None):
    n, C = x_s.shape
    lags = np.arange(l_max, dtype=float)
    X_lag = lagged_stack(x_s, l_max)
    X_lag_cf = lagged_stack(x_cf_s, l_max) if x_cf_s is not None else None

    a_prior = np.array([DECAY_PRIOR_MEAN[CHANNEL_CLASS[c]] for c in channels]) * DECAY_PRIOR_CONCENTRATION / prior_scale
    b_prior = (1 - np.array([DECAY_PRIOR_MEAN[CHANNEL_CLASS[c]] for c in channels])) * DECAY_PRIOR_CONCENTRATION / prior_scale

    with pm.Model(coords={"channel": channels, "control": [f"c{i}" for i in range(X_ctrl.shape[1] - 1)]}) as model:
        decay = pm.Beta("decay", alpha=a_prior, beta=b_prior, dims="channel")
        k = pm.LogNormal("k", mu=0.0, sigma=0.5 * prior_scale, dims="channel")
        s = pm.LogNormal("s", mu=0.0, sigma=0.4 * prior_scale, dims="channel")
        # ROI ~ 2 with Hill ~ 0.5 at typical spend means beta ~ 4 * (mean spend / mean revenue).
        beta = pm.HalfNormal("beta", sigma=2 * ROI_PRIOR_CENTRE * spend_scale_ratio * prior_scale, dims="channel")
        intercept = pm.Normal("intercept", mu=0.6, sigma=0.3 * prior_scale)
        gamma = pm.Normal("gamma", mu=0.0, sigma=0.2 * prior_scale, dims="control")
        sigma = pm.HalfNormal("sigma", sigma=0.1 * prior_scale)

        w = decay[None, :] ** lags[:, None]                     # (l_max, C)
        w = w / w.sum(axis=0, keepdims=True)
        adstock = pt.sum(w[:, None, :] * X_lag, axis=0) + EPS   # (n, C); EPS keeps Hill's gradient finite at zero spend
        sat = adstock ** s / (k ** s + adstock ** s)
        media = sat * beta                                      # (n, C), in units of mean revenue
        mu = intercept + pt.dot(X_ctrl[:, 1:], gamma) + media.sum(axis=1)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_s)
        pm.Deterministic("contrib_total", media.sum(axis=0), dims="channel")

        if lift is not None:
            j = lift["j"]
            adstock_cf = pt.sum(w[:, None, :] * X_lag_cf, axis=0) + EPS
            sat_cf = adstock_cf ** s / (k ** s + adstock_cf ** s)
            w0, w1 = lift["w0"], lift["w1"]
            implied = pt.sum(sat[w0:w1, j] * beta[j] - sat_cf[w0:w1, j] * beta[j])
            pm.Normal("lift", mu=implied, sigma=lift["sd_s"], observed=lift["measured_s"])
    return model


def fit_bayes(regime_dir: Path, out_dir: Path, args) -> dict:
    t0 = time.time()
    obs, channels = load_observed(regime_dir)
    y = obs["revenue"].to_numpy(dtype=float)
    spend = spend_matrix(obs, channels)
    X_ctrl, ctrl_names = baseline_design(obs)
    n, C = spend.shape

    y_mean = y.mean()
    y_s = y / y_mean
    x_mean = spend.mean(axis=0)
    x_s = spend / x_mean
    spend_scale_ratio = x_mean / y_mean                         # mean spend as a share of mean revenue

    lift, x_cf_s = None, None
    if args.calibrate:
        lt = json.loads((regime_dir / "lift_test.json").read_text())
        j = channels.index(lt["channel"])
        x_cf = spend.copy()
        x_cf[lt["start_week"]:lt["end_week_exclusive"], j] = 0.0
        x_cf_s = x_cf / x_mean
        lift = {"j": j, "w0": lt["start_week"], "w1": lt["end_week_exclusive"],
                "measured_s": lt["incremental_revenue_measured"] / y_mean,
                "sd_s": lt["measurement_sd"] / y_mean}

    model = build_model(y_s, X_ctrl, x_s, spend_scale_ratio, channels, args.l_max, args.prior_scale, lift, x_cf_s)
    label = "M3 bayes + lift calibration" if args.calibrate else "M2 bayes"
    if args.prior_scale != 1.0:
        label += f" (prior scale {args.prior_scale:g})"
    print(f"\n{label}, regime {regime_dir.name[-1]}: sampling {args.chains} chains x {args.draws} draws ...")
    with model:
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
                          target_accept=args.target_accept, random_seed=args.seed, progressbar=False)

    post = idata.posterior
    decay = post["decay"].values.reshape(-1, C)
    k = post["k"].values.reshape(-1, C)
    s = post["s"].values.reshape(-1, C)
    beta = post["beta"].values.reshape(-1, C)
    contrib_total = post["contrib_total"].values.reshape(-1, C) * y_mean     # EUR over the sample
    n_draws = decay.shape[0]

    # ROI draws, marginal ROI at mean spend, response curves; all from posterior draws.
    roi = contrib_total / spend.sum(axis=0)
    mroi = np.zeros_like(roi)
    curves = {}
    for j, c in enumerate(channels):
        xm = 1.0                                                   # mean spend in scaled units
        mroi[:, j] = beta[:, j] * (s[:, j] * xm ** (s[:, j] - 1) * k[:, j] ** s[:, j]
                                   / (k[:, j] ** s[:, j] + xm ** s[:, j]) ** 2) * y_mean / x_mean[j]
        grid = np.linspace(0, 2.5, 26)
        resp = np.stack([beta[d, j] * hill(grid, k[d, j], s[d, j]) * y_mean for d in range(0, n_draws, max(1, n_draws // 400))])
        curves[c] = {"spend": (grid * x_mean[j]).round(2).tolist(),
                     "revenue_median": np.median(resp, axis=0).round(2).tolist(),
                     "revenue_hdi5": np.percentile(resp, 5, axis=0).round(2).tolist(),
                     "revenue_hdi95": np.percentile(resp, 95, axis=0).round(2).tolist()}

    total_media = contrib_total.sum(axis=1)
    summary = {}
    for j, c in enumerate(channels):
        summary[c] = {
            "beta": float(np.median(beta[:, j]) * y_mean),
            "decay": float(np.median(decay[:, j])), "k_rel": float(np.median(k[:, j])), "s": float(np.median(s[:, j])),
            "roi": float(np.median(roi[:, j])),
            "roi_hdi5": float(np.percentile(roi[:, j], 5)), "roi_hdi95": float(np.percentile(roi[:, j], 95)),
            "mroi_at_mean_spend": float(np.median(mroi[:, j])),
            "contribution_share_of_media": float(np.median(contrib_total[:, j] / total_media)),
            "response_curve": curves[c],
            "roi_draws": roi[:: max(1, n_draws // 500), j].round(4).tolist(),
        }

    # Fit quality of the posterior-mean prediction and sampler diagnostics.
    with model:
        ppc = pm.sample_posterior_predictive(idata, var_names=["y"], random_seed=args.seed, progressbar=False)
    y_hat = ppc.posterior_predictive["y"].values.reshape(-1, n).mean(axis=0) * y_mean
    stats = az.summary(idata, var_names=["decay", "k", "s", "beta", "sigma"])
    fit = {"r2_in_sample": r2(y, y_hat), "nrmse_in_sample": nrmse(y, y_hat),
           "r_hat_max": float(stats["r_hat"].max()), "ess_bulk_min": float(stats["ess_bulk"].min()),
           "divergences": int(idata.sample_stats["diverging"].values.sum()),
           "prior_scale": args.prior_scale, "calibrated": bool(args.calibrate),
           "sampling_seconds": round(time.time() - t0, 1)}

    cmp = compare_to_truth(summary, regime_dir, channels)
    cmp["roi_in_90pct_interval"] = {c: bool(summary[c]["roi_hdi5"] <= cmp["roi_true"][c] <= summary[c]["roi_hdi95"])
                                    for c in channels}
    cmp["coverage_90"] = float(np.mean(list(cmp["roi_in_90pct_interval"].values())))
    print_comparison(f"{label}, regime {regime_dir.name[-1]}", summary, cmp, channels, fit)
    print("  90% intervals on ROI:")
    for c in channels:
        flag = "in" if cmp["roi_in_90pct_interval"][c] else "OUT"
        print(f"    {c:16s} [{summary[c]['roi_hdi5']:.2f}, {summary[c]['roi_hdi95']:.2f}]  truth {cmp['roi_true'][c]:.2f}  {flag}")
    print(f"  coverage of the 90% intervals: {cmp['coverage_90']:.0%}")

    name = "bayes_calibrated" if args.calibrate else "bayes"
    if args.prior_scale != 1.0:
        name += f"_prior{args.prior_scale:g}"
    result = {"model": "M3_bayes_calibrated" if args.calibrate else "M2_bayes", "regime": regime_dir.name[-1],
              "fit": fit, "sampler": {"draws": args.draws, "tune": args.tune, "chains": args.chains,
                                       "target_accept": args.target_accept, "seed": args.seed},
              "priors": {"decay_prior_mean_by_class": DECAY_PRIOR_MEAN, "decay_concentration": DECAY_PRIOR_CONCENTRATION,
                         "k": "LogNormal(0, 0.5)", "s": "LogNormal(0, 0.4)", "roi_prior_centre": ROI_PRIOR_CENTRE,
                         "prior_scale": args.prior_scale},
              "channels": summary, "comparison_to_truth": cmp}
    save_json(result, out_dir / regime_dir.name / f"{name}.json")
    if args.save_trace:
        idata.to_netcdf(str(out_dir / regime_dir.name / f"{name}_trace.nc"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--regime", default=None)
    parser.add_argument("--calibrate", action="store_true", help="add the lift test as a likelihood term (M3)")
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--l-max", type=int, default=13)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-trace", action="store_true")
    args = parser.parse_args()
    dirs = [Path(args.data) / f"regime_{args.regime}"] if args.regime else sorted(Path(args.data).glob("regime_*"))
    for d in dirs:
        fit_bayes(d, Path(args.out), args)


if __name__ == "__main__":
    main()
