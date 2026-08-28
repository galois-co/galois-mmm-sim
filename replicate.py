"""Replication study: how estimation error moves with spend collinearity and endogeneity.

Two sweeps, each replicated over many seeds:
    collinearity   budget_cycle_weight in {0, 0.25, 0.5, 0.75, 1}, no endogeneity
                   (spend correlation rises from ~0 to ~0.8)
    endogeneity    full cycle plus the regime-C mechanisms (brand search follows unobserved
                   demand, performance chasing, anticipatory spend, quarterly feedback)

For every simulated dataset, M0 (OLS) and M1 (ridge, reduced search budget) are fitted; with
--bayes, M2 is fitted too (slow: several minutes per seed, run it in the background with a
small --n-seeds). Each row of results/replications.csv records the realised spend correlation,
ROI absolute error per channel and on average, rank correlation, and the decision regret of
the model's budget allocation scored on the true response.

Usage
    python replicate.py --n-seeds 40                      # M0 and M1, 30 to 45 min on a 2017 laptop
    python replicate.py --bayes --n-seeds 20 --sweep endogeneity --weights 1
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluate import model_response, optimise, true_response
from features import (baseline_design, fit_bounded_ridge, load_observed, media_summary, rolling_origin_folds,
                      spend_matrix, transform_media)
from fit_ridge import search, unpack
from simulate import simulate_regime

CHANNELS = ["search_nonbrand", "search_brand", "social", "display", "video", "affiliate"]


def make_regime(cfg: dict, weight: float, endogenous: bool) -> dict:
    base = dict(cfg["regimes"]["C" if endogenous else "B"])
    base["budget_cycle_weight"] = weight
    # The common shock fades and channel-specific noise grows as the cycle weight falls, so the
    # realised spend correlation spans roughly 0.1 (weight 0) to 0.8 (weight 1).
    base["common_shock_sd"] = 0.15 * weight
    base["channel_noise_sd"] = 0.10 + 0.20 * (1 - weight)
    base["label"] = f"replication w={weight:g} endogenous={endogenous}"
    return base


def fit_models(regime_dir: Path, args, rng) -> dict:
    obs, channels = load_observed(regime_dir)
    y = obs["revenue"].to_numpy(dtype=float)
    spend = spend_matrix(obs, channels)
    X_ctrl, _ = baseline_design(obs)
    folds = rolling_origin_folds(len(y))
    out = {}

    # M0
    X = np.hstack([X_ctrl, spend])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta = coef[X_ctrl.shape[1]:]
    out["ols"] = {"model": "M0_ols", "channels": {c: {"beta": float(beta[j]), "roi": float(beta[j]),
                                                       "k_rel": 0.0, "s": 1.0} for j, c in enumerate(channels)}}

    # M1, reduced search budget
    best, _ = search(spend, X_ctrl, y, folds, args.l_max, rng, args.n_random, args.n_mutate, args.n_rounds,
                     args.top_k, verbose=False)
    params, alpha = unpack(best[0], len(channels))
    H = transform_media(spend, params, args.l_max)
    fit = fit_bounded_ridge(X_ctrl, H, y, alpha)
    out["ridge"] = {"model": "M1_ridge", "channels": media_summary(fit["coef_media"], H, spend, params, channels)}

    # M2, optional
    if args.bayes:
        import types
        from fit_bayes import fit_bayes
        bargs = types.SimpleNamespace(calibrate=False, prior_scale=1.0, l_max=args.l_max, draws=args.draws,
                                      tune=args.tune, chains=args.chains, cores=args.cores, target_accept=0.95,
                                      seed=int(rng.integers(1e6)), save_trace=False)
        res = fit_bayes(regime_dir, regime_dir / "bayes_tmp", bargs)
        out["bayes"] = res
    return out


def score(result: dict, truth: dict, channels: list[str], bound: float) -> dict:
    roi_true = np.array([truth["channels"][c]["roi_realised"] for c in channels])
    roi_est = np.array([result["channels"][c]["roi"] for c in channels])
    mean_spend = np.array([truth["channels"][c]["mean_weekly_spend_observed"] for c in channels])
    f_true, f_model = true_response(truth, channels), model_response(result, channels, mean_spend)
    x_sq = mean_spend.copy()
    rev_sq, rev_opt = f_true(x_sq), f_true(optimise(f_true, x_sq, bound))
    x_rec = optimise(f_model, x_sq, bound)
    row = {"roi_mae": float(np.mean(np.abs(roi_est - roi_true))),
           "roi_rank_corr": float(spearmanr(roi_true, roi_est).correlation),
           "regret_weekly": float(rev_opt - f_true(x_rec)),
           "regret_share_of_gain": float((rev_opt - f_true(x_rec)) / (rev_opt - rev_sq)) if rev_opt > rev_sq else None}
    for c in channels:
        row[f"abs_err_{c}"] = float(abs(result["channels"][c]["roi"] - truth["channels"][c]["roi_realised"]))
    if "coverage_90" in result.get("comparison_to_truth", {}):
        row["coverage_90"] = result["comparison_to_truth"]["coverage_90"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", default="results")
    parser.add_argument("--n-seeds", type=int, default=40)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--sweep", choices=["collinearity", "endogeneity", "both"], default="both")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--bound", type=float, default=0.3)
    parser.add_argument("--l-max", type=int, default=13)
    parser.add_argument("--n-random", type=int, default=1500)
    parser.add_argument("--n-mutate", type=int, default=20)
    parser.add_argument("--n-rounds", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bayes", action="store_true")
    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--tune", type=int, default=800)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    tmp_root = Path(args.out) / "replication_tmp"
    out_path = Path(args.out) / ("replications_bayes.csv" if args.bayes else "replications.csv")
    sweeps = []
    if args.sweep in ("collinearity", "both"):
        sweeps += [("collinearity", w, False) for w in args.weights]
    if args.sweep in ("endogeneity", "both"):
        sweeps += [("endogeneity", w, True) for w in (args.weights if args.sweep == "endogeneity" else [1.0])]

    rows = []
    t0 = time.time()
    total = len(sweeps) * args.n_seeds
    done = 0
    for sweep_name, weight, endogenous in sweeps:
        regime = make_regime(cfg, weight, endogenous)
        for i in range(args.n_seeds):
            seed = args.seed0 + i
            rng = np.random.default_rng(seed)
            shutil.rmtree(tmp_root, ignore_errors=True)
            summary = simulate_regime(cfg, "R", seed, tmp_root, regime_override=regime, plot=False)
            regime_dir = tmp_root / "regime_R"
            corr = pd.DataFrame(summary["spend_correlation"]).to_numpy()
            rho = float(corr[~np.eye(len(corr), dtype=bool)].mean())
            results = fit_models(regime_dir, args, rng)
            for model_name, res in results.items():
                row = {"sweep": sweep_name, "budget_cycle_weight": weight, "endogenous": endogenous,
                       "seed": seed, "spend_corr_mean": rho, "model": model_name}
                row.update(score(res, summary, CHANNELS, args.bound))
                rows.append(row)
            done += 1
            if done % 5 == 0 or done == total:
                pd.DataFrame(rows).to_csv(out_path, index=False)
                el = time.time() - t0
                print(f"  {done}/{total} datasets done, {el / 60:.1f} min elapsed, "
                      f"~{el / done * (total - done) / 60:.1f} min remaining")
    shutil.rmtree(tmp_root, ignore_errors=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print("\nMedian ROI mean absolute error by sweep, cycle weight and model")
    piv = df.pivot_table(index=["sweep", "budget_cycle_weight"], columns="model", values="roi_mae", aggfunc="median")
    print(piv.round(3).to_string())
    print("\nMedian realised spend correlation by cycle weight")
    print(df.groupby(["sweep", "budget_cycle_weight"])["spend_corr_mean"].median().round(2).to_string())
    print("\nMedian rank correlation of channel ROIs")
    print(df.pivot_table(index=["sweep", "budget_cycle_weight"], columns="model", values="roi_rank_corr",
                         aggfunc="median").round(2).to_string())
    print("\nMedian regret as a share of the available gain")
    print(df.pivot_table(index=["sweep", "budget_cycle_weight"], columns="model", values="regret_share_of_gain",
                         aggfunc="median").round(2).to_string())
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
