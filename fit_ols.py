"""M0: ordinary least squares on raw spend, with the baseline controls.

No adstock, no saturation, no sign constraint. This is the model a spreadsheet produces.
Its ROI per channel is simply the coefficient. Kept as the floor every other model must beat.

Usage
    python fit_ols.py                # all regimes found in data/
    python fit_ols.py --regime C
"""

import argparse
from pathlib import Path

import numpy as np

from features import (baseline_design, compare_to_truth, load_observed, nrmse, print_comparison, r2,
                      rolling_origin_folds, save_json, spend_matrix)


def fit_ols(regime_dir: Path, out_dir: Path) -> dict:
    obs, channels = load_observed(regime_dir)
    y = obs["revenue"].to_numpy(dtype=float)
    spend = spend_matrix(obs, channels)
    X_ctrl, ctrl_names = baseline_design(obs)
    X = np.hstack([X_ctrl, spend])

    # Rolling-origin validation, then a full-sample fit.
    cv = []
    for tr, va in rolling_origin_folds(len(y)):
        coef, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        cv.append(nrmse(y[va], X[va] @ coef))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    beta = coef[X_ctrl.shape[1]:]

    # In a linear model contribution = coef * spend, so ROI and mROI are both the coefficient.
    contrib = spend * beta
    total = contrib.sum()
    summary = {}
    for j, c in enumerate(channels):
        grid = np.linspace(0, 2.5 * spend[:, j].mean(), 26)
        summary[c] = {"beta": float(beta[j]), "decay": 0.0, "k_rel": 0.0, "s": 1.0,
                      "roi": float(beta[j]), "mroi_at_mean_spend": float(beta[j]),
                      "contribution_share_of_media": float(contrib[:, j].sum() / total) if total != 0 else 0.0,
                      "response_curve": {"spend": grid.round(2).tolist(),
                                         "revenue": (beta[j] * grid).round(2).tolist()}}

    fit = {"r2_in_sample": r2(y, fitted), "nrmse_cv_mean": float(np.mean(cv)),
           "negative_media_coefficients": int((beta < 0).sum())}
    cmp = compare_to_truth(summary, regime_dir, channels)
    print_comparison(f"M0 OLS, regime {regime_dir.name[-1]}", summary, cmp, channels, fit)

    result = {"model": "M0_ols", "regime": regime_dir.name[-1], "fit": fit,
              "controls": dict(zip(ctrl_names, coef[:X_ctrl.shape[1]].round(2).tolist())),
              "channels": summary, "comparison_to_truth": cmp}
    save_json(result, out_dir / regime_dir.name / "ols.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--regime", default=None)
    args = parser.parse_args()
    dirs = [Path(args.data) / f"regime_{args.regime}"] if args.regime else sorted(Path(args.data).glob("regime_*"))
    for d in dirs:
        fit_ols(d, Path(args.out))


if __name__ == "__main__":
    main()
