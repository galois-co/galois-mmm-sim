"""Shared pieces for the estimators: data loading, design matrices, media transforms, metrics.

Everything here works only on what a practitioner could observe (observed.csv), except
compare_to_truth, which reads truth.json and is used during development and evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

from transforms import fourier_terms, geometric_adstock, hill

WEEKS_PER_YEAR = 52.1775
EVENTS = ["black_friday", "black_friday_next", "christmas", "post_christmas",
          "soldes_early", "soldes_late", "rentree", "august"]


# --------------------------------------------------------------------------- data

def load_observed(regime_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    obs = pd.read_csv(regime_dir / "observed.csv", parse_dates=["week_start"])
    channels = [c.replace("spend_", "") for c in obs.columns if c.startswith("spend_")]
    return obs, channels


def spend_matrix(obs: pd.DataFrame, channels: list[str]) -> np.ndarray:
    return obs[[f"spend_{c}" for c in channels]].to_numpy(dtype=float)


def baseline_design(obs: pd.DataFrame, fourier_order: int = 3) -> tuple[np.ndarray, list[str]]:
    """Controls: intercept, linear trend, annual Fourier terms, event flags, discount depth.

    Continuous columns are standardised; the intercept is not.
    """
    n = len(obs)
    cols, names = [np.ones(n)], ["intercept"]
    cols.append(np.arange(n) / WEEKS_PER_YEAR); names.append("trend")
    F = fourier_terms(n, order=fourier_order)
    for h in range(fourier_order):
        cols += [F[:, 2 * h], F[:, 2 * h + 1]]
        names += [f"sin{h + 1}", f"cos{h + 1}"]
    for ev in EVENTS:
        cols.append(obs[ev].to_numpy(dtype=float)); names.append(ev)
    cols.append(obs["discount_depth"].to_numpy(dtype=float)); names.append("discount_depth")
    X = np.column_stack(cols)
    for j in range(1, X.shape[1]):
        sd = X[:, j].std()
        if sd > 0:
            X[:, j] = (X[:, j] - X[:, j].mean()) / sd
    return X, names


# --------------------------------------------------------------------------- media transforms

def transform_media(spend: np.ndarray, params: list[dict], l_max: int = 13) -> np.ndarray:
    """Adstock then Hill, per channel. params[j] = {decay, k_rel, s}; K = k_rel * mean spend."""
    H = np.zeros_like(spend)
    for j, p in enumerate(params):
        a = geometric_adstock(spend[:, j], p["decay"], l_max=l_max, normalize=True)
        k = p["k_rel"] * spend[:, j].mean()
        H[:, j] = hill(a, k, p["s"])
    return H


def hill_derivative(x, k, s):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = s * x[pos] ** (s - 1) * k ** s / (k ** s + x[pos] ** s) ** 2
    return out


# --------------------------------------------------------------------------- fitting

def fit_bounded_ridge(X_controls: np.ndarray, H: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    """Ridge regression with non-negative media coefficients.

    y is standardised for the fit so that alpha is scale-free; coefficients are returned in
    the units of y. The intercept is not penalised. Solved as a bounded least squares problem
    on the ridge-augmented system.
    """
    y_mean, y_sd = y.mean(), y.std()
    ys = (y - y_mean) / y_sd
    X = np.hstack([X_controls, H])
    p_ctrl, p_media = X_controls.shape[1], H.shape[1]
    pen = np.sqrt(alpha) * np.eye(X.shape[1])
    pen[0, 0] = 0.0                                          # intercept unpenalised
    A = np.vstack([X, pen])
    b = np.concatenate([ys, np.zeros(X.shape[1])])
    lb = np.concatenate([np.full(p_ctrl, -np.inf), np.zeros(p_media)])
    ub = np.full(X.shape[1], np.inf)
    res = lsq_linear(A, b, bounds=(lb, ub), method="bvls")
    coef = res.x * y_sd
    coef[0] += y_mean
    return {"coef_controls": coef[:p_ctrl], "coef_media": coef[p_ctrl:],
            "fitted": X @ coef}


def rolling_origin_folds(n: int, n_folds: int = 3, horizon: int = 13) -> list[tuple[np.ndarray, np.ndarray]]:
    """Train on everything before the fold's window, validate on the next `horizon` weeks."""
    folds = []
    for f in range(n_folds, 0, -1):
        end_train = n - f * horizon
        folds.append((np.arange(end_train), np.arange(end_train, end_train + horizon)))
    return folds


# --------------------------------------------------------------------------- metrics

def nrmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)) / (y.max() - y.min()))


def r2(y, yhat) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def media_summary(beta: np.ndarray, H: np.ndarray, spend: np.ndarray, params: list[dict],
                  channels: list[str]) -> dict:
    """ROI, marginal ROI at mean spend, contribution shares and response curves per channel."""
    contrib = H * beta
    total_media = contrib.sum()
    out = {}
    for j, c in enumerate(channels):
        x = spend[:, j]
        k = params[j]["k_rel"] * x.mean()
        grid = np.linspace(0, 2.5 * x.mean(), 26)
        out[c] = {
            "beta": float(beta[j]),
            "decay": float(params[j]["decay"]), "k_rel": float(params[j]["k_rel"]), "s": float(params[j]["s"]),
            "roi": float(contrib[:, j].sum() / x.sum()) if x.sum() > 0 else 0.0,
            "mroi_at_mean_spend": float(beta[j] * hill_derivative(np.array([x.mean()]), k, params[j]["s"])[0]),
            "contribution_share_of_media": float(contrib[:, j].sum() / total_media) if total_media > 0 else 0.0,
            "response_curve": {"spend": grid.round(2).tolist(),
                               "revenue": (beta[j] * hill(grid, k, params[j]["s"])).round(2).tolist()},
        }
    return out


def decomp_rssd(summary: dict, spend: np.ndarray, channels: list[str]) -> float:
    """Robyn's decomposition distance: how far effect shares sit from spend shares."""
    spend_share = spend.sum(axis=0) / spend.sum()
    effect_share = np.array([summary[c]["contribution_share_of_media"] for c in channels])
    return float(np.sqrt(np.sum((effect_share - spend_share) ** 2)))


def compare_to_truth(summary: dict, regime_dir: Path, channels: list[str]) -> dict:
    """Development-time comparison. Never call this on the sealed reference instance before reveal."""
    truth = json.loads((regime_dir / "truth.json").read_text())
    roi_true = np.array([truth["channels"][c]["roi_realised"] for c in channels])
    roi_est = np.array([summary[c]["roi"] for c in channels])
    share_true = np.array([truth["channels"][c]["contribution_share_of_media"] for c in channels])
    share_est = np.array([summary[c]["contribution_share_of_media"] for c in channels])
    return {
        "roi_true": dict(zip(channels, roi_true.round(3).tolist())),
        "roi_abs_error": dict(zip(channels, np.abs(roi_est - roi_true).round(3).tolist())),
        "roi_mean_abs_error": float(np.mean(np.abs(roi_est - roi_true))),
        "roi_spearman": float(spearmanr(roi_true, roi_est).correlation),
        "share_mean_abs_error": float(np.mean(np.abs(share_est - share_true))),
        "baseline_share_true": truth["baseline_share"],
    }


def print_comparison(name: str, summary: dict, cmp: dict, channels: list[str], fit: dict):
    print(f"\n{name}")
    for k, v in fit.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
    print(f"  {'channel':16s} {'ROI true':>9s} {'ROI est':>9s} {'abs err':>8s} {'mROI est':>9s} {'decay':>6s} {'k_rel':>6s} {'s':>5s}")
    for c in channels:
        s = summary[c]
        print(f"  {c:16s} {cmp['roi_true'][c]:>9.2f} {s['roi']:>9.2f} {cmp['roi_abs_error'][c]:>8.2f} "
              f"{s['mroi_at_mean_spend']:>9.2f} {s['decay']:>6.2f} {s['k_rel']:>6.2f} {s['s']:>5.2f}")
    print(f"  ROI mean abs error {cmp['roi_mean_abs_error']:.3f}   rank correlation {cmp['roi_spearman']:.2f}   "
          f"share mean abs error {cmp['share_mean_abs_error']:.3f}")


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
