"""M1: ridge regression on transformed media, hyperparameters found by search.

This is the Robyn recipe in miniature: fix (decay, K, S) per channel, transform spend,
fit a ridge with non-negative media coefficients, score the candidate, search. Robyn uses
Nevergrad's evolutionary optimiser and a multi-objective Pareto front; here the search is a
two-stage evolutionary random search and the selection criterion is rolling-origin NRMSE only.
DECOMP.RSSD is computed and reported but deliberately not used for selection, so that its
effect can be discussed rather than silently baked in.

Also recorded: the ROI dispersion across the best `top_k` candidates, because models with
nearly identical out-of-sample error can imply very different channel ROIs.

Usage
    python fit_ridge.py --regime C
    python fit_ridge.py --n-random 1500 --n-mutate 20 --n-rounds 4   # smaller, faster search
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from features import (baseline_design, compare_to_truth, decomp_rssd, fit_bounded_ridge, load_observed,
                      media_summary, nrmse, print_comparison, r2, rolling_origin_folds, save_json,
                      spend_matrix, transform_media)

BOUNDS = {"decay": (0.0, 0.9), "log_k_rel": (np.log(0.3), np.log(3.0)),
          "log_s": (np.log(0.5), np.log(3.0)), "log_alpha": (np.log(1e-3), np.log(10.0))}


def random_candidate(rng: np.random.Generator, n_channels: int) -> np.ndarray:
    """Vector layout: [decay_1..C, log_k_rel_1..C, log_s_1..C, log_alpha]."""
    d = rng.uniform(*BOUNDS["decay"], n_channels)
    k = rng.uniform(*BOUNDS["log_k_rel"], n_channels)
    s = rng.uniform(*BOUNDS["log_s"], n_channels)
    a = rng.uniform(*BOUNDS["log_alpha"], 1)
    return np.concatenate([d, k, s, a])


def mutate(v: np.ndarray, rng: np.random.Generator, n_channels: int, scale: float) -> np.ndarray:
    out = v.copy()
    C = n_channels
    out[:C] = np.clip(out[:C] + rng.normal(0, 0.08 * scale, C), *BOUNDS["decay"])
    out[C:2 * C] = np.clip(out[C:2 * C] + rng.normal(0, 0.25 * scale, C), *BOUNDS["log_k_rel"])
    out[2 * C:3 * C] = np.clip(out[2 * C:3 * C] + rng.normal(0, 0.25 * scale, C), *BOUNDS["log_s"])
    out[3 * C] = np.clip(out[3 * C] + rng.normal(0, 0.5 * scale), *BOUNDS["log_alpha"])
    return out


def unpack(v: np.ndarray, n_channels: int) -> tuple[list[dict], float]:
    C = n_channels
    params = [{"decay": float(v[j]), "k_rel": float(np.exp(v[C + j])), "s": float(np.exp(v[2 * C + j]))}
              for j in range(C)]
    return params, float(np.exp(v[3 * C]))


def score(v, spend, X_ctrl, y, folds, l_max, rssd_weight=0.0) -> float:
    """Rolling-origin NRMSE, optionally plus rssd_weight * DECOMP.RSSD (Robyn's business prior).

    RSSD is computed on the training fold: the distance between each channel's share of the
    fitted media effect and its share of spend. With rssd_weight > 0 the search is pulled towards
    decompositions that mirror the budget split, whatever the data say.
    """
    params, alpha = unpack(v, spend.shape[1])
    H = transform_media(spend, params, l_max)
    errs, rssds = [], []
    for tr, va in folds:
        fit = fit_bounded_ridge(X_ctrl[tr], H[tr], y[tr], alpha)
        coef = np.concatenate([fit["coef_controls"], fit["coef_media"]])
        errs.append(nrmse(y[va], np.hstack([X_ctrl[va], H[va]]) @ coef))
        if rssd_weight > 0:
            effect = (H[tr] * fit["coef_media"]).sum(axis=0)
            effect_share = effect / effect.sum() if effect.sum() > 0 else np.full(len(effect), 1 / len(effect))
            spend_share = spend[tr].sum(axis=0) / spend[tr].sum()
            rssds.append(np.sqrt(np.sum((effect_share - spend_share) ** 2)))
    return float(np.mean(errs) + (rssd_weight * np.mean(rssds) if rssd_weight > 0 else 0.0))


def search(spend, X_ctrl, y, folds, l_max, rng, n_random, n_mutate, n_rounds, top_k, rssd_weight=0.0, verbose=True):
    """Stage 1: random candidates. Stage 2: mutate the best ones, shrinking the step each round."""
    C = spend.shape[1]
    pool = [random_candidate(rng, C) for _ in range(n_random)]
    scores = [score(v, spend, X_ctrl, y, folds, l_max, rssd_weight) for v in pool]
    if verbose:
        print(f"  stage 1: {n_random} random candidates, best objective {min(scores):.4f}")
    for r in range(n_rounds):
        order = np.argsort(scores)[:top_k]
        elite = [pool[i] for i in order]
        scale = 1.0 / (r + 1)
        children = [mutate(e, rng, C, scale) for e in elite for _ in range(n_mutate)]
        child_scores = [score(v, spend, X_ctrl, y, folds, l_max, rssd_weight) for v in children]
        pool += children
        scores += child_scores
        if verbose:
            print(f"  round {r + 1}: {len(children)} mutants, best objective {min(scores):.4f}")
    order = np.argsort(scores)
    return [pool[i] for i in order[:top_k]], [scores[i] for i in order[:top_k]]


def fit_ridge(regime_dir: Path, out_dir: Path, args) -> dict:
    t0 = time.time()
    obs, channels = load_observed(regime_dir)
    y = obs["revenue"].to_numpy(dtype=float)
    spend = spend_matrix(obs, channels)
    X_ctrl, ctrl_names = baseline_design(obs)
    folds = rolling_origin_folds(len(y))
    rng = np.random.default_rng(args.seed)

    tag = f" with DECOMP.RSSD weight {args.rssd_weight}" if args.rssd_weight > 0 else ""
    print(f"\nM1 ridge search{tag}, regime {regime_dir.name[-1]}")
    best, best_scores = search(spend, X_ctrl, y, folds, args.l_max, rng,
                               args.n_random, args.n_mutate, args.n_rounds, args.top_k, args.rssd_weight)

    # Full-sample refit of the winner and of the runners-up (to measure dispersion).
    summaries = []
    for v in best:
        params, alpha = unpack(v, len(channels))
        H = transform_media(spend, params, args.l_max)
        fit = fit_bounded_ridge(X_ctrl, H, y, alpha)
        summaries.append((params, alpha, fit, media_summary(fit["coef_media"], H, spend, params, channels)))
    params, alpha, fit, summary = summaries[0]
    fitted = fit["fitted"]

    roi_dispersion = {c: {"min": float(min(s[3][c]["roi"] for s in summaries)),
                          "max": float(max(s[3][c]["roi"] for s in summaries)),
                          "values": [round(s[3][c]["roi"], 3) for s in summaries]} for c in channels}

    H_best = transform_media(spend, params, args.l_max)
    cv_errs = []
    for tr, va in folds:
        f = fit_bounded_ridge(X_ctrl[tr], H_best[tr], y[tr], alpha)
        coef = np.concatenate([f["coef_controls"], f["coef_media"]])
        cv_errs.append(nrmse(y[va], np.hstack([X_ctrl[va], H_best[va]]) @ coef))
    nrmse_cv = float(np.mean(cv_errs))
    fit_metrics = {"r2_in_sample": r2(y, fitted), "objective_best": float(best_scores[0]),
                   "objective_top_k_range": f"{best_scores[0]:.4f} to {best_scores[-1]:.4f}",
                   "nrmse_cv_mean": nrmse_cv, "rssd_weight": args.rssd_weight,
                   "alpha": alpha, "decomp_rssd": decomp_rssd(summary, spend, channels),
                   "search_seconds": round(time.time() - t0, 1)}
    cmp = compare_to_truth(summary, regime_dir, channels)
    print_comparison(f"M1 ridge (best of {len(best)} finalists), regime {regime_dir.name[-1]}",
                     summary, cmp, channels, fit_metrics)
    print("  ROI range across finalists with near-identical NRMSE:")
    for c in channels:
        print(f"    {c:16s} {roi_dispersion[c]['min']:.2f} to {roi_dispersion[c]['max']:.2f}")

    result = {"model": "M1_ridge", "regime": regime_dir.name[-1], "fit": fit_metrics,
              "search": {"n_random": args.n_random, "n_mutate": args.n_mutate, "n_rounds": args.n_rounds,
                         "top_k": args.top_k, "seed": args.seed, "selection": "rolling-origin NRMSE, 3 folds of 13 weeks"},
              "controls": dict(zip(ctrl_names, fit["coef_controls"].round(2).tolist())),
              "channels": summary, "finalist_roi_dispersion": roi_dispersion,
              "comparison_to_truth": cmp}
    suffix = f"_rssd{args.rssd_weight:g}" if args.rssd_weight > 0 else ""
    save_json(result, out_dir / regime_dir.name / f"ridge{suffix}.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--regime", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--l-max", type=int, default=13)
    parser.add_argument("--n-random", type=int, default=3000)
    parser.add_argument("--n-mutate", type=int, default=30)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rssd-weight", type=float, default=0.0,
                        help="weight of DECOMP.RSSD in the objective; 0 = pure out-of-sample NRMSE")
    args = parser.parse_args()
    dirs = [Path(args.data) / f"regime_{args.regime}"] if args.regime else sorted(Path(args.data).glob("regime_*"))
    for d in dirs:
        fit_ridge(d, Path(args.out), args)


if __name__ == "__main__":
    main()
