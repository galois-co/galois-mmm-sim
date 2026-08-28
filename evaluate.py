"""Evaluate every fitted model against the ground truth, including the cost of acting on it.

Part 1, recovery: one row per (regime, model) with ROI mean absolute error, rank correlation,
contribution-share error, interval coverage where available, and fit statistics.

Part 2, decisions: a constrained budget optimiser is run on each fitted model's response
curves (same total weekly budget, each channel within +/- `bound` of its current level), and
the recommended allocation is scored on the TRUE response. Reported per model:
    predicted uplift   what the model promises versus the status quo
    realised uplift    what the true business would actually deliver
    regret             true optimum minus realised, in EUR per week and per year
Two scenarios are also priced by each model and by the truth: video cut by half, and a 20%
CPM inflation on social (the same money buys 20% less delivery).

Steady state is assumed for decisions: with constant weekly spend, normalised adstock equals
spend, so the weekly response is beta * Hill(spend). Bayesian models use posterior medians.

Usage
    python evaluate.py                 # all regimes with results
    python evaluate.py --bound 0.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from features import compare_to_truth
from transforms import hill

MODEL_ORDER = ["ols", "ridge", "ridge_rssd0.1", "bayes", "bayes_calibrated", "bayes_prior0.5", "bayes_prior2"]
MODEL_LABEL = {"ols": "M0 OLS", "ridge": "M1 ridge", "ridge_rssd0.1": "M1 ridge + RSSD",
               "bayes": "M2 Bayes", "bayes_calibrated": "M3 Bayes + lift",
               "bayes_prior0.5": "M2 priors x0.5", "bayes_prior2": "M2 priors x2"}


# --------------------------------------------------------------------------- response functions

def true_response(truth: dict, channels: list[str]):
    """Weekly revenue from a vector of OBSERVED spend levels, under the true model.

    Observed spend is converted to true spend with the ratio recorded in truth.json (only
    regime D differs, through the display fee markup and measurement error).
    """
    beta = np.array([truth["channels"][c]["beta"] for c in channels])
    k = np.array([truth["channels"][c]["k_abs"] for c in channels])
    s = np.array([truth["channels"][c]["s"] for c in channels])
    ratio = np.array([truth["channels"][c]["mean_weekly_spend_true"] / truth["channels"][c]["mean_weekly_spend_observed"]
                      for c in channels])

    def f(x_obs):
        x = np.asarray(x_obs, dtype=float) * ratio
        return float(np.sum(beta * hill(x, k, s)))
    return f


def model_response(result: dict, channels: list[str], mean_spend: np.ndarray):
    """Weekly revenue from observed spend levels, as the fitted model believes it."""
    ch = result["channels"]
    if result["model"] == "M0_ols":
        beta = np.array([ch[c]["beta"] for c in channels])
        return lambda x: float(np.sum(beta * np.asarray(x, dtype=float)))
    beta = np.array([ch[c]["beta"] for c in channels])
    k = np.array([ch[c]["k_rel"] for c in channels]) * mean_spend
    s = np.array([ch[c]["s"] for c in channels])
    return lambda x: float(np.sum(beta * hill(np.asarray(x, dtype=float), k, s)))


# --------------------------------------------------------------------------- optimiser

def optimise(f, x0: np.ndarray, bound: float) -> np.ndarray:
    """Maximise f(x) subject to sum(x) = sum(x0) and (1-bound) x0 <= x <= (1+bound) x0."""
    total = x0.sum()
    bounds = [((1 - bound) * v, (1 + bound) * v) for v in x0]
    cons = [{"type": "eq", "fun": lambda x: x.sum() - total}]
    best, best_val = x0.copy(), f(x0)
    rng = np.random.default_rng(0)
    starts = [x0] + [np.clip(x0 * rng.uniform(1 - bound, 1 + bound, len(x0)), [b[0] for b in bounds], [b[1] for b in bounds])
                     for _ in range(7)]
    for st in starts:
        st = st * total / st.sum()
        res = minimize(lambda x: -f(x), st, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-10})
        if res.success and f(res.x) > best_val:
            best, best_val = res.x, f(res.x)
    return best


# --------------------------------------------------------------------------- evaluation

def load_results(results_dir: Path) -> dict:
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.stem in ("decisions", "summary"):
            continue
        out[p.stem] = json.loads(p.read_text())
    return out


def evaluate_regime(regime_dir: Path, results_dir: Path, bound: float) -> tuple[list[dict], dict]:
    truth = json.loads((regime_dir / "truth.json").read_text())
    channels = list(truth["channels"].keys())
    results = load_results(results_dir)
    if not results:
        return [], {}
    mean_spend = np.array([truth["channels"][c]["mean_weekly_spend_observed"] for c in channels])
    f_true = true_response(truth, channels)

    # Truth-based references.
    x_sq = mean_spend.copy()
    rev_sq = f_true(x_sq)
    x_opt = optimise(f_true, x_sq, bound)
    rev_opt = f_true(x_opt)
    scen_true = {"video_cut_50": f_true(x_sq * np.where(np.array(channels) == "video", 0.5, 1.0)) - rev_sq,
                 "social_cpm_plus_20": f_true(x_sq * np.where(np.array(channels) == "social", 1 / 1.2, 1.0)) - rev_sq}

    rows, decisions = [], {"regime": truth["regime"], "bound": bound, "status_quo_weekly_media_revenue": rev_sq,
                           "true_optimal_weekly_media_revenue": rev_opt,
                           "true_optimal_allocation": dict(zip(channels, x_opt.round(0).tolist())),
                           "scenarios_true": scen_true, "models": {}}
    for name, res in results.items():
        cmp = res.get("comparison_to_truth")
        if cmp is None:
            # Blind fit evaluated after the reveal: recompute the comparison from the stored
            # estimates, and the interval coverage where the model provides intervals.
            cmp = compare_to_truth(res["channels"], regime_dir, channels)
            if cmp is not None and all("roi_hdi5" in res["channels"][c] for c in channels):
                inside = {c: res["channels"][c]["roi_hdi5"] <= cmp["roi_true"][c] <= res["channels"][c]["roi_hdi95"]
                          for c in channels}
                cmp["coverage_90"] = float(np.mean(list(inside.values())))
        if cmp is None:
            continue                                        # truth still sealed: nothing to evaluate yet
        rows.append({"regime": truth["regime"], "model": name, "label": MODEL_LABEL.get(name, name),
                     "roi_mae": cmp["roi_mean_abs_error"], "roi_rank_corr": cmp["roi_spearman"],
                     "share_mae": cmp["share_mean_abs_error"], "coverage_90": cmp.get("coverage_90"),
                     "r2_in_sample": res["fit"].get("r2_in_sample"),
                     "nrmse_cv": res["fit"].get("nrmse_cv_mean"),
                     "negative_coefs": res["fit"].get("negative_media_coefficients")})

        f_model = model_response(res, channels, mean_spend)
        x_rec = optimise(f_model, x_sq, bound)
        predicted_uplift = f_model(x_rec) - f_model(x_sq)
        realised_uplift = f_true(x_rec) - rev_sq
        regret = rev_opt - f_true(x_rec)
        scen_model = {"video_cut_50": f_model(x_sq * np.where(np.array(channels) == "video", 0.5, 1.0)) - f_model(x_sq),
                      "social_cpm_plus_20": f_model(x_sq * np.where(np.array(channels) == "social", 1 / 1.2, 1.0)) - f_model(x_sq)}
        decisions["models"][name] = {
            "label": MODEL_LABEL.get(name, name),
            "recommended_allocation": dict(zip(channels, x_rec.round(0).tolist())),
            "allocation_change_pct": dict(zip(channels, ((x_rec / x_sq - 1) * 100).round(1).tolist())),
            "predicted_uplift_weekly": predicted_uplift, "realised_uplift_weekly": realised_uplift,
            "regret_weekly": regret, "regret_annual": regret * 52.1775,
            "regret_share_of_available_gain": regret / (rev_opt - rev_sq) if rev_opt > rev_sq else None,
            "scenarios_model": scen_model,
        }
        rows[-1].update({"predicted_uplift_weekly": predicted_uplift, "realised_uplift_weekly": realised_uplift,
                         "regret_weekly": regret})
    return rows, decisions


def print_tables(df: pd.DataFrame, decisions: dict):
    pd.set_option("display.width", 200)
    print("\nRECOVERY  (lower error, higher rank correlation and coverage near 0.9 are better)")
    cols = ["regime", "label", "roi_mae", "roi_rank_corr", "share_mae", "coverage_90", "r2_in_sample", "nrmse_cv"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep=""))
    print("\nDECISIONS  (EUR per week; regret = true optimum minus what the model's allocation really delivers)")
    for reg, d in decisions.items():
        gain = d["true_optimal_weekly_media_revenue"] - d["status_quo_weekly_media_revenue"]
        print(f"\n  regime {reg}: status quo media revenue {d['status_quo_weekly_media_revenue']:,.0f}, "
              f"true optimum {d['true_optimal_weekly_media_revenue']:,.0f} (+{gain:,.0f}), "
              f"bound +/-{d['bound']:.0%}")
        print(f"  {'model':20s} {'predicted':>11s} {'realised':>10s} {'regret/wk':>10s} {'regret/yr':>12s} {'of gain':>8s}")
        for name, m in d["models"].items():
            share = f"{m['regret_share_of_available_gain']:.0%}" if m["regret_share_of_available_gain"] is not None else ""
            print(f"  {m['label']:20s} {m['predicted_uplift_weekly']:>+11,.0f} {m['realised_uplift_weekly']:>+10,.0f} "
                  f"{m['regret_weekly']:>10,.0f} {m['regret_annual']:>12,.0f} {share:>8s}")
        st = d["scenarios_true"]
        print(f"  scenarios, true effect: video -50% {st['video_cut_50']:+,.0f}/wk, social CPM +20% {st['social_cpm_plus_20']:+,.0f}/wk")
        for name, m in d["models"].items():
            sm = m["scenarios_model"]
            print(f"    {m['label']:20s} predicts video -50% {sm['video_cut_50']:+,.0f}, social CPM +20% {sm['social_cpm_plus_20']:+,.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--results", default="results")
    parser.add_argument("--bound", type=float, default=0.3, help="max relative change per channel in the optimiser")
    args = parser.parse_args()

    all_rows, all_decisions = [], {}
    for regime_dir in sorted(Path(args.data).glob("regime_*")):
        results_dir = Path(args.results) / regime_dir.name
        if not results_dir.exists():
            continue
        rows, decisions = evaluate_regime(regime_dir, results_dir, args.bound)
        if rows:
            all_rows += rows
            all_decisions[regime_dir.name[-1]] = decisions
            (results_dir / "decisions.json").write_text(json.dumps(decisions, indent=2))
    df = pd.DataFrame(all_rows)
    df["order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df = df.sort_values(["regime", "order"]).drop(columns="order")
    df.to_csv(Path(args.results) / "summary.csv", index=False)
    print_tables(df, all_decisions)


if __name__ == "__main__":
    main()
