"""Export frozen JSON snapshots for the research note on galois-co.com.

Reads the committed results of the study and writes one JSON per chart into export/<slug>/,
ready to be copied into the site repository at public/data/snapshots/<slug>/. Everything is
rounded and trimmed: these files are frozen with the note and never regenerated.

Files produced
    spend-revenue.json       regime C observed series (stacked spend + revenue timeline)
    roi-recovery.json        sealed reference instance: true ROI vs the four blind estimates
    response-curves.json     sealed reference instance: true curve vs M1 and M2 (median + 90% band)
    collinearity-error.json  replication study: ROI error vs realised spend correlation, by model
    decision-regret.json     sealed reference instance: promised vs delivered uplift and regret
    sandbox.json             true response parameters of the reference, for the interactive island
    meta.json                dates, seeds, commitment hashes, environment versions

Usage
    python export.py                      # writes export/which-euro-moves-revenue/
    python export.py --slug my-slug --out export
"""

import argparse
import json
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from transforms import hill

CHANNELS = ["search_nonbrand", "search_brand", "social", "display", "video", "affiliate"]
MODEL_LABEL = {"ols": "M0 OLS", "ridge": "M1 ridge", "bayes": "M2 Bayes", "bayes_calibrated": "M3 Bayes + lift"}


def sanitize(obj):
    """NaN and infinities become null: strict JSON only, whatever pandas produced."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def rnd(x, n=2):
    return None if x is None else round(float(x), n)


# --------------------------------------------------------------------------- individual exports

def export_spend_revenue(data_dir: Path) -> dict | None:
    obs_path = data_dir / "regime_C" / "observed.csv"
    if not obs_path.exists():
        return None
    obs = pd.read_csv(obs_path)
    return {
        "weeks": obs["week_start"].tolist(),
        "revenue": [round(v) for v in obs["revenue"]],
        "spend": {c: [round(v) for v in obs[f"spend_{c}"]] for c in CHANNELS},
        "events": {ev: obs[ev].astype(int).tolist() for ev in ["black_friday", "soldes_early", "christmas"]},
        "note": "regime C: correlated and endogenous spend; the development instance, not the sealed one",
    }


def export_roi_recovery(reference_dir: Path, results_ref: Path) -> dict | None:
    truth = load_json(reference_dir / "regime_R" / "truth.json")
    if truth is None:
        return None
    out = {"channels": CHANNELS,
           "roi_true": {c: rnd(truth["channels"][c]["roi_realised"]) for c in CHANNELS},
           "models": {}}
    for name in ["ols", "ridge", "bayes", "bayes_calibrated"]:
        res = load_json(results_ref / "regime_R" / f"{name}.json")
        if res is None:
            continue
        entry = {"label": MODEL_LABEL[name],
                 "roi": {c: rnd(res["channels"][c]["roi"]) for c in CHANNELS}}
        if all("roi_hdi5" in res["channels"][c] for c in CHANNELS):
            entry["roi_hdi5"] = {c: rnd(res["channels"][c]["roi_hdi5"]) for c in CHANNELS}
            entry["roi_hdi95"] = {c: rnd(res["channels"][c]["roi_hdi95"]) for c in CHANNELS}
        out["models"][name] = entry
    return out


def export_response_curves(reference_dir: Path, results_ref: Path) -> dict | None:
    truth = load_json(reference_dir / "regime_R" / "truth.json")
    if truth is None:
        return None
    obs = pd.read_csv(reference_dir / "regime_R" / "observed.csv")
    ridge = load_json(results_ref / "regime_R" / "ridge.json")
    bayes = load_json(results_ref / "regime_R" / "bayes.json")
    out = {"channels": {}}
    for c in CHANNELS:
        t = truth["channels"][c]
        x_obs = obs[f"spend_{c}"].to_numpy(dtype=float)
        ratio = t["mean_weekly_spend_true"] / t["mean_weekly_spend_observed"]
        grid = np.linspace(0, 2.5 * x_obs.mean(), 26)
        true_curve = t["beta"] * hill(grid * ratio, t["k_abs"], t["s"])
        entry = {
            "spend_grid": [round(v) for v in grid],
            "true": [round(v) for v in true_curve],
            "mean_weekly_spend": round(x_obs.mean()),
            "observed_spend_min": round(float(np.percentile(x_obs, 5))),
            "observed_spend_max": round(float(np.percentile(x_obs, 95))),
        }
        if ridge is not None:
            rc = ridge["channels"][c]["response_curve"]
            entry["ridge"] = [round(v) for v in np.interp(grid, rc["spend"], rc["revenue"])]
        if bayes is not None:
            bc = bayes["channels"][c]["response_curve"]
            for key, name in [("revenue_median", "bayes_median"), ("revenue_hdi5", "bayes_hdi5"),
                              ("revenue_hdi95", "bayes_hdi95")]:
                entry[name] = [round(v) for v in np.interp(grid, bc["spend"], bc[key])]
        out["channels"][c] = entry
    return out


def export_collinearity(results_dir: Path) -> dict | None:
    path = results_dir / "replications.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    out = {"series": [], "n_seeds": int(df["seed"].nunique()),
           "note": "median over seeds; whiskers are the interquartile range"}
    for (sweep, w), grp in df.groupby(["sweep", "budget_cycle_weight"]):
        for model, g in grp.groupby("model"):
            out["series"].append({
                "sweep": sweep, "cycle_weight": float(w), "model": model,
                "label": MODEL_LABEL.get(model, model),
                "spend_corr_median": rnd(g["spend_corr_mean"].median(), 3),
                "roi_mae_median": rnd(g["roi_mae"].median(), 3),
                "roi_mae_q25": rnd(g["roi_mae"].quantile(0.25), 3),
                "roi_mae_q75": rnd(g["roi_mae"].quantile(0.75), 3),
                "rank_corr_median": rnd(g["roi_rank_corr"].median(), 3),
                "regret_share_median": rnd(g["regret_share_of_gain"].median(), 3),
            })
    b_path = results_dir / "replications_bayes.csv"
    if b_path.exists():
        b = pd.read_csv(b_path)
        for (w, model), g in b.groupby(["budget_cycle_weight", "model"]):
            row = {"sweep": "endogeneity_bayes_run", "cycle_weight": float(w), "model": model,
                   "label": MODEL_LABEL.get(model, model),
                   "spend_corr_median": rnd(g["spend_corr_mean"].median(), 3),
                   "roi_mae_median": rnd(g["roi_mae"].median(), 3),
                   "roi_mae_q25": rnd(g["roi_mae"].quantile(0.25), 3),
                   "roi_mae_q75": rnd(g["roi_mae"].quantile(0.75), 3),
                   "rank_corr_median": rnd(g["roi_rank_corr"].median(), 3),
                   "regret_share_median": rnd(g["regret_share_of_gain"].median(), 3)}
            if "coverage_90" in g:
                row["coverage_90_mean"] = rnd(g["coverage_90"].mean(), 3)
            out["series"].append(row)
    return out


def export_decision_regret(results_ref: Path) -> dict | None:
    d = load_json(results_ref / "regime_R" / "decisions.json")
    if d is None:
        return None
    out = {"status_quo_weekly": round(d["status_quo_weekly_media_revenue"]),
           "true_optimal_weekly": round(d["true_optimal_weekly_media_revenue"]),
           "bound": d["bound"], "models": {}, "scenarios_true": {k: round(v) for k, v in d["scenarios_true"].items()}}
    for name, m in d["models"].items():
        out["models"][name] = {
            "label": m["label"],
            "predicted_uplift_weekly": round(m["predicted_uplift_weekly"]),
            "realised_uplift_weekly": round(m["realised_uplift_weekly"]),
            "regret_weekly": round(m["regret_weekly"]),
            "regret_annual": round(m["regret_annual"]),
            "scenarios": {k: round(v) for k, v in m["scenarios_model"].items()},
        }
    return out


def export_sandbox(reference_dir: Path) -> dict | None:
    truth = load_json(reference_dir / "regime_R" / "truth.json")
    if truth is None:
        return None
    out = {"channels": {}, "note": "true steady-state response of the sealed reference instance, "
                                    "published at the reveal; spend in EUR per week (observed basis)"}
    for c in CHANNELS:
        t = truth["channels"][c]
        out["channels"][c] = {
            "beta": rnd(t["beta"]), "k_abs": rnd(t["k_abs"]), "s": rnd(t["s"], 3),
            "spend_ratio_true_over_observed": rnd(t["mean_weekly_spend_true"] / t["mean_weekly_spend_observed"], 4),
            "mean_weekly_spend": round(t["mean_weekly_spend_observed"]),
            "roi_true": rnd(t["roi_realised"]),
        }
    return out


def export_meta(reference_dir: Path, data_dir: Path) -> dict:
    commitment = (reference_dir / "COMMITMENT.txt")
    dev_truth = load_json(data_dir / "regime_C" / "truth.json")
    versions = {}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "pymc", "pytensor", "arviz"]:
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass
    return {"exported": date.today().isoformat(),
            "python": platform.python_version(),
            "versions": versions,
            "dev_seed": dev_truth["seed"] if dev_truth else None,
            "commitment": commitment.read_text() if commitment.exists() else None,
            "repository": "https://github.com/galois-co/galois-mmm-sim"}


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", default="which-euro-moves-revenue")
    parser.add_argument("--out", default="export")
    parser.add_argument("--data", default="data")
    parser.add_argument("--results", default="results")
    parser.add_argument("--reference", default="reference")
    parser.add_argument("--results-reference", default="results_reference")
    args = parser.parse_args()

    out = Path(args.out) / args.slug
    out.mkdir(parents=True, exist_ok=True)
    exports = {
        "spend-revenue.json": export_spend_revenue(Path(args.data)),
        "roi-recovery.json": export_roi_recovery(Path(args.reference), Path(args.results_reference)),
        "response-curves.json": export_response_curves(Path(args.reference), Path(args.results_reference)),
        "collinearity-error.json": export_collinearity(Path(args.results)),
        "decision-regret.json": export_decision_regret(Path(args.results_reference)),
        "sandbox.json": export_sandbox(Path(args.reference)),
        "meta.json": export_meta(Path(args.reference), Path(args.data)),
    }
    for name, payload in exports.items():
        if payload is None:
            print(f"  SKIPPED {name}: source files not found")
            continue
        (out / name).write_text(json.dumps(sanitize(payload), indent=1, allow_nan=False))
        print(f"  wrote {out / name}  ({(out / name).stat().st_size / 1024:.0f} KB)")
    print(f"\nCopy into the site repository with:\n  cp -r {out} ~/code/galois-co/galois-co/public/data/snapshots/")


if __name__ == "__main__":
    main()
