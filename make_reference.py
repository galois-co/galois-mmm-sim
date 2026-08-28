"""Create the sealed reference instance: fit it blind, reveal the truth afterwards.

Protocol
1. This script draws a master seed from the OS entropy pool (not reproducible from code) and
   RANDOMISES the true parameters: channel ROIs, adstock decays, half-saturations, Hill slopes,
   budget shares, sentiment volatility, and whether the misspecification block is active.
   config.json therefore tells you nothing about the truth of this instance.
2. The observable files (observed.csv, lift_test.json) are written to reference/.
   Everything else (truth.csv, truth.json, the decomposition plot) goes to reference/sealed/,
   which is kept out of git until the reveal.
3. reference/COMMITMENT.txt records the SHA-256 of both truth files. Commit it BEFORE any model
   is fitted: the git history then proves the truth was fixed first and never edited.
4. After all models are fitted and their results committed, reveal with
       git add -f reference/sealed && git commit
   and anyone can check the hashes:  python make_reference.py --verify

Usage
    python make_reference.py            # create (refuses to overwrite an existing reference)
    python make_reference.py --verify   # check sealed files against COMMITMENT.txt
"""

import argparse
import hashlib
import json
import secrets
from datetime import date
from pathlib import Path

import numpy as np

from simulate import simulate_regime

CLASS_DECAY_RANGE = {"search": (0.05, 0.35), "digital": (0.20, 0.60), "video": (0.50, 0.85)}
CHANNEL_CLASS = {"search_nonbrand": "search", "search_brand": "search", "social": "digital",
                 "display": "digital", "affiliate": "digital", "video": "video"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def randomise_truth(cfg: dict, rng: np.random.Generator) -> dict:
    cfg = json.loads(json.dumps(cfg))                       # deep copy
    channels = list(cfg["media"]["channels"].keys())

    # One channel is forced below break-even; the others draw around a centre of 2.
    weak = channels[rng.integers(len(channels))]
    shares = rng.dirichlet(np.array([cfg["media"]["channels"][c]["share"] for c in channels]) * 40)
    for j, c in enumerate(channels):
        spec = cfg["media"]["channels"][c]
        lo, hi = CLASS_DECAY_RANGE[CHANNEL_CLASS[c]]
        spec["decay"] = float(rng.uniform(lo, hi))
        spec["k_rel"] = float(np.exp(rng.normal(0.0, 0.4)))
        spec["s"] = float(np.clip(np.exp(rng.normal(0.1, 0.3)), 0.6, 2.8))
        spec["roi_target"] = float(rng.uniform(0.5, 0.9)) if c == weak else \
            float(np.clip(np.exp(rng.normal(np.log(2.0), 0.35)), 1.0, 4.0))
        spec["share"] = float(np.clip(shares[j], 0.04, 0.45))
    cfg["business"]["sentiment"]["sd"] = float(rng.uniform(0.03, 0.07))
    cfg["business"]["noise_sd_share"] = float(rng.uniform(0.03, 0.06))

    regime = dict(cfg["regimes"]["C"])
    regime["misspecification"] = bool(rng.random() < 0.5)
    regime["label"] = "sealed reference instance"
    return cfg, regime


def create(args):
    ref = Path(args.out)
    if (ref / "COMMITMENT.txt").exists():
        raise SystemExit(f"{ref}/COMMITMENT.txt already exists; the reference is created once and never regenerated.")
    master_seed = secrets.randbits(63)
    rng = np.random.default_rng(master_seed)
    cfg = json.loads(Path(args.config).read_text())
    cfg, regime = randomise_truth(cfg, rng)
    seed = int(rng.integers(2**31))

    simulate_regime(cfg, "R", seed, ref, regime_override=regime, plot=True)
    src = ref / "regime_R"
    sealed = ref / "sealed"
    sealed.mkdir(parents=True, exist_ok=True)
    for name in ["truth.csv", "truth.json", "decomposition.png"]:
        (src / name).replace(sealed / name)
    # The full generating configuration and the master seed go into the sealed folder too.
    (sealed / "generating_config.json").write_text(json.dumps(
        {"master_seed": master_seed, "simulation_seed": seed, "config": cfg, "regime": regime}, indent=2))

    commitment = "\n".join([
        "Sealed reference instance, Galois & Co. MMM simulation study",
        f"created {date.today().isoformat()}",
        "",
        "The files below were generated once, with randomised true parameters and a master seed",
        "drawn from the OS entropy pool, then sealed. All models are fitted on reference/observed.csv",
        "(and reference/lift_test.json for calibrated models) with no access to the sealed folder.",
        "The git history shows this commitment before any result on the reference instance.",
        "",
        f"sha256(truth.json)  {sha256(sealed / 'truth.json')}",
        f"sha256(truth.csv)   {sha256(sealed / 'truth.csv')}",
        f"sha256(observed.csv) {sha256(src / 'observed.csv')}",
        f"sha256(lift_test.json) {sha256(src / 'lift_test.json')}",
        "",
        "Reveal: git add -f reference/sealed && commit; verify: python make_reference.py --verify",
    ]) + "\n"
    (ref / "COMMITMENT.txt").write_text(commitment)

    print(commitment)
    print("Created:")
    print(f"  public  {src}/observed.csv, {src}/lift_test.json, {ref}/COMMITMENT.txt")
    print(f"  sealed  {sealed}/  (keep out of git until the reveal; keep a private backup)")
    print("\nNext: add 'reference/sealed/' to .gitignore, commit the public files, then fit blind, e.g.")
    print("  python fit_ols.py --data reference --regime R --out results_reference")


def verify(args):
    ref = Path(args.out)
    text = (ref / "COMMITMENT.txt").read_text()
    ok = True
    for name, path in [("truth.json", ref / "sealed/truth.json"), ("truth.csv", ref / "sealed/truth.csv"),
                       ("observed.csv", ref / "regime_R/observed.csv"), ("lift_test.json", ref / "regime_R/lift_test.json")]:
        h = sha256(path)
        found = h in text
        ok &= found
        print(f"  {name:16s} {h}  {'MATCHES' if found else 'DOES NOT MATCH'}")
    print("commitment verified" if ok else "COMMITMENT BROKEN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", default="reference")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    (verify if args.verify else create)(args)


if __name__ == "__main__":
    main()
