# galois-mmm-sim

A simulation study of marketing mix models: four estimators of increasing sophistication are
tested against a synthetic business whose ground truth is known, sealed, and revealed only
after all models were fitted blind.

Companion research note: *Which euro moves revenue?* (galois-co.com/research/which-euro-moves-revenue).
Every figure and number in the note is produced by this repository.

## Three results

- On the sealed reference instance, a Robyn-style ridge with a 0.93 in-sample R² and clean
  rolling-origin validation estimated the brand-search ROI at **15.25**. The truth, revealed
  afterwards, was **1.31**. Its twenty near-identical finalists agreed with each other.
- A hand-written Bayesian MMM produced 90% intervals that covered the truth **93%** of the
  time without endogenous spend and **83%** with it, losing almost exactly the confounded
  channel. Calibration on a single simulated go-dark experiment recovered the tested channel
  to within 0.05 of its true ROI while leaving the confounded one untouched.
- Acting on the fitted models: with the budget reallocated under a ±30% constraint, the two
  simplest models promised weekly gains of 18k to 36k EUR and delivered losses; the Bayesian
  allocations delivered about a third of the achievable gain.

## The sealed-truth protocol

`make_reference.py` draws a master seed from the OS entropy pool and randomises every true
parameter (channel ROIs, adstock retentions, saturation points, budget shares, and, on a coin
flip, whether an additional misspecification layer is active). The observable data are
published in `reference/regime_R/`; the truth is written to `reference/sealed/` and its
SHA-256 recorded in `reference/COMMITMENT.txt`.

The git history is the proof of order:

1. `Sealed reference instance: randomised truth, SHA-256 commitment before any model is fitted`
2. `Blind fits on the sealed reference instance`
3. `Reveal: sealed truth published, hashes match the pre-registered commitment`
4. `Reveal evaluation: blind predictions scored against the revealed truth`

Verify at any time with `python make_reference.py --verify`.

## What is in here

| File | Role |
|---|---|
| `transforms.py` | Geometric adstock, Hill saturation, Fourier basis, with self-checks |
| `config.json` | The synthetic DTC fashion brand, six channels, four difficulty regimes |
| `simulate.py` | The generator: baseline with unobserved sentiment, endogenous spend mechanisms, go-dark lift test, ground truth |
| `fit_ols.py` | M0: naive OLS on raw spend |
| `fit_ridge.py` | M1: ridge on transformed media, evolutionary hyperparameter search, optional DECOMP.RSSD objective |
| `fit_bayes.py` | M2/M3: Bayesian MMM in PyMC, documented priors, lift-test calibration, prior-scale sensitivity |
| `evaluate.py` | Recovery metrics, constrained budget optimiser, decision regret against the true response |
| `replicate.py` | Replication study: collinearity and endogeneity sweeps across seeds |
| `make_reference.py` | Sealed reference instance: creation, verification |
| `export.py` | Frozen JSON snapshots for the research note |

Regimes: **A** independent spend, **B** correlated spend through a common budget cycle,
**C** correlated and endogenous spend (quarterly budget feedback, anticipatory spending ahead
of the promotional calendar, scheduled video flights, algorithmic performance chasing, and
brand search riding an unobserved sentiment process), **D** = C plus a delayed-peak video
adstock outside the estimators' model class, unobserved competitor promotions, spend
measurement error, and agency fees hidden in observed display spend.

## Reproduce it

macOS and Linux, Python 3.12. On macOS every new terminal must activate the venv first.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --only-binary=:all: "numba==0.62.1"   # Intel macOS: newer numba has no wheel
pip install --only-binary=:all: numpy pandas scipy scikit-learn matplotlib "pymc<6" "arviz<1"
python transforms.py                              # self-checks

python simulate.py                                # four regimes, ground truth, plots
python fit_ols.py && python fit_ridge.py          # seconds to ~15 s per regime
python fit_bayes.py --regime C                    # ~4-5 min per fit on a 2017 laptop
python fit_bayes.py --regime C --calibrate
python evaluate.py                                # recovery table + decision regret
python replicate.py --n-seeds 40                  # ~45 min; add --bayes overnight
```

Exact versions are pinned in `requirements.txt` (PyMC 5.28, pytensor 2.38, numba 0.62 for
Intel macOS wheels). All sampling code is guarded by `if __name__ == "__main__"` because
macOS spawns subprocesses by re-importing the main module. Regimes C and D contain feedback
loops, so bit-identical reproduction across machines is not guaranteed for them; the sealed
reference instance is therefore generated once and frozen, never regenerated, which is also
why `reference/` is committed while `data/` and `results/` are not.

## Honest limitations

The simulator and the estimators share an author; the sealed randomised draw mitigates but
does not eliminate the risk of building a world the models are good at. The general claims
rest on replication medians, not on any single instance. There is no geographic structure,
no competition, no channel synergies, no time-varying effectiveness. The full discussion is
in the note's counter-narratives section.

An external benchmark is prepared: Heusch (2026, arXiv:2608.21130) released a comparable
generator with endogenous spend days before this study was finalised; its companion
repository (oygo/mmm-synthetic) was announced but not yet public at the time of writing.
Running the four estimators on that reference instance is the natural next step.

## References

Jin, Wang, Sun, Chan, Koehler (2017), *Bayesian methods for media mix modeling with carryover
and shape effects*. Chan, Perry (2017), *Challenges and opportunities in media mix modeling*.
Runge, Patter, Skokan (2023), *Robyn: semi-automated marketing mix modeling*. Zhang et al.
(2024), *Media mix model calibration with Bayesian priors* (Meridian). Orduz (2024),
*Lift test calibration* (pymc-marketing). Heusch (2026), arXiv:2608.21130 and arXiv:2608.21128.

## License

MIT. If you use the generator or the sealed-truth protocol, a citation of the research note
is appreciated.
