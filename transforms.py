"""Shared media transformations: adstock, saturation, seasonality.

Numpy implementations used by the simulator, the ridge estimator and the
evaluation code. The Bayesian model reimplements the same formulas with
pytensor ops (see fit_bayes.py) so that simulator and estimators agree exactly.

Conventions
- Adstock is geometric with a finite window l_max and, by default, weights
  normalised to sum to one. Normalisation keeps the adstocked series on the
  same scale as spend, so the Hill half-saturation K is expressed in spend units.
- Hill saturation returns values in [0, 1); the channel coefficient beta
  carries the scale.
- Fourier terms use a period of 52.1775 weeks (365.25 / 7).
"""

import numpy as np


def adstock_weights(decay: float, l_max: int = 13, normalize: bool = True) -> np.ndarray:
    """Weights decay**0, decay**1, ..., decay**(l_max-1), optionally summing to 1."""
    w = decay ** np.arange(l_max, dtype=float)
    if normalize:
        w = w / w.sum()
    return w


def geometric_adstock(x, decay: float, l_max: int = 13, normalize: bool = True) -> np.ndarray:
    """Carryover: a_t = sum_{lag} w_lag * x_{t-lag}, causal, zero before the series starts."""
    x = np.asarray(x, dtype=float)
    w = adstock_weights(decay, l_max, normalize)
    out = np.zeros_like(x)
    n = len(x)
    for lag, w_lag in enumerate(w):
        if lag >= n:
            break
        out[lag:] += w_lag * x[: n - lag]
    return out


def half_life(decay: float) -> float:
    """Weeks until the carried-over effect halves."""
    return float(np.log(0.5) / np.log(decay))


def hill(x, k: float, s: float) -> np.ndarray:
    """Diminishing returns: x**s / (k**s + x**s). Equals 0.5 at x = k."""
    x = np.asarray(x, dtype=float)
    return x**s / (k**s + x**s)


def fourier_terms(n: int, period: float = 52.1775, order: int = 3) -> np.ndarray:
    """Annual seasonality basis: columns sin(h), cos(h) for h = 1..order, shape (n, 2*order)."""
    t = np.arange(n, dtype=float)
    cols = []
    for h in range(1, order + 1):
        cols.append(np.sin(2 * np.pi * h * t / period))
        cols.append(np.cos(2 * np.pi * h * t / period))
    return np.column_stack(cols)


if __name__ == "__main__":
    # Adstock of a unit impulse reproduces the weights, then zero.
    impulse = np.zeros(20)
    impulse[0] = 1.0
    raw = geometric_adstock(impulse, 0.5, l_max=5, normalize=False)
    assert np.allclose(raw[:5], [1, 0.5, 0.25, 0.125, 0.0625]) and np.allclose(raw[5:], 0)

    # Normalised weights sum to one, so constant spend is left unchanged once the window is full.
    assert np.isclose(geometric_adstock(impulse, 0.5, l_max=5).sum(), 1.0)
    const = np.full(60, 100.0)
    assert np.isclose(geometric_adstock(const, 0.7, l_max=13)[-1], 100.0)

    # Hill: zero at zero, half at k, increasing, bounded below one.
    assert np.isclose(hill(2.0, 2.0, 1.5), 0.5)
    grid = np.linspace(0, 10, 101)
    h = hill(grid, 2.0, 1.5)
    assert h[0] == 0.0 and np.all(np.diff(h) > 0) and h[-1] < 1.0

    # Fourier basis: right shape, columns roughly centred over three years.
    F = fourier_terms(156, order=3)
    assert F.shape == (156, 6) and np.all(np.abs(F.mean(axis=0)) < 0.05)

    print("half-lives (weeks) for the planned channel decays:")
    for name, d in [("search_nonbrand", 0.15), ("search_brand", 0.10), ("social", 0.35),
                    ("display", 0.40), ("video", 0.70), ("affiliate", 0.25)]:
        print(f"  {name:16s} decay={d:.2f}  half-life={half_life(d):.2f}")
    print("transforms: all checks passed")
