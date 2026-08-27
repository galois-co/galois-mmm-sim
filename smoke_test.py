import time
import numpy as np
import pymc as pm
import arviz as az


def main():
    rng = np.random.default_rng(0)
    n = 156
    x = rng.normal(size=(n, 3))
    y = x @ np.array([1.0, 2.0, 0.5]) + rng.normal(scale=0.5, size=n)

    with pm.Model():
        beta = pm.Normal("beta", 0, 5, shape=3)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("y", mu=x @ beta, sigma=sigma, observed=y)
        t0 = time.time()
        idata = pm.sample(1000, tune=1000, chains=4, cores=4, random_seed=0, progressbar=False)

    print(f"elapsed {time.time() - t0:.1f}s")
    print(az.summary(idata, var_names=["beta", "sigma"]))


if __name__ == "__main__":
    main()
