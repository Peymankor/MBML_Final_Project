"""Handy functions to train Kalman filters in numpyro."""
from jax import random
from numpyro.infer import MCMC, NUTS


def train_kf(model, data, n_train, n_test, x=None, num_samples=9000, num_warmup=3000):
    """Train a Kalman Filter model."""
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)

    nuts_kernel = NUTS(model=model)
    # burn-in is still too much in comparison with the samples
    mcmc = MCMC(
        nuts_kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=1
    )
    # let T be guessed from the length of observed
    if x is None:
        mcmc.run(rng_key_, T=n_train, T_forecast=n_test, obs=data)
    else:
        mcmc.run(rng_key_, T=n_train, T_forecast=n_test, x=x, obs=data)
    return mcmc
