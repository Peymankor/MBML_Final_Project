"""Define Kalman filter."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from jax import lax


def f(carry, input_t):
    x_t, noise_t = input_t
    W, beta, z_prev, tau = carry
    z_t = beta * z_prev + W @ x_t + noise_t
    z_prev = z_t
    return (W, beta, z_prev, tau), z_t


def model_wo_c(T, T_forecast, x, obs=None):
    # Define priors over beta, tau, sigma, z_1 (keep the shapes in mind)
    W = numpyro.sample(
        name="W", fn=dist.Normal(loc=jnp.zeros((2, 4)), scale=jnp.ones((2, 4)))
    )
    beta = numpyro.sample(
        name="beta", fn=dist.Normal(loc=jnp.zeros(2), scale=jnp.ones(2))
    )
    tau = numpyro.sample(name="tau", fn=dist.HalfCauchy(scale=jnp.ones(2)))
    sigma = numpyro.sample(name="sigma", fn=dist.HalfCauchy(scale=0.1))
    z_prev = numpyro.sample(
        name="z_1", fn=dist.Normal(loc=jnp.zeros(2), scale=jnp.ones(2))
    )
    # Define LKJ prior
    L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(2, 10.0))
    Sigma_lower = jnp.matmul(
        jnp.diag(jnp.sqrt(tau)), L_Omega
    )  # lower cholesky factor of the covariance matrix
    noises = numpyro.sample(
        "noises",
        fn=dist.MultivariateNormal(loc=jnp.zeros(2), scale_tril=Sigma_lower),
        sample_shape=(T + T_forecast,),
    )
    # Propagate the dynamics forward using jax.lax.scan
    carry = (W, beta, z_prev, tau)
    z_collection = [z_prev]
    carry, zs_exp = lax.scan(f, carry, (x, noises), T + T_forecast)
    z_collection = jnp.concatenate((jnp.array(z_collection), zs_exp), axis=0)

    obs_mean = z_collection[:T, 1]
    pred_mean = z_collection[T:, 1]

    # Sample the observed y (y_obs)
    numpyro.sample(name="y_obs", fn=dist.Normal(loc=obs_mean, scale=sigma), obs=obs)
    numpyro.sample(name="y_pred", fn=dist.Normal(loc=pred_mean, scale=sigma), obs=None)


def model_w_c(T, T_forecast, x, obs=None):
    # Define priors over beta, tau, sigma, z_1 (keep the shapes in mind)
    W = numpyro.sample(
        name="W", fn=dist.Normal(loc=jnp.zeros((2, 4)), scale=jnp.ones((2, 4)))
    )
    beta = numpyro.sample(
        name="beta", fn=dist.Normal(loc=jnp.array([0.0, 0.0]), scale=jnp.ones(2))
    )
    tau = numpyro.sample(name="tau", fn=dist.HalfCauchy(scale=jnp.ones(2)))
    sigma = numpyro.sample(name="sigma", fn=dist.HalfCauchy(scale=0.1))
    z_prev = numpyro.sample(
        name="z_1", fn=dist.Normal(loc=jnp.zeros(2), scale=jnp.ones(2))
    )
    # Define LKJ prior
    L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(2, 10.0))
    Sigma_lower = jnp.matmul(
        jnp.diag(jnp.sqrt(tau)), L_Omega
    )  # lower cholesky factor of the covariance matrix
    noises = numpyro.sample(
        "noises",
        fn=dist.MultivariateNormal(loc=jnp.zeros(2), scale_tril=Sigma_lower),
        sample_shape=(T + T_forecast,),
    )
    # Propagate the dynamics forward using jax.lax.scan
    carry = (W, beta, z_prev, tau)
    z_collection = [z_prev]
    carry, zs_exp = lax.scan(f, carry, (x, noises), T + T_forecast)
    z_collection = jnp.concatenate((jnp.array(z_collection), zs_exp), axis=0)

    c = numpyro.sample(
        name="c", fn=dist.Normal(loc=jnp.array([[0.0], [0.0]]), scale=jnp.ones((2, 1)))
    )
    obs_mean = jnp.dot(z_collection[:T, :], c).squeeze()
    pred_mean = jnp.dot(z_collection[T:, :], c).squeeze()

    # Sample the observed y (y_obs)
    numpyro.sample(name="y_obs", fn=dist.Normal(loc=obs_mean, scale=sigma), obs=obs)
    numpyro.sample(name="y_pred", fn=dist.Normal(loc=pred_mean, scale=sigma), obs=None)
