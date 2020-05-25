"""Normal Kalman Filters."""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax


def f(carry, noise_t):
    """Propagate forward the time series."""
    beta, z_prev, tau = carry
    z_t = beta * z_prev + noise_t
    z_prev = z_t
    return (beta, z_prev, tau), z_t


def singlevariate_kf(T=None, T_forecast=15, obs=None):
    """Define Kalman Filter in a single variate fashion.

    Parameters
    ----------
    T:  int
    T_forecast: int
        Times to forecast ahead.
    obs: np.array
        observed variable (infected, deaths...)

    """
    # Define priors over beta, tau, sigma, z_1 (keep the shapes in mind)
    T = len(obs) if T is None else T
    beta = numpyro.sample(name="beta", fn=dist.Normal(loc=0.0, scale=1))
    tau = numpyro.sample(name="tau", fn=dist.HalfCauchy(scale=0.1))
    noises = numpyro.sample(
        "noises", fn=dist.Normal(0, 1.0), sample_shape=(T + T_forecast - 2,)
    )
    sigma = numpyro.sample(name="sigma", fn=dist.HalfCauchy(scale=0.1))
    z_prev = numpyro.sample(name="z_1", fn=dist.Normal(loc=0, scale=0.1))

    # Propagate the dynamics forward using jax.lax.scan
    carry = (beta, z_prev, tau)
    z_collection = [z_prev]
    carry, zs_exp = lax.scan(f, carry, noises, T + T_forecast - 2)
    z_collection = jnp.concatenate((jnp.array(z_collection), zs_exp), axis=0)

    # Sample the observed y (y_obs) and missing y (y_mis)
    numpyro.sample(
        name="y_obs", fn=dist.Normal(loc=z_collection[:T], scale=sigma), obs=obs
    )
    numpyro.sample(
        name="y_pred", fn=dist.Normal(loc=z_collection[T:], scale=sigma), obs=None
    )


def twoh_c_kf(T=None, T_forecast=15, obs=None):
    """Define Kalman Filter with two hidden variates."""
    T = len(obs) if T is None else T
    
    # Define priors over beta, tau, sigma, z_1 (keep the shapes in mind)
    #W = numpyro.sample(name="W", fn=dist.Normal(loc=jnp.zeros((2,4)), scale=jnp.ones((2,4))))
    beta = numpyro.sample(name="beta", fn=dist.Normal(loc=jnp.array([0.,0.]), scale=jnp.ones(2)))
    tau = numpyro.sample(name="tau", fn=dist.HalfCauchy(scale=jnp.ones(2)))
    sigma = numpyro.sample(name="sigma", fn=dist.HalfCauchy(scale=.1))
    z_prev = numpyro.sample(name="z_1", fn=dist.Normal(loc=jnp.zeros(2), scale=jnp.ones(2)))
    # Define LKJ prior
    L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(2, 10.))
    Sigma_lower = jnp.matmul(jnp.diag(jnp.sqrt(tau)), L_Omega) # lower cholesky factor of the covariance matrix
    noises = numpyro.sample("noises", fn=dist.MultivariateNormal(loc=jnp.zeros(2), scale_tril=Sigma_lower), sample_shape=(T+T_forecast,))
    # Propagate the dynamics forward using jax.lax.scan
    carry = (beta, z_prev, tau)
    z_collection = [z_prev]
    carry, zs_exp = lax.scan(f, carry, noises, T+T_forecast)
    z_collection = jnp.concatenate((jnp.array(z_collection), zs_exp), axis=0)

    c = numpyro.sample(name="c", fn=dist.Normal(loc=jnp.array([[0.], [0.]]), scale=jnp.ones((2,1))))
    obs_mean = jnp.dot(z_collection[:T,:], c).squeeze()
    pred_mean = jnp.dot(z_collection[T:,:], c).squeeze()

    # Sample the observed y (y_obs)
    numpyro.sample(name="y_obs", fn=dist.Normal(loc=obs_mean, scale=sigma), obs=obs)
    numpyro.sample(name="y_pred", fn=dist.Normal(loc=pred_mean, scale=sigma), obs=None)


def multih_kf(T=None, T_forecast=15, hidden=4, obs=None):
    """Define Kalman Filter: multiple hidden variables; just one time series.

    Parameters
    ----------
    T:  int
    T_forecast: int
        Times to forecast ahead.
    hidden: int
        number of variables in the latent space
    obs: np.array
        observed variable (infected, deaths...)

    """
    # Define priors over beta, tau, sigma, z_1 (keep the shapes in mind)
    T = len(obs) if T is None else T
    beta = numpyro.sample(
        name="beta", fn=dist.Normal(loc=jnp.zeros(hidden), scale=jnp.ones(hidden))
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
        sample_shape=(T + T_forecast - 1,),
    )

    # Propagate the dynamics forward using jax.lax.scan
    carry = (beta, z_prev, tau)
    z_collection = [z_prev]
    carry, zs_exp = lax.scan(f, carry, noises, T + T_forecast - 1)
    z_collection = jnp.concatenate((jnp.array(z_collection), zs_exp), axis=0)

    # Sample the observed y (y_obs) and missing y (y_mis)
    numpyro.sample(
        name="y_obs",
        fn=dist.Normal(loc=z_collection[:T, :].sum(axis=1), scale=sigma),
        obs=obs[:, 0],
    )
    numpyro.sample(
        name="y_pred", fn=dist.Normal(loc=z_collection[T:, :].sum(axis=1), scale=sigma), obs=None
    )


def multivariate_kf(T=None, T_forecast=15, obs=None):
    """Define Kalman Filter in a multivariate fashion.

    The "time-series" are correlated. To define these relationships in
    a efficient manner, the covarianze matrix of h_t (or, equivalently, the
    noises) is drown from a Cholesky decomposed matrix.

    Parameters
    ----------
    T:  int
    T_forecast: int
    obs: np.array
       observed variable (infected, deaths...)

    """
    T = len(obs) if T is None else T
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
        sample_shape=(T + T_forecast - 1,),
    )

    # Propagate the dynamics forward using jax.lax.scan
    carry = (beta, z_prev, tau)
    z_collection = [z_prev]
    carry, zs_exp = lax.scan(f, carry, noises, T + T_forecast - 1)
    z_collection = jnp.concatenate((jnp.array(z_collection), zs_exp), axis=0)

    # Sample the observed y (y_obs) and missing y (y_mis)
    numpyro.sample(
        name="y_obs1",
        fn=dist.Normal(loc=z_collection[:T, 0], scale=sigma),
        obs=obs[:, 0],
    )
    numpyro.sample(
        name="y_pred1", fn=dist.Normal(loc=z_collection[T:, 0], scale=sigma), obs=None
    )
    numpyro.sample(
        name="y_obs2",
        fn=dist.Normal(loc=z_collection[:T, 1], scale=sigma),
        obs=obs[:, 1],
    )
    numpyro.sample(
        name="y_pred2", fn=dist.Normal(loc=z_collection[T:, 1], scale=sigma), obs=None
    )
