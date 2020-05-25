"""Visualize results of training MCMC."""
import seaborn as sns
import matplotlib.pyplot as plt


def get_samples(mcmc):
    """Get samples from variables in MCMC."""
    return {k: v for k, v in mcmc.get_samples().items()}


def plot_samples(hmc_samples, nodes):
    """Plot samples from the variables in `nodes`."""
    for node in nodes:
        if len(hmc_samples[node].shape) > 1:
            n_vars = hmc_samples[node].shape[1]
            for i in range(n_vars):
                plt.figure(figsize=(4, 3))
                sns.distplot(hmc_samples[node][:, i], label=node + "%d" % i)
                plt.legend()
                plt.show()
        else:
            plt.figure(figsize=(4, 3))
            sns.distplot(hmc_samples[node], label=node)
            plt.legend()
            plt.show()


def plot_forecast(hmc_samples, idx_train, idx_test, y_train, y_test):
    """Plot the results of forecasting."""
    y_hat = hmc_samples["y_pred"].mean(axis=0)
    y_std = hmc_samples["y_pred"].std(axis=0)
    y_pred_025 = y_hat - 1.96 * y_std
    y_pred_975 = y_hat + 1.96 * y_std
    plt.plot(idx_train, y_train, "b-")
    plt.plot(idx_test, y_test, "bx")
    plt.plot(idx_test, y_hat[:-1], "r-")
    plt.plot(idx_test, y_pred_025[:-1], "r--")
    plt.plot(idx_test, y_pred_975[:-1], "r--")
    plt.fill_between(idx_test, y_pred_025[:-1], y_pred_975[:-1], alpha=0.3)
    plt.legend(
        [
            "true (train)",
            "true (test)",
            "forecast",
            "forecast + stddev",
            "forecast - stddev",
        ]
    )
    plt.show()
