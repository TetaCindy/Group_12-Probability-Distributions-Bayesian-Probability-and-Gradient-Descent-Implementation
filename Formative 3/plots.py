"""Shared plotting helpers (kept here so notebooks/scripts never repeat plotting code)."""
import numpy as np
import matplotlib.pyplot as plt
from em import gaussian_pdf


def mixture_curve(g, mu, var, pi):
    return sum(pi[k] * gaussian_pdf(g, mu[k], var[k]) for k in range(len(mu)))


def plot_em_fit(x, mu, var, pi, groups=None, cut=None, title="EM fit"):
    """Histogram of the pooled heights with each fitted component, the mixture, and
    (optionally) the hard-split line. `groups` = {label: array} for background histograms."""
    g = np.linspace(x.min() - 2, x.max() + 2, 400)
    fig, ax = plt.subplots(figsize=(9, 5))
    if groups:
        for name, arr in groups.items():
            ax.hist(arr, bins=30, density=True, alpha=.3, label=name)
    else:
        ax.hist(x, bins=40, density=True, alpha=.3, label="all heights")
    for k in range(len(mu)):
        ax.plot(g, pi[k] * gaussian_pdf(g, mu[k], var[k]), lw=2, label=f"component {k+1} (mu={mu[k]:.1f})")
    ax.plot(g, mixture_curve(g, mu, var, pi), "k--", label="EM mixture")
    if cut is not None:
        ax.axvline(cut, color="red", ls=":", lw=2, label=f"global-mean split ({cut:.1f})")
    ax.set(xlabel="height (inches)", ylabel="density", title=title)
    ax.legend()
    return fig


def plot_posterior_curve(mu, var, pi, h=None):
    from em import posterior
    g = np.linspace(58, 78, 300)
    p = np.array([posterior(v, mu, var, pi) for v in g])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g, p[:, 0], label="P(component 1 | height)")
    ax.plot(g, p[:, 1], label="P(component 2 | height)")
    if h is not None:
        ax.axvline(h, color="gray", ls=":")
    ax.set(xlabel="height (inches)", ylabel="posterior", title="Soft assignment vs. height")
    ax.legend()
    return fig


def plot_parameters(history):
    """m1, m2, b1, b2 against iteration."""
    its = [h["iter"] for h in history]
    m = np.array([h["m"] for h in history]); b = np.array([h["b"] for h in history])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(m.shape[1]):
        ax.plot(its, m[:, i], label=f"m{i+1}")
    for i in range(b.shape[1]):
        ax.plot(its, b[:, i], "--", label=f"b{i+1}")
    ax.set(xlabel="iteration", ylabel="value", title="Parameters m and b over iterations")
    ax.legend(); ax.grid(alpha=.3)
    return fig


def plot_error(history, log=False):
    its = [h["iter"] for h in history]
    J = [h["J"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(its, J, color="crimson")
    if log:
        ax.set_yscale("log")
    ax.set(xlabel="iteration", ylabel="MSE  J(m, b)", title="Error over iterations")
    ax.grid(alpha=.3)
    return fig
