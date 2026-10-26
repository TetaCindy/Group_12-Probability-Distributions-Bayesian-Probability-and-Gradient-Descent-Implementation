"""Two-component 1-D Gaussian Mixture fitted with Expectation-Maximization (from scratch)."""
import numpy as np


def gaussian_pdf(x, mu, var):
    """N(x | mu, var) evaluated element-wise."""
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def log_weighted_densities(x, mu, var, pi):
    """(n, K) array of log( pi_k * N(x_i | mu_k, var_k) ), computed in log space so that
    heights far from both means (e.g. 170) do not underflow to 0 and give NaN."""
    x = np.asarray(x, dtype=float)[:, None]
    mu, var, pi = np.asarray(mu), np.asarray(var), np.asarray(pi)
    return np.log(pi) - 0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)


def log_likelihood(x, mu, var, pi):
    """sum_i log( sum_k pi_k N(x_i | mu_k, var_k) ), via the log-sum-exp trick."""
    lw = log_weighted_densities(x, mu, var, pi)
    top = lw.max(axis=1, keepdims=True)
    return float(np.sum(top[:, 0] + np.log(np.exp(lw - top).sum(axis=1))))


def e_step(x, mu, var, pi):
    """Responsibilities r[i, k] = pi_k N(x_i|mu_k,var_k) / sum_j pi_j N(x_i|mu_j,var_j).
    Subtracting the row maximum before exponentiating does not change the ratio."""
    lw = log_weighted_densities(x, mu, var, pi)
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def m_step(x, r):
    """Re-estimate mu, var, pi using responsibilities as soft weights."""
    Nk = r.sum(axis=0)                                   # effective points per component
    mu = (r * x[:, None]).sum(axis=0) / Nk
    var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk
    pi = Nk / len(x)
    return mu, var, pi


def initialize(x):
    """Deterministic init: means at 25th/75th percentile, shared variance, equal weights."""
    mu = np.percentile(x, [25, 75]).astype(float)
    var = np.array([x.var(), x.var()])
    pi = np.array([0.5, 0.5])
    return mu, var, pi


def fit_em(x, mu=None, var=None, pi=None, max_iter=500, tol=1e-6):
    """Run EM until the log-likelihood improves by less than tol.
    Returns final (mu, var, pi) and a history list of dicts (iteration 0 = initial state)."""
    x = np.asarray(x, dtype=float)
    if mu is None:
        mu, var, pi = initialize(x)
    history = [dict(iter=0, mu=mu.copy(), var=var.copy(), pi=pi.copy(),
                    ll=log_likelihood(x, mu, var, pi))]
    for it in range(1, max_iter + 1):
        r = e_step(x, mu, var, pi)
        mu, var, pi = m_step(x, r)
        ll = log_likelihood(x, mu, var, pi)
        history.append(dict(iter=it, mu=mu.copy(), var=var.copy(), pi=pi.copy(), ll=ll))
        if abs(ll - history[-2]["ll"]) < tol:
            break
    return mu, var, pi, history


def posterior(h, mu, var, pi):
    """P(component k | height h) for a single height."""
    return e_step(np.array([float(h)]), mu, var, pi)[0]


def classify_height(h, mu, var, pi, unit="in"):
    """Posterior for one test height. The Galton data is in INCHES, so a height given in
    centimetres must be converted first (unit="cm"). Returns (height_in_inches, posterior)."""
    h_in = float(h) / 2.54 if unit == "cm" else float(h)
    return h_in, posterior(h_in, mu, var, pi)


def hard_split(x):
    """The 'draw a line at the global mean' approach: cut at x.mean(), then take the
    mean / variance / share of each pile. Returns (cut, mu, var, pi) in the same layout as EM."""
    cut = x.mean()
    low, high = x[x <= cut], x[x > cut]
    return (cut, np.array([low.mean(), high.mean()]),
            np.array([low.var(), high.var()]), np.array([len(low), len(high)]) / len(x))


def uncertain_points(x, mu, var, pi, threshold=0.8):
    """Number of points whose best posterior is below `threshold`: points a hard split
    would label with 100% confidence even though EM is not sure which group they belong to."""
    return int((e_step(x, mu, var, pi).max(axis=1) < threshold).sum())


def agreement_both_mappings(x, mu, var, pi, is_male):
    """Component numbers are arbitrary, so compare BOTH mappings of the two components to the
    hidden labels and report the better one. Returns (best_agreement, mapping_text, both)."""
    comp2 = e_step(x, mu, var, pi).argmax(axis=1) == 1       # True -> Component 2
    a_high_male = float((comp2 == is_male).mean())            # Comp 2 = male,   Comp 1 = female
    a_high_female = float((comp2 != is_male).mean())          # Comp 2 = female, Comp 1 = male
    if a_high_male >= a_high_female:
        return a_high_male, "Component 2 = male, Component 1 = female", (a_high_male, a_high_female)
    return a_high_female, "Component 2 = female, Component 1 = male", (a_high_male, a_high_female)
