import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def binomial_pmf(n, p, k):
    """
    Bereken de probability mass function (pmf) voor de binomiale verdeling.

    Parameters:
    n (int): Het aantal experimenten.
    p (float): De kans op succes bij elk experiment.
    k (int): Het aantal successen waarvoor de kans wordt berekend.

    Returns:
    float: De kans op k successen in n experimenten.
    """
    return stats.binom.pmf(k, n, p)


def binomial_pmf_distribution(n, p):
    """
    Bereken de volledige probability mass function (pmf) distributie voor de binomiale verdeling.

    Parameters:
    n (int): Het aantal experimenten.
    p (float): De kans op succes bij elk experiment.

    Returns:
    numpy.ndarray: De pmf waarden voor k = 0 tot k = n.
    """
    k = np.arange(0, n + 1)
    pmf = stats.binom.pmf(k, n, p)
    return k, pmf


def plot_binomial_pmf(n, p):
    """
    Plot de pmf voor de binomiale verdeling.

    Parameters:
    n (int): Het aantal experimenten.
    p (float): De kans op succes bij elk experiment.
    """
    k, pmf = binomial_pmf_distribution(n, p)
    plt.bar(k, pmf)
    plt.xlabel('Aantal successen (k)')
    plt.ylabel('Kans (P(k))')
    plt.title(f'Binomiale PMF met n = {n}, p = {p}')
    plt.show()


def binomial_cdf(n, p, k):
    """
    Bereken de cumulatieve verdeling (cdf) voor de binomiale verdeling.

    Parameters:
    n (int): Het aantal experimenten.
    p (float): De kans op succes bij elk experiment.
    k (int): Het aantal successen waarvoor de cumulatieve kans wordt berekend.

    Returns:
    float: De cumulatieve kans voor k of minder successen.
    """
    return stats.binom.cdf(k, n, p)


def binomial_survival_function(n, p, k):
    """
    Bereken de kans op meer dan k successen voor de binomiale verdeling.

    Parameters:
    n (int): Het aantal experimenten.
    p (float): De kans op succes bij elk experiment.
    k (int): Het aantal successen waarvoor de overlevingsfunctie wordt berekend.

    Returns:
    float: De kans op meer dan k successen.
    """
    return 1 - binomial_cdf(n, p, k)


def poisson_probability(lam, k):
    """
    Bereken de kans op k gebeurtenissen voor de Poisson-verdeling.

    Parameters:
    lam (float): Het gemiddelde aantal gebeurtenissen in een gegeven tijdsinterval.
    k (int): Het aantal gebeurtenissen waarvoor de kans wordt berekend.

    Returns:
    float: De kans op k gebeurtenissen.
    """
    return stats.poisson.pmf(k, lam)


def poisson_pmf(lam, x_max):
    """
    Bereken de probability mass function (pmf) voor de Poisson-verdeling.

    Parameters:
    lam (float): Het gemiddelde aantal gebeurtenissen in een gegeven tijdsinterval.
    x_max (int): Het maximale aantal gebeurtenissen waarvoor de pmf wordt berekend.

    Returns:
    numpy.ndarray: De pmf waarden voor x = 0 tot x = x_max.
    """
    x = np.arange(0, x_max + 1)
    pmf = stats.poisson.pmf(x, lam)
    return x, pmf


def plot_poisson_pmf(lam, x_max):
    """
    Plot de pmf voor de Poisson-verdeling.

    Parameters:
    lam (float): Het gemiddelde aantal gebeurtenissen in een gegeven tijdsinterval.
    x_max (int): Het maximale aantal gebeurtenissen waarvoor de pmf wordt berekend.
    """
    x, pmf = poisson_pmf(lam, x_max)
    plt.bar(x, pmf)
    plt.xlabel('Aantal gebeurtenissen (k)')
    plt.ylabel('Kans (P(k))')
    plt.title(f'Poisson PMF met λ = {lam}')
    plt.show()


def poisson_cdf(lam, k):
    """
    Bereken de cumulatieve verdeling (cdf) voor de Poisson-verdeling.

    Parameters:
    lam (float): Het gemiddelde aantal gebeurtenissen in een gegeven tijdsinterval.
    k (int): Het aantal gebeurtenissen waarvoor de cdf wordt berekend.

    Returns:
    float: De cumulatieve kans voor k of minder gebeurtenissen.
    """
    return stats.poisson.cdf(k, lam)


def poisson_survival_function(lam, k):
    """
    Bereken de kans op meer dan k gebeurtenissen voor de Poisson-verdeling.

    Parameters:
    lam (float): Het gemiddelde aantal gebeurtenissen in een gegeven tijdsinterval.
    k (int): Het aantal gebeurtenissen waarvoor de overlevingsfunctie wordt berekend.

    Returns:
    float: De kans op meer dan k gebeurtenissen.
    """
    return 1 - poisson_cdf(lam, k)


def poisson_ppf(lam, prob):
    """
    Bereken het aantal gebeurtenissen k waarvoor de cumulatieve kans gelijk is aan prob.

    Parameters:
    lam (float): Het gemiddelde aantal gebeurtenissen in een gegeven tijdsinterval.
    prob (float): De cumulatieve kans.

    Returns:
    float: Het aantal gebeurtenissen k waarvoor de cumulatieve kans gelijk is aan prob.
    """
    return stats.poisson.ppf(prob, lam)
