import numpy as np
from scipy import stats

# Bayes threshold
def bayes_threshold(priors, cost_false_alarm, cost_missed_detection):
    """Compute eta.

    Args:
        priors (float): array of prior probabilities
        cost_false_alarm (float): cost of false alarm
        cost_missed_detection (float): cost of missed detection

    Returns:
        float: the threshold eta
    """
    num = priors[0] * cost_false_alarm
    den = priors[1] * cost_missed_detection
    threshold = num / den
    return threshold

# Hypothesis selector
def select_hypothesis(priors):
    """Randomly pick a hypothesis based where priors[i] is the probability of hypothesis H_i.

    Args:
        priors (float): hypothesis prior probabilities

    Returns:
        float: the hypothesis index, 0 or 1
    """
    hypothesis = stats.bernoulli(p=priors[1]).rvs()
    return hypothesis

# Samples generators
def generate_samples_gaussian(mean, std, n_samples):
    """Generate observed samples.

    Args:
        mean (float): mean of the gaussian distribution
        std (float): standard deviation of the gaussian distribution
        n_samples (int): number of samples to draw

    Returns:
        float: observed samples
    """
    y = stats.norm(loc=mean, scale=std).rvs(size=n_samples)
    return y

def generate_samples_poisson(lamb, n_samples):
    """Generate samples from a Poisson distributed random variable.

    Args:
        lamb (float): parameter of the Poisson distribution
        n_samples (_type_): number of samples to draw

    Returns:
        float: Poisson samples
    """
    y = stats.poisson(lamb).rvs(size=n_samples)
    return y

def generate_samples_exponential(lamb, n_samples):
    """Generate samples from an Exponential distributed random variable.

    Args:
        lamb (float): parameter of the Exponential distribution
        n_samples (_type_): number of samples to draw

    Returns:
        float: Poisson samples
    """
    y = stats.expon(scale=lamb).rvs(size=n_samples)
    return y

def generate_samples_binary(lamb, n_samples):
    """Generate samples from a Bernoulli distributed random variable.

    Args:
        lamb (float): probability of success
        n_samples (_type_): number of samples to draw

    Returns:
        float: Bernoulli samples
    """
    y = stats.bernoulli(p=lamb).rvs(size=n_samples)
    return y


# Test thresholds
def test_threshold_gaussian(eta, means, std, n_samples):
    """Compute the threshold for the Bayes test with Gaussian likelihoods.

    Args:
        eta (float): Bayes threshold
        means (float): hypothesis means
        std (float): noise standard deviation
        n_samples (int): number of observed samples

    Returns:
        float: the threshold gamma
    """
    a = std**2 * np.log(eta) / (n_samples * np.abs(means[1] - means[0]))
    b = 0.5 * (means[0] + means[1])
    threshold = a + b
    return threshold

def test_threshold_exponential(eta, lambs, n_samples):
    """Compute the threshold for the Bayes test with Gaussian likelihoods.

    Args:
        eta (float): Bayes threshold
        lambs (float): noise standard deviations
        n_samples (int): number of observed samples

    Returns:
        float: the threshold gamma
    """
    a = 2 * np.log(eta) + n_samples * (np.log(lambs[1]) - np.log(lambs[0]))
    b = 2 * n_samples * (1 / lambs[0] - 1 / lambs[1])
    threshold = a / b
    return threshold

def test_threshold_poisson(eta, lambs, n_samples):
    """Compute the threshold for the Bayes test with Poisson likelihoods.

    Args:
        eta (float): Bayes threshold
        lambds (float): hypothesis lambdas
        n_samples (int): number of observed samples

    Returns:
        float: the threshold gamma
    """
    a = np.log(eta) + n_samples * (lambs[1] - lambs[0])
    b = n_samples * (np.log(lambs[1]) - np.log(lambs[0]))
    threshold = a / b
    return threshold

def test_threshold_binary(eta, lambs, n_samples):
    """Compute the threshold for the Bayes test with Bernoulli likelihoods.

    Args:
        eta (float): Bayes threshold
        lambds (float): transition probabilities
        n_samples (int): number of observed samples

    Returns:
        float: the threshold gamma
    """
    a = np.log(eta) + n_samples * (np.log(1 - lambs[0]) - np.log(lambs[1]))
    b = n_samples * (np.log(1 - lambs[0]) - np.log(lambs[1]) + np.log(1 - lambs[1]) - np.log(lambs[0]))
    threshold = a / b
    return threshold

# Bayes tests
def bayes_test(y, threshold):
    """Compute the hypothesis decided by the Bayes test from the observed samples for some threshold and ret

    Args:
        y (float): observed samples
        threshold (float): test threshold

    Returns:
        int: the hypothesis index, 0 or 1
    """
    S = y.mean()
    hypothesis =  int(S > threshold)
    return hypothesis