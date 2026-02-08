import numpy as np

def generate_distribution_samples(dist_type: str, params: dict, n: int = 1000):

    if dist_type == 'bernoulli':
        p = params.get('p', 0.5)
        return np.random.binomial(1, p, n)

    elif dist_type == 'binomial':
        n_trials = params.get('n', 10)
        p = params.get('p', 0.5)
        return np.random.binomial(n_trials, p, n)

    elif dist_type == 'geometric':
        p = params.get('p', 0.5)
        return np.random.geometric(p, n)

    elif dist_type == 'poisson':
        lambd = params.get('lambda', 1.0)
        return np.random.poisson(lambd, n)

    elif dist_type == 'uniform':
        a = params.get('a', 0.0)
        b = params.get('b', 1.0)
        return np.random.uniform(a, b, n)

    elif dist_type == 'exponential':
        alpha = params.get('alpha', 1.0)
        return np.random.exponential(1 / alpha, n)

    elif dist_type == 'laplace':
        alpha = params.get('alpha', 1.0)
        return np.random.laplace(0, 1 / alpha, n)

    elif dist_type == 'normal':
        a = params.get('a', 0.0)
        sigma = params.get('sigma', 1.0)
        return np.random.normal(a, sigma, n)

    elif dist_type == 'cauchy':
        return np.random.standard_cauchy(n)

    elif dist_type == 'gamma':
        shape = params.get('shape', 2.0)
        scale = params.get('scale', 1.0)
        return np.random.gamma(shape, scale, n)

    elif dist_type == 'beta':
        a = params.get('a', 2.0)
        b = params.get('b', 2.0)
        return np.random.beta(a, b, n)


def sample_skewness(data) -> float:
    n = len(data)
    if n < 3:
        return float('nan')

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    if std_dev == 0:
        return 0.0

    m3 = np.mean((data - mean) ** 3)
    skewness = m3 / ((std_dev ** 3))
    return skewness


def sample_kurtosis(data) -> float:

    n = len(data)
    if n < 4:
        return float('nan')

    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    if std_dev == 0:
        return -3.0

    m4 = np.mean((data - mean) ** 4)
    kurtosis = m4 / (std_dev ** 4) - 3

    return kurtosis

data = generate_distribution_samples('bernoulli', {'p' : 0.2})
print(sample_skewness(data))
print(sample_kurtosis(data))