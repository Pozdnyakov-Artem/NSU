import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from second import generate_distribution_samples


def get_theoretical_moments(dist_type, params):
    """Вычисление теоретических математического ожидания и дисперсии"""
    if dist_type == 'bernoulli':
        p = params.get('p', 0.5)
        return p, p * (1 - p)

    elif dist_type == 'binomial':
        n_trials = params.get('n', 10)
        p = params.get('p', 0.5)
        return n_trials * p, n_trials * p * (1 - p)

    elif dist_type == 'geometric':
        p = params.get('p', 0.5)
        return 1 / p, (1 - p) / (p ** 2)

    elif dist_type == 'poisson':
        lambd = params.get('lambda', 1.0)
        return lambd, lambd

    elif dist_type == 'uniform':
        a = params.get('a', 0.0)
        b = params.get('b', 1.0)
        return (a + b) / 2, (b - a) ** 2 / 12

    elif dist_type == 'exponential':
        alpha = params.get('alpha', 1.0)
        return 1 / alpha, 1 / (alpha ** 2)

    elif dist_type == 'laplace':
        alpha = params.get('alpha', 1.0)
        return 0, 2 / (alpha ** 2)

    else:
        # Для неизвестных распределений генерируем большую выборку для оценки
        large_sample = generate_distribution_samples(dist_type, params, 100000)
        return np.mean(large_sample), np.var(large_sample)


def generate_standard_normal_by_clt(n, N, params, dist_type):

    normal_samples = []

    sample_mean, sample_std2 = get_theoretical_moments(dist_type, params)
    sample_std = math.sqrt(sample_std2)
    for i in range(N):
        xi_samples = generate_distribution_samples(dist_type, params, n)

        standardized_sum = (np.sum(xi_samples) - n * sample_mean) / (sample_std * np.sqrt(n))
        normal_samples.append(standardized_sum)

    return normal_samples


def visualize_clt(dist_name, n_values, N = 10000, dist_params = None):
    if dist_params is None:
        dist_params = {}

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i, n in enumerate(n_values):
        if i >= len(axes):
            break

        normal_samples = generate_standard_normal_by_clt( n, N, dist_params, dist_name)

        axes[i].hist(normal_samples, bins=50, density=True, alpha=0.7,
                     color='skyblue', label='ЦПТ приближение')

        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x)
        axes[i].plot(x, y, 'r-', linewidth=2, label='N(0,1) плотность')

        sample_mean = np.mean(normal_samples)
        sample_std = np.std(normal_samples)

        axes[i].set_title(f'{dist_name}, n = {n}\n'
                          f'Среднее: {sample_mean:.3f}, Ст. отклонение: {sample_std:.3f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('Плотность')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(-4, 4)

    plt.tight_layout()
    plt.show()


visualize_clt('binomial',[5, 10, 30, 10000], 10000, {'p': 0.3})