import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def Ber(p):
    return 1 if np.random.random() < p else 0

def Bin(n,p):
    return sum([Ber(p) for _ in range(n)])

def Geom(p):
    count = 0
    while np.random.random() > p:
        count+=1
    return count

def Puas(la):
    l = math.exp(-la)
    k = 0
    p = 1

    while p > l:
        p *= np.random.random()
        k+=1
    return k-1

def Unif(a,b):
    return (b-a)*np.random.random() + a

def Exp(a):
    return -math.log(1 - np.random.random()) / a

def Lap(alpha):
    u = np.random.random()
    if u < 0.5:
        return math.log(2 * u) / alpha
    else:
        return -math.log(2 * (1 - u)) / alpha

def Stand():
    u1 = np.random.random()
    u2 = np.random.random()

    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

def Norm(a,si):
    sq_si = math.sqrt(si)
    y = Stand()
    return sq_si * y + a

def Cauch():
    return math.tan(math.pi * (np.random.random() - 0.5))

def lst():
    u = np.random.random()

    return 1 / math.sqrt(2*(1-u))


def demonstrate_distributions():
    np.random.seed(42)

    sample_size = 10000

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.ravel()

    p_ber = 0.3
    samples_ber = [Ber(p_ber) for _ in range(sample_size)]

    axes[0].hist(samples_ber, bins=3, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    x_ber = [0, 1]
    y_ber = [1 - p_ber, p_ber]
    axes[0].bar(x_ber, y_ber, alpha=0.5, color='red', width=0.3)
    axes[0].set_title(f'Бернулли (p={p_ber})')
    axes[0].set_xlabel('Значение')
    axes[0].set_ylabel('Вероятность')
    axes[0].legend(['Выборка', 'Теоретическое'])

    n_bin, p_bin = 10, 0.4
    samples_bin = [Bin(n_bin, p_bin) for _ in range(sample_size)]

    axes[1].hist(samples_bin, bins=range(n_bin + 2), density=True, alpha=0.7,
                 color='lightgreen', edgecolor='black')

    x_bin = np.arange(0, n_bin + 1)
    y_bin = [math.comb(n_bin, k) * (p_bin ** k) * ((1 - p_bin) ** (n_bin - k)) for k in x_bin]
    axes[1].bar(x_bin, y_bin, alpha=0.5, color='red', width=0.3)
    axes[1].set_title(f'Биномиальное (n={n_bin}, p={p_bin})')
    axes[1].set_xlabel('Значение')
    axes[1].set_ylabel('Вероятность')


    print("3. Геометрическое...")
    p_geom = 0.3
    samples_geom = [Geom(p_geom) for _ in range(sample_size)]

    max_geom = min(max(samples_geom), 20)
    axes[2].hist(samples_geom, bins=range(1, max_geom + 2), density=True, alpha=0.7,
                 color='lightcoral', edgecolor='black')

    x_geom = np.arange(1, max_geom + 1)
    y_geom = [p_geom * ((1 - p_geom) ** (k - 1)) for k in x_geom]
    axes[2].bar(x_geom, y_geom, alpha=0.5, color='red', width=0.3)
    axes[2].set_title(f'Геометрическое (p={p_geom})')
    axes[2].set_xlabel('Номер первого успеха')
    axes[2].set_ylabel('Вероятность')

    lambda_pois = 3
    samples_pois = [Puas(lambda_pois) for _ in range(sample_size)]

    max_pois = min(max(samples_pois), 15)
    axes[3].hist(samples_pois, bins=range(max_pois + 2), density=True, alpha=0.7,
                 color='gold', edgecolor='black')

    x_pois = np.arange(0, max_pois + 1)
    y_pois = [math.exp(-lambda_pois) * (lambda_pois ** k) / math.factorial(k) for k in x_pois]
    axes[3].bar(x_pois, y_pois, alpha=0.5, color='red', width=0.3)
    axes[3].set_title(f'Пуассона (λ={lambda_pois})')
    axes[3].set_xlabel('Значение')
    axes[3].set_ylabel('Вероятность')

    a_unif, b_unif = 2, 7
    samples_unif = [Unif(a_unif, b_unif) for _ in range(sample_size)]

    axes[4].hist(samples_unif, bins=50, density=True, alpha=0.7, color='violet')

    x_unif = np.linspace(a_unif - 1, b_unif + 1, 1000)
    y_unif = [1 / (b_unif - a_unif) if a_unif <= x <= b_unif else 0 for x in x_unif]
    axes[4].plot(x_unif, y_unif, 'r-', linewidth=2)
    axes[4].set_title(f'Равномерное [{a_unif}, {b_unif}]')
    axes[4].set_xlabel('x')
    axes[4].set_ylabel('Плотность')
    axes[4].legend(['Выборка', 'Теоретическая плотность'])

    alpha_exp = 0.5
    samples_exp = [Exp(alpha_exp) for _ in range(sample_size)]

    axes[5].hist(samples_exp, bins=50, density=True, alpha=0.7, color='orange')

    x_exp = np.linspace(0, 10, 1000)
    y_exp = alpha_exp * np.exp(-alpha_exp * x_exp)
    axes[5].plot(x_exp, y_exp, 'r-', linewidth=2)
    axes[5].set_title(f'Показательное (α={alpha_exp})')
    axes[5].set_xlabel('x')
    axes[5].set_ylabel('Плотность')
    axes[5].set_xlim(0, 10)

    alpha_laplace = 1
    samples_laplace = [Lap(alpha_laplace) for _ in range(sample_size)]

    axes[6].hist(samples_laplace, bins=50, density=True, alpha=0.7, color='cyan')

    x_laplace = np.linspace(-5, 5, 1000)
    y_laplace = (alpha_laplace / 2) * np.exp(-alpha_laplace * np.abs(x_laplace))
    axes[6].plot(x_laplace, y_laplace, 'r-', linewidth=2)
    axes[6].set_title(f'Лапласа (α={alpha_laplace})')
    axes[6].set_xlabel('x')
    axes[6].set_ylabel('Плотность')

    a_norm, sigma2_norm = 2, 4
    samples_norm = [Norm(a_norm, sigma2_norm) for _ in range(sample_size)]

    axes[7].hist(samples_norm, bins=50, density=True, alpha=0.7, color='pink')

    x_norm = np.linspace(a_norm - 10, a_norm + 10, 1000)
    sigma_norm = math.sqrt(sigma2_norm)
    y_norm = stats.norm.pdf(x_norm, a_norm, sigma_norm)
    axes[7].plot(x_norm, y_norm, 'r-', linewidth=2)
    axes[7].set_title(f'Нормальное (μ={a_norm}, σ²={sigma2_norm})')
    axes[7].set_xlabel('x')
    axes[7].set_ylabel('Плотность')

    samples_cauchy = [Cauch() for _ in range(sample_size)]

    samples_cauchy_trimmed = [x for x in samples_cauchy if -20 < x < 20]
    axes[8].hist(samples_cauchy_trimmed, bins=50, density=True, alpha=0.7, color='brown')

    x_cauchy = np.linspace(-20, 20, 1000)
    y_cauchy = 1 / (np.pi * (1 + x_cauchy ** 2))
    axes[8].plot(x_cauchy, y_cauchy, 'r-', linewidth=2)
    axes[8].set_title('Коши')
    axes[8].set_xlabel('x')
    axes[8].set_ylabel('Плотность')
    axes[8].set_xlim(-20, 20)

    samples_custom = [lst() for _ in range(sample_size)]

    samples_custom_trimmed = [x for x in samples_custom if x < 10]
    axes[9].hist(samples_custom_trimmed, bins=50, density=True, alpha=0.7, color='gray')

    x_custom = np.linspace(1, 10, 1000)
    y_custom = 2 / (x_custom ** 3)
    axes[9].plot(x_custom, y_custom, 'r-', linewidth=2)
    axes[9].set_title('Кастомное (f(t) = 1/t³ I{t>1})')
    axes[9].set_xlabel('t')
    axes[9].set_ylabel('Плотность')

    for i in range(10, 12):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

demonstrate_distributions()
