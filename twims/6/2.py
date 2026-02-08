import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def load_housing_data():
    df = pd.read_csv('House-Prices - House-Prices.csv')

    df['waterbody'] = df['waterbody'].fillna('None')

    return df


def beta_params_from_mean_std(mean, std):

    var = std ** 2

    k = (mean * (1 - mean) / var) - 1

    alpha = mean * k
    beta = (1 - mean) * k

    return alpha, beta


def mean_std_from_beta_params(alpha, beta):
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    std = np.sqrt(var)

    return mean, std

def bayesian_housing_analysis(df, prior_mean=0.5, prior_std=0.1, update_order=None):
    prior_alpha, prior_beta = beta_params_from_mean_std(prior_mean, prior_std)

    groups = {}
    waterbody_types = sorted(df['waterbody'].unique())

    for wb_type in waterbody_types:
        group_data = df[df['waterbody'] == wb_type]
        n_sold = group_data['Sold'].sum()
        n_total = len(group_data)
        p_observed = n_sold / n_total

        groups[wb_type] = {
            'data': group_data,
            'n_sold': n_sold,
            'n_total': n_total,
            'p_observed': p_observed
        }

    posteriors = {}
    for wb_type in waterbody_types:
        group = groups[wb_type]

        alpha_post = prior_alpha + group['n_sold']
        beta_post = prior_beta + (group['n_total'] - group['n_sold'])

        post_mean, post_std = mean_std_from_beta_params(alpha_post, beta_post)

        posteriors[wb_type] = {
            'alpha': alpha_post,
            'beta': beta_post,
            'mean': post_mean,
            'std': post_std,
            'n_sold': group['n_sold'],
            'n_total': group['n_total']
        }

    if update_order is None:
        update_order = sorted(waterbody_types,
                              key=lambda x: groups[x]['n_total'],
                              reverse=True)

    current_alpha = prior_alpha
    current_beta = prior_beta
    current_mean, current_std = prior_mean, prior_std

    updates = [{
        'stage': 'Априор',
        'alpha': current_alpha,
        'beta': current_beta,
        'mean': current_mean,
        'std': current_std,
        'group': None,
        'n_sold': 0,
        'n_total': 0
    }]

    for i, wb_type in enumerate(update_order, 1):
        group = groups[wb_type]

        current_alpha += group['n_sold']
        current_beta += (group['n_total'] - group['n_sold'])

        current_mean, current_std = mean_std_from_beta_params(current_alpha, current_beta)

        updates.append({
            'stage': f'Шаг {i}: {wb_type}',
            'alpha': current_alpha,
            'beta': current_beta,
            'mean': current_mean,
            'std': current_std,
            'group': wb_type,
            'n_sold': group['n_sold'],
            'n_total': group['n_total']
        })

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes = axes.flatten()

    ax = axes[0]
    x = np.linspace(0, 1, 1000)

    ax.plot(x, stats.beta.pdf(x, prior_alpha, prior_beta),
            'k--', linewidth=3, label='Априор', alpha=0.7)

    colors = plt.cm.Set1(np.linspace(0, 1, len(waterbody_types)))
    for (wb_type, post), color in zip(posteriors.items(), colors):
        ax.plot(x, stats.beta.pdf(x, post['alpha'], post['beta']),
                linewidth=2, label=f'{wb_type}', color=color, alpha=0.8)

    ax.set_xlabel('Вероятность продажи (p)', fontsize=12)
    ax.set_ylabel('Плотность вероятности', fontsize=12)
    ax.set_title('Априорное и апостериорные распределения\nпо группам водоёмов', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, update in enumerate(updates):
        if i == 0:
            ax.plot(x, stats.beta.pdf(x, update['alpha'], update['beta']),
                    'k--', linewidth=3, label=update['stage'], alpha=0.7)
        else:
            color = plt.cm.viridis(i / len(updates))
            ax.plot(x, stats.beta.pdf(x, update['alpha'], update['beta']),
                    linewidth=2, label=update['stage'], color=color, alpha=0.8)

    ax.set_xlabel('Вероятность продажи (p)', fontsize=12)
    ax.set_ylabel('Плотность вероятности', fontsize=12)
    ax.set_title('Последовательные байесовские обновления', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("=" * 60)
    print("СРАВНЕНИЕ ГРУПП ПО ВОДОЁМАМ")
    print("=" * 60)
    print(f"{'Тип водоёма':<20} {'Домов':<10} {'Продано':<10} {'p_набл':<10} {'p_апост':<10} {'σ_апост':<10}")
    print("-" * 80)

    for wb_type in sorted(waterbody_types, key=lambda x: posteriors[x]['mean'], reverse=True):
        post = posteriors[wb_type]
        print(f"{wb_type:<20} {post['n_total']:<10} {post['n_sold']:<10} "
              f"{post['n_sold'] / post['n_total']:.4f}     {post['mean']:.4f}      {post['std']:.4f}")

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ С АПРИОРОМ")
    print("=" * 60)
    print(f"Априор: среднее = {prior_mean:.4f}, стандартное отклонение = {prior_std:.4f}")

    for wb_type in waterbody_types:
        post = posteriors[wb_type]
        mean_change = abs(post['mean'] - prior_mean) / prior_std
        std_change = (prior_std - post['std']) / prior_std

        print(f"\n{wb_type}:")
        print(f"  Изменение среднего: {post['mean']:.4f} ({mean_change:} от априорного σ)")
        print(f"  Изменение неопределенности: σ уменьшилась на {std_change:}")




df = load_housing_data()
bayesian_housing_analysis(df,0.8, 0.1)