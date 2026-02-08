import matplotlib.pyplot as plt
import pandas
import numpy


def is_list_of_numbers(value):
    """
    Проверяет, является ли значение списком чисел.

    Args:
        value: Значение для проверки

    Returns:
        bool: True если значение является списком чисел, иначе False
    """
    if not isinstance(value, (list, pandas.core.series.Series, numpy.ndarray)):
        return False

    # Преобразуем в список для удобства проверки
    if hasattr(value, 'tolist'):
        value_list = value.tolist()
    else:
        value_list = list(value)

    return all(isinstance(item, (int, float, numpy.integer, numpy.floating))
               for item in value_list)


def create_scatter_plot(
        data: pandas.DataFrame,
        x_column: str,
        y_column: str,
        x_label: str = None,
        y_label: str = None,
        title: str = 'Scatter plot'
):
    """
    Создает точечный график (scatter plot) на основе данных DataFrame.

    Args:
        data (pandas.DataFrame): DataFrame с данными для построения графика
        x_column (str): Название колонки для оси X
        y_column (str): Название колонки для оси Y
        x_label (str, optional): Подпись для оси X. Если не указана, используется название колонки
        y_label (str, optional): Подпись для оси Y. Если не указана, используется название колонки
        title (str, optional): Заголовок графика. По умолчанию 'Scatter plot'

    Raises:
        TypeError: Если передан не DataFrame или колонки содержат нечисловые данные
        ValueError: Если указанные колонки отсутствуют в DataFrame
    """
    if not isinstance(data, pandas.DataFrame):
        raise TypeError('Параметр data должен быть pandas.DataFrame')

    # Проверка наличия колонок
    if x_column not in data.columns:
        raise ValueError(f'Колонка {x_column} отсутствует в данных')

    if y_column not in data.columns:
        raise ValueError(f'Колонка {y_column} отсутствует в данных')

    # Проверка типов данных в колонках
    if not is_list_of_numbers(data[x_column]):
        raise TypeError(f'Колонка {x_column} должна содержать числовые данные')

    if not is_list_of_numbers(data[y_column]):
        raise TypeError(f'Колонка {y_column} должна содержать числовые данные')

    # Настройка и отображение графика
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(x_label if x_label else x_column)
    plt.ylabel(y_label if y_label else y_column)

    plt.scatter(data[x_column], data[y_column], alpha=0.7, edgecolors='w', s=100)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Пример использования
    data = {'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'points': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

    data_frame = pandas.DataFrame(data)

    create_scatter_plot(
        data_frame,
        'points',
        'time',
        title='Points scatter',
        x_label='Points scored',
        y_label='Time (minutes)'
    )