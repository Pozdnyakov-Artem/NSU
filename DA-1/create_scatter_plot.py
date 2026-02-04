import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Any, Tuple, Optional, Iterable, Union, List, Dict
import math


def contains_nan_or_inf(value: Any) -> bool:
    """
    Проверяет наличие NaN или inf значений в данных.

    Parameters:
    -----------
    value : Any
        Данные для проверки. Может быть числом, массивом, списком, etc.

    Returns:
    --------
    bool
        True если данные содержат NaN или inf значения, иначе False.

    Examples:
    ---------
    contains_nan_or_inf([1, 2, np.nan])
    True
    contains_nan_or_inf([1, 2, 3])
    False
    contains_nan_or_inf(np.array([1, np.inf, 3]))
    True
    """
    if hasattr(value, 'dtype') and hasattr(value, '__array__'):
        arr = np.asarray(value)
        return np.isnan(arr).any() or np.isinf(arr).any()
    else:
        try:
            for item in value:
                if isinstance(item, (int, float, np.number)):
                    if np.isnan(item) or np.isinf(item):
                        return True
        except (TypeError, ValueError):
            pass
    return False


def is_list_of_numbers(value: Any) -> bool:
    """
    Проверяет, является ли значение итерируемым объектом, содержащим только числа.
    Выполняет проверку на наличие NaN/inf значений и корректные числовые типы.

    Parameters:
    -----------
    value : Any
        Значение для проверки

    Returns:
    --------
    bool
        True если значение является итерируемым объектом, содержащим только числа
        без NaN/inf значений, иначе False.

    Notes:
    ------
    - Для массивов размером <= 1000 проверяются все элементы
    - Для больших массивов выполняется выборочная проверка
    - Автоматически проверяет на наличие NaN/inf значений

    Examples:
    ---------
    is_list_of_numbers([1, 2, 3])
    True
    is_list_of_numbers([1, 2, 'text'])
    False
    is_list_of_numbers([1, 2, np.nan])
    False
    """
    if not hasattr(value, '__iter__') or isinstance(value, str):
        return False

    try:
        if len(value) == 0:
            return False
    except TypeError:
        return False

    # Проверка на NaN/inf значения
    if contains_nan_or_inf(value):
        return False

    # Для pandas Series и numpy arrays используем встроенные методы
    if hasattr(value, 'dtype'):
        return np.issubdtype(value.dtype, np.number)

    # Определяем стратегию проверки в зависимости от размера данных
    try:
        data_length = len(value) if hasattr(value, '__len__') else None

        # Для небольших наборов данных проверяем ВСЕ элементы
        if data_length is not None and data_length <= 1000:
            for item in value:
                if not isinstance(item, (int, float, np.integer, np.floating)):
                    return False
            return True
        else:
            # Для больших наборов или неизвестного размера - выборочная проверка
            iterator = iter(value)
            check_count = min(100, data_length) if data_length else 100

            for _ in range(check_count):
                try:
                    item = next(iterator)
                    if not isinstance(item, (int, float, np.integer, np.floating)):
                        return False
                except StopIteration:
                    break

            # Дополнительная проверка последнего элемента для последовательностей
            if data_length and data_length > 1:
                try:
                    if hasattr(value, '__getitem__'):
                        last_item = value[-1] if data_length > 0 else None
                        if last_item is not None and not isinstance(last_item, (int, float, np.integer, np.floating)):
                            return False
                except (IndexError, TypeError):
                    pass

            return True

    except (StopIteration, TypeError, ValueError):
        # Резервная проверка первого элемента
        try:
            first_item = next(iter(value))
            return isinstance(first_item, (int, float, np.integer, np.floating))
        except (StopIteration, TypeError, ValueError):
            return False


def create_scatter_plot(
        data: Union[pd.DataFrame, List, Tuple, np.ndarray, Dict[str, Iterable]],
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        x_data: Optional[Iterable] = None,
        y_data: Optional[Iterable] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        title: str = 'Scatter Plot',
        show_plot: bool = True,
        save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Создает точечный график (scatter plot) на основе различных типов данных.

    Parameters:
    -----------
    data : Union[pd.DataFrame, List, Tuple, np.ndarray, Dict[str, Iterable]]
        Источник данных для построения графика. Может быть:
        - pandas.DataFrame: использует x_column и y_column для выбора данных
        - 2D list/tuple/array: использует первый столбец для X, второй для Y
        - dict: использует ключи 'x' и 'y' или первые два ключа

    x_column : str, optional
        Название колонки для оси X (только для DataFrame)

    y_column : str, optional
        Название колонки для оси Y (только для DataFrame)

    x_data : Iterable, optional
        Прямая передача данных для оси X (альтернатива data параметру)

    y_data : Iterable, optional
        Прямая передача данных для оси Y (альтернатива data параметру)

    x_label : str, optional
        Подпись для оси X. Если не указана, используется имя колонки или 'X'

    y_label : str, optional
        Подпись для оси Y. Если не указана, используется имя колонки или 'Y'

    title : str, optional
        Заголовок графика. По умолчанию 'Scatter Plot'

    show_plot : bool, optional
        Показывать график после построения. По умолчанию True

    save_path : str, optional
        Путь для сохранения графика в файл. Если None, график не сохраняется

    Returns:
    --------
    Tuple[plt.Figure, plt.Axes]
        Кортеж содержащий объекты Figure и Axes для дальнейшей настройки графика

    Raises:
    -------
    TypeError
        - Если одновременно переданы data и (x_data, y_data)
        - Если передан неподдерживаемый тип данных
        - Если данные содержат нечисловые значения или NaN/inf
    ValueError
        - Если для DataFrame не указаны x_column и y_column
        - Если указанные колонки отсутствуют в DataFrame
        - Если данные для X и Y имеют разную длину
        - Если передан 1D массив вместо 2D

    Examples:
    ---------
    # Использование с DataFrame
     df = pd.DataFrame({'time': [1,2,3], 'points': [10,20,30]})
     fig, ax = create_scatter_plot(df, 'time', 'points', title='Example')

    # Использование со списками
     x_data = [1, 2, 3, 4, 5]
     y_data = [2, 4, 6, 8, 10]
     fig, ax = create_scatter_plot(None, x_data=x_data, y_data=y_data)

    # Использование с словарем
     data_dict = {'x': [1,2,3], 'y': [4,5,6]}
     fig, ax = create_scatter_plot(data_dict)
    """
    x_values = None
    y_values = None

    if (x_data is not None and data is not None) or (y_data is not None and data is not None):
        raise TypeError("Слишком много значений")
    # Обработка различных типов входных данных
    if isinstance(data, pd.DataFrame):
        if x_column is None or y_column is None:
            raise ValueError("Для DataFrame необходимо указать x_column и y_column")

        if x_column not in data.columns:
            raise ValueError(f'Колонка "{x_column}" отсутствует в данных. Доступные колонки: {list(data.columns)}')
        if y_column not in data.columns:
            raise ValueError(f'Колонка "{y_column}" отсутствует в данных. Доступные колонки: {list(data.columns)}')

        x_values = data[x_column]
        y_values = data[y_column]

        # Устанавливаем подписи по умолчанию из названий колонок
        if x_label is None:
            x_label = x_column
        if y_label is None:
            y_label = y_column

    elif x_data is not None and y_data is not None:
        # Прямая передача данных
        x_values = x_data
        y_values = y_data

        # Устанавливаем подписи по умолчанию
        if x_label is None:
            x_label = "X"
        if y_label is None:
            y_label = "Y"

    elif isinstance(data, (list, tuple, np.ndarray)):
        # Если передан 2D массив/список
        data_array = np.array(data)
        if data_array.ndim == 2 and data_array.shape[1] >= 2:
            x_values = data_array[:, 0]
            y_values = data_array[:, 1]

            if x_label is None:
                x_label = "X"
            if y_label is None:
                y_label = "Y"
        else:
            raise ValueError("Для list/tuple/array нужен 2D массив с минимум 2 колонками")

    elif isinstance(data, dict):
        # Если передан словарь
        if 'x' in data and 'y' in data:
            x_values = data['x']
            y_values = data['y']
        elif len(data) >= 2:
            keys = list(data.keys())
            x_values = data[keys[0]]
            y_values = data[keys[1]]
        else:
            raise ValueError("Словарь должен содержать как минимум 2 ключа с данными")

        if x_label is None:
            x_label = "X"
        if y_label is None:
            y_label = "Y"
    else:
        raise TypeError(f"Неподдерживаемый тип данных: {type(data)}. "
                        f"Поддерживаемые типы: DataFrame, list, tuple, ndarray, dict")

    # Проверка наличия данных
    if x_values is None or y_values is None:
        raise ValueError("Не удалось извлечь данные для построения графика")

    # Проверка одинаковой длины данных
    if len(x_values) != len(y_values):
        raise ValueError(f'Данные для X и Y должны быть одинаковой длины: X={len(x_values)}, Y={len(y_values)}')

        # Проверка типов данных с полной проверкой всех элементов
    if not is_list_of_numbers(x_values):
        if contains_nan_or_inf(x_values):
            raise TypeError(f'Данные для оси X содержат NaN или inf значения')

    if not is_list_of_numbers(y_values):
        if contains_nan_or_inf(y_values):
            raise TypeError(f'Данные для оси Y содержат NaN или inf значения')

    # Создание графика
    fig, ax = plt.subplots()
    ax.scatter(x_values, y_values)

    # Настройка внешнего вида
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Отображение графика
    if show_plot:
        plt.show()

    # Сохранение графика
    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight',)
            print(f'График успешно сохранён в {save_path}')
        except Exception as e:
            print(f'Ошибка сохранения графика: {e}')

    return fig, ax


if __name__ == '__main__':
    # Примеры использования

    # 1. Пример с DataFrame
    data = {'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'points': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

    data_frame = pd.DataFrame(data)

    fig1, ax1 = create_scatter_plot(
        data_frame,
        x_column='points',
        y_column='time',
        title='Points vs Time',
        x_label='Points',
        y_label='Time',
        show_plot=False,
        save_path='scatter_plot.png'
    )

    # 2. Пример со списками
    x_data = [1, 2, 3, 4, 5]
    y_data = [2, 4, 6, 8, 10]

    fig2, ax2 = create_scatter_plot(
        None,
        x_data=x_data,
        y_data=y_data,
        title='Linear Relationship',
        show_plot=True
    )

    # 3. Пример с словарем
    data_dict = {'x_values': [1, 2, 3, 4], 'y_values': [1, 4, 9, 9]}

    fig3, ax3 = create_scatter_plot(
        data_dict,
        title='Quadratic Relationship',
        show_plot=False
    )

    # 4 Пример
    fig4, ax4 = create_scatter_plot(
        data_dict,
        x_data=x_data,
        y_data=y_data,
        title='Quadratic Relationship',
        show_plot=True
    )