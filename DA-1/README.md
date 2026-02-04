# Создания scatter plots (точечных графиков) с помощью библеотек Pandas и Matplotlib.
## Установка библиотек для создания графиков
`pip install matplotlib pandas`

## Параметры функции `create_scatter_plot`
| Параметр | Тип            | Обязательный | Описание | По умолчанию |
|:---------|:---------------|:-------------|:---------|:-------------|
| data     |Union[pd.DataFrame, List, Tuple, np.ndarray, Dict[str, Iterable]]|✔             |данные для построения|-|
|x_column  |str             |-             |параметр для оси x                |None|
|y_column  |str             |-             |параметр для оси y                |None|
|x_data    |Iterable        |-             |данные для построения оси x             |None|
|y_data    |Iterable        |-             |данные для построения оси y             |None|
|x_label   |str             |-             |название оси x                    |None|
|y_label   |str             |-             |название оси y                    |None|
|title     |str             |-             |название графика                  |Scatter plot|
|show_plot |bool            |-             |отображение графика               |True|
|save_path |str             |-             |путь для сохранения графика       |None

## Пример использования основной функции:
```python
from scatter_plot_utils import create_scatter_plot
import pandas

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
```
## Requirements
matplotlib==3.7.1
pandas==2.0.3

## Обработка ошибок
### Функция включает проверки для предотвращения ошибок:

* TypeError
    - Если одновременно переданы data и (x_data, y_data)
    - Если передан неподдерживаемый тип данных
    - Если данные содержат нечисловые значения или NaN/inf
* ValueError
    - Если для DataFrame не указаны x_column и y_column
    - Если указанные колонки отсутствуют в DataFrame
    - Если данные для X и Y имеют разную длину
    - Если передан 1D массив вместо 2D
