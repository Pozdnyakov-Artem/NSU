# Создания scatter plots (точечных графиков) с помощью библеотек Pandas и Matplotlib.
## Установка библиотек для создания графиков
`pip install matplotlib pandas`

## Параметры функции `create_scatter_plot`
| Параметр | Тип            | Обязательный | Описание | По умолчанию |
|:---------|:---------------|:-------------|:---------|:-------------|
| data     |pandas.DataFrame|✔             |DataFrame с данными для построения|-|
|x_column  |str             |✔             |параметр для оси x                |-|
|y_column  |str             |✔             |параметр для оси y                |-|
|x_label   |str             |-             |название оси x                    |название колонки x|
|y_label   |str             |-             |название оси y                    |название колонки y|
|title     |str             |-             |название графика                  |Scatter plot|

## Пример использования основной функции:
```python
from scatter_plot_utils import create_scatter_plot
import pandas

# Создание данных
data = pandas.DataFrame({
    'x_data': [1, 2, 3, 4, 5],
    'y_data': [10, 20, 30, 40, 50]
})

# Создание scatter plot
create_scatter_plot(
    data=data,
    x_column='x_data',
    y_column='y_data',
    title='Мой график',
    x_label='Ось X',
    y_label='Ось Y'
)
```
## Requirements
matplotlib==3.7.1
pandas==2.0.3

## Обработка ошибок
### Функция включает проверки для предотвращения ошибок:

* TypeError: если передан не pandas DataFrame

* ValueError: если указанные колонки отсутствуют в данных или переданы не числовые функции
