from typing import Union, List, Tuple, Optional
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# Реализация загрузки пользовательского изображения
# Скопировано из https://github.com/capibarco/CV-1-43/blob/main/hist.py
# Изменения: нет
def load_custom_image(file_path):
    """
    Загрузка пользовательского изображения
    """
    try:
        image = cv.imread(file_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {file_path}")
        return image
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return None

# Реализация подготовки изображения для создания гистограммы
# Скопировано из https://github.com/capibarco/CV-1-43/blob/main/hist.py
# Изменения: нет
def prepare_image(image):
    """
    Подготовка изображения для создания гистограммы с обработкой ошибок
    """
    try:

        # Конвертация в grayscale если нужно
        if image.ndim == 3:
            if image.shape[2] == 3:
                # Конвертируем BGR в grayscale
                image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                image_gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
            else:
                raise ValueError(f"Неожиданное количество каналов: {image.shape[2]}")
        elif image.ndim == 2:
            image_gray = image
        else:
            raise ValueError(f"Неожиданная размерность: {image.ndim}")

        # Нормализация в диапазон [0, 255]
        if image_gray.dtype != np.uint8:
            if np.max(image_gray) <= 1.0:  # Предполагаем диапазон [0, 1]
                image_uint8 = (image_gray * 255).astype(np.uint8)
            else:
                # Масштабирование к диапазону [0, 255]
                image_normalized = cv.normalize(image_gray, None, 0, 255, cv.NORM_MINMAX)
                image_uint8 = image_normalized.astype(np.uint8)
        else:
            image_uint8 = image_gray

        # Дополнительная проверка после преобразования
        if image_uint8.min() < 0 or image_uint8.max() > 255:
            raise ValueError("Изображение вышло за пределы диапазона [0, 255] после преобразования")

        return image_uint8

    except Exception as e:
        raise RuntimeError(f"Ошибка при подготовке изображения: {e}") from e


def correlation_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Сравнивает две гистограммы с помощью корреляции.

    Args:
        hist1: Первая гистограмма
        hist2: Вторая гистограмма

    Returns:
        float: Коэффициент корреляции от -1.0 до 1.0

    Raises:
        RuntimeError: При ошибках сравнения
    """
    # Корреляция (1 - полная похожесть, -1 - полная противоположность)
    try:
        correlation=cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
        return correlation
    except Exception as e:
        raise RuntimeError('ошибка при сравнение гистограмм')


def compare_histograms(histograms: List[np.ndarray]) -> List[float]:
    """
        Сравнивает попарно все гистограммы в списке.

        Args:
            histograms: Список гистограмм для сравнения

        Returns:
            List[float]: Список коэффициентов корреляции для всех пар

        Raises:
            ValueError: Если список содержит менее 2 гистограмм
        """

    if len(histograms)<2:
        raise ValueError('Для сравнения нужно минимум 2 изображения')

    print("Корреляция изображений:\n1.0 — идеальное совпадение.\n0.0 — отсутствие корреляции.\n-1.0 — идеальная обратная корреляция.")

    correlations_of_images=list()

    for ind in range(len(histograms)):
        for ind2 in range(ind+1,len(histograms)):
            correlation = correlation_histograms(histograms[ind], histograms[ind2])
            print(f'изображение {ind+1} и изображение {ind2+1} при корреляции возвращают {correlation}')
            correlations_of_images.append(correlation)

    return correlations_of_images

def create_plot_histogram(
    histograms: List[np.ndarray],
    average_hist: np.ndarray,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
        Создает график с гистограммами и средней гистограммой.

        Args:
            histograms: Список гистограмм для отображения
            average_hist: Средняя гистограмма
            save_path: Путь для сохранения графика (опционально)

        Returns:
            Tuple[plt.Figure, plt.Axes]: Объекты figure и axes matplotlib

        Raises:
            RuntimeError: При ошибках построения графика
        """

    try:
        fig, ax = plt.subplots(figsize=(15, 10))

        for ind,img in enumerate(histograms):
            ax.plot(img,linewidth=2, label=f'image {ind+1}',alpha=0.5)

        ax.plot(average_hist, color='red', linewidth=4, label='Средняя гистограмма')

        ax.set_title('Диаграммы яркости изображений')
        ax.set_xlabel('Значение яркости',fontsize=12)
        ax.set_ylabel('Нормализованная частота',fontsize=12)
        ax.legend()

        # Сохранение графика
        if save_path:
            try:
                fig.savefig(save_path, bbox_inches='tight', )
                print(f'График успешно сохранён в {save_path}')
            except Exception as e:
                print(f'Ошибка сохранения графика: {e}')

        plt.show()
        return fig,ax

    except Exception as e:
        plt.close('all')  # Закрыть все фигуры при ошибке
        raise RuntimeError(f"Ошибка при построении графиков: {e}") from e

def create_histograms_and_avg_histogram(custom_images: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
    """
        Создает гистограммы для списка изображений и вычисляет среднюю гистограмму.

        Args:
            custom_images: Список изображений

        Returns:
            Tuple[List[np.ndarray], np.ndarray]: Список гистограмм и средняя гистограмма
        """

    histograms=list()
    try:
        for image in custom_images:
            # подготовка изображения для создания гистограммы
            prepared_image = prepare_image(image)

            # создание гистограммы
            hist = cv.calcHist([prepared_image], [0], None, [256], [0, 256])

            # нормализация гистограммы для сравнения
            cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

            histograms.append(hist)
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        raise

    # вычисление средней гистограммы
    average_hist = np.mean(histograms, axis=0)

    return histograms, average_hist

def create_histograms_and_compare_images(
    images: Union[List[str], Tuple[str], np.ndarray, set],
    save_path: Optional[str] = None
) -> Tuple[List[np.ndarray], np.ndarray, List[float], plt.Figure, plt.Axes]:
    """
        Основная функция: загружает изображения, создает гистограммы, сравнивает их и строит графики.

        Args:
            images: Коллекция путей к изображениям
            save_path: Путь для сохранения графика (опционально)

        Returns:
            Tuple: Список гистограмм, средняя гистограмма, список корреляций, figure, axes

        Raises:
            TypeError: Если передан некорректный тип данных
            AttributeError: Если не удалось загрузить изображения или другие ошибки валидации
            ValueError: Если передана пустая коллекция
        """

    if isinstance(images, (list, tuple, np.ndarray, set)):
        custom_images=list()

        if len(images) == 0:
            raise ValueError("Список изображений пуст")

        for image in images:
            custom_images.append(load_custom_image(image))

        if all(img is not None for img in custom_images):

            histograms, average_histogram = create_histograms_and_avg_histogram(custom_images)

            fig, ax = create_plot_histogram(histograms, average_histogram,save_path)

            corellations = compare_histograms(histograms)

            return histograms, average_histogram, corellations, fig, ax
        else:
            raise AttributeError('Изображение не открылось')
    else:
        raise TypeError('некорректные входные данные')


if __name__ == '__main__':
    images = np.array(['images/forest.jpg', 'images/nature.jpg', 'images/cat.jpg'])

    hists, average_histogram, corellations, fig, ax = create_histograms_and_compare_images(images,"output.jpg")