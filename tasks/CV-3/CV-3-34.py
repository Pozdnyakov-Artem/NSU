import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from collections import Counter
import cv2
from typing import Tuple, Optional, Dict, List, Any
import os


class ProcessingMode:
    """Класс для определения режимов обработки изображений."""
    LFW_PEOPLE = "fetch_lfw_people"
    FACE_DETECTION = "find_face"


def estimate_age_by_size(face_array: List[int], processing_mode: str) -> Tuple[float, float]:
    """
    Приближенная оценка возраста на основе размера лица.

    Использует площадь лица как прокси для возраста. Для LFW_PEOPLE вычисляется
    сумма интенсивностей пикселей, для FACE_DETECTION - площадь bounding box.

    Args:
        face_array: Массив данных лица (пиксели или размеры bounding box)
        processing_mode: Режим обработки (LFW_PEOPLE или FACE_DETECTION)

    Returns:
        Tuple[float, float]: Оцененный возраст и размер лица

    Raises:
        ValueError: Если передан некорректный processing_mode
    """

    if processing_mode == ProcessingMode.LFW_PEOPLE:
        # Нормализуем значения пикселей
        normalized_face = face_array / 255.0

        # Вычисляем "размер" лица как сумму интенсивностей пикселей
        face_size = np.sum(normalized_face)
    elif processing_mode == ProcessingMode.FACE_DETECTION:
        face_size = face_array//10000
    else:
        raise TypeError("Некорректная операция")

    # Эмпирическая формула для оценки возраста
    # Предполагаем, что большие лица соответствуют взрослым, маленькие - детям
    estimated_age = max(10, min(80, face_size * 10))

    return estimated_age, face_size


def age_to_category(age: float) -> str:
    """
    Группировка возраста в категории.

    Args:
        age: Возраст в годах

    Returns:
        str: Возрастная категория ('ребёнок', 'взрослый', 'пожилой')
    """
    if age < 18:
        return 'ребёнок'
    elif age < 50:
        return 'взрослый'
    else:
        return 'пожилой'


def show_stats(age_categories: List[str], estimated_ages: List[float],
               category_counts: Dict[str, int]) -> None:
    """
    Вывод статистики по возрастным категориям.

    Args:
        age_categories: Список возрастных категорий для каждого лица
        estimated_ages: Список оцененных возрастов
        category_counts: Счетчик категорий

    Raises:
        ValueError: Если списки имеют разную длину
    """

    if len(age_categories) != len(estimated_ages):
        raise ValueError("Списки age_categories и estimated_ages должны иметь одинаковую длину")

    if not age_categories:
        print("Нет данных для отображения статистики")
        return

    print("\nРаспределение по возрастным категориям:")
    for category, count in category_counts.items():
        percentage = (count / len(age_categories)) * 100
        print(f"{category}: {count} лиц ({percentage:.1f}%)")

    # Статистика по категориям
    print("\nСтатистика по возрастным категориям:")
    for category in ['ребёнок', 'взрослый', 'пожилой']:
        indices = [i for i, cat in enumerate(age_categories) if cat == category]
        if indices:
            ages_in_category = [estimated_ages[i] for i in indices]
            print(f"\n{category}:")
            print(f"  Количество: {len(indices)}")
            print(f"  Средний возраст: {np.mean(ages_in_category):.1f} лет")
            print(f"  Мин. возраст: {np.min(ages_in_category):.1f} лет")
            print(f"  Макс. возраст: {np.max(ages_in_category):.1f} лет")


def create_bar_plot(category_counts: Dict[str, int], estimated_ages: List[float],
                    save_path: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Создание бар-плота и гистограммы распределения возрастов.

    Args:
        category_counts: Счетчик возрастных категорий
        estimated_ages: Список оцененных возрастов
        save_path: Путь для сохранения графика (опционально)

    Returns:
        Tuple[plt.Figure, np.ndarray]: Figure и axes объекты matplotlib

    Raises:
        RuntimeError: Если произошла ошибка при построении графиков
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        ax1 = axes[0]
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = ['lightblue', 'lightgreen', 'salmon']

        ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Распределение по возрастным категориям')
        ax1.set_xlabel('Возрастная категория')
        ax1.set_ylabel('Количество лиц')

        for i, count in enumerate(counts):
            ax1.text(i, count + 0.5, str(count), ha='center', va='bottom')

        ax2 = axes[1]
        if estimated_ages:
            ax2.hist(estimated_ages, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax2.set_title('Распределение оцененных возрастов')
            ax2.set_xlabel('Оцененный возраст (лет)')
            ax2.set_ylabel('Количество')
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Распределение оцененных возрастов')
            ax2.set_xlabel('Оцененный возраст (лет)')
            ax2.set_ylabel('Количество')


        plt.tight_layout()

        if save_path:
            try:
                fig.savefig(save_path, bbox_inches='tight')
                print(f'График успешно сохранён в {save_path}')
            except Exception as e:
                print(f'Ошибка сохранения графика: {e}')

        plt.show()

        return fig, axes

    except Exception as e:
        plt.close('all')
        raise RuntimeError(f"Ошибка при построении графиков: {e}") from e


def work_with_fetch_lfw_people(min_face_per_person: int = 20,
                               resize: float = 0.4,
                               save_path_to_bar: Optional[str] = None) -> Tuple[List[float], List[float], plt.Figure, np.ndarray]:
    """
    Анализ лиц из датасета LFW People.

    Args:
        min_face_per_person: Минимальное количество лиц на человека
        resize: Коэффициент изменения размера изображений
        save_path_to_bar: Путь для сохранения графиков

    Returns:
        Tuple: estimated_ages, face_sizes, figure, axes

    Raises:
        Exception: Если произошла ошибка при загрузке датасета
    """
    # Загрузка датасета
    try:
        # Загрузка датасета
        lfw_people = fetch_lfw_people(min_faces_per_person=min_face_per_person,
                                      resize=resize, color=False)
        X = lfw_people.data
        y = lfw_people.target
        target_names = lfw_people.target_names

        print(f"Размер датасета: {X.shape}")
        print(f"Количество людей: {len(target_names)}")
        print(f"Количество изображений: {len(y)}")

        # Проверка наличия данных
        if len(X) == 0:
            raise ValueError("Загруженный датасет не содержит изображений")

        # Применяем функцию ко всем лицам
        estimated_ages = []
        face_sizes = []

        for i in range(len(X)):
            age, size = estimate_age_by_size(X[i], ProcessingMode.LFW_PEOPLE)
            estimated_ages.append(age)
            face_sizes.append(size)

        age_categories = [age_to_category(age) for age in estimated_ages]
        category_counts = Counter(age_categories)

        show_stats(age_categories, estimated_ages, category_counts)



        fig, ax = create_bar_plot(category_counts, estimated_ages, save_path_to_bar)

        return estimated_ages, face_sizes, fig, ax

    except Exception as e:
        print(f"Ошибка при работе с датасетом LFW People: {e}")
        raise


def create_borders_and_put_text(img: np.ndarray, x: int, y: int, w: int, h: int,
                                age_category: str, face_size: float) -> None:
    """
    Отрисовка bounding box и текста на изображении.
    """
    # Отрисовка прямоугольника
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    category_eng = {
        'ребёнок': 'Child',
        'взрослый': 'Adult',
        'пожилой': 'Senior'
    }.get(age_category, age_category)

    cv2.putText(img, f"Face: {category_eng}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, f"Size: {face_size:.1f}", (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def find_and_analyze_faces(image_path: str, save_path_to_img: Optional[str] = None,
                           save_path_to_bar: Optional[str] = None) -> Tuple[
    List[Any], List[str], plt.Figure, np.ndarray]:
    """
    Найти лица на изображении и проанализировать их.

    Args:
        image_path: Путь к входному изображению
        save_path_to_img: Путь для сохранения изображения с разметкой
        save_path_to_bar: Путь для сохранения графиков

    Returns:
        Tuple: faces, age_categories, figure, axes

    Raises:
        FileNotFoundError: Если изображение не найдено или классификатор не загружен
        Exception: Если произошла ошибка при обработке изображения
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")

    # Детекция лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        raise FileNotFoundError("Ошибка загрузки классификатора!")

    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        print(f"Найдено лиц: {len(faces)}")

        # Анализ каждого лица
        age_categories = []
        estimated_ages = []

        for i, (x, y, w, h) in enumerate(faces):

            # Оценка "возраста" по размеру (упрощенная)
            face_size = w * h  # площадь bounding box

            estimate_age, _ = estimate_age_by_size(face_size, "find_face")

            estimated_ages.append(estimate_age)

            age_category = age_to_category(estimate_age)

            age_categories.append(age_category)

            create_borders_and_put_text(img,x,y,w,h,age_category, face_size)



        category_counts = Counter(age_categories)

        # Показ результата
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Найдено лиц: {len(faces)}')
        plt.axis('off')

        plt.show()

        if save_path_to_img:
            try:
                cv2.imwrite(save_path_to_img, img)
                print(f'График успешно сохранён в {save_path_to_img}')
            except Exception as e:
                print(f'Ошибка сохранения графика: {e}')

        fig, ax = create_bar_plot(category_counts, estimated_ages,save_path_to_bar)

        show_stats(age_categories, estimated_ages, category_counts)

        return faces, age_categories, fig, ax
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        raise


def main():
    # Пример нахождение лиц на произвольном фото
    print("=== Анализ лиц на произвольном изображении ===")
    faces, categories, fig, ax = find_and_analyze_faces('images/faces.jpg',"face_output.jpg","face_output2.jpg")

    # Пример работы с sklearn.datasets.fetch_lfw_people()
    print("\n=== Анализ датасета LFW People ===")
    work_with_fetch_lfw_people(20, 0.4,"output.jpg")

if __name__ == "__main__":
    main()
