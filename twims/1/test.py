import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


class MonteCarloPi:
    def __init__(self):
        self.points_inside = 0
        self.total_points = 0
        self.pi_estimates = []
        self.errors = []
        self.points_history = []

    def generate_points(self, num_points):
        """Генерация случайных точек в квадрате [-1, 1] x [-1, 1]"""
        x = np.random.uniform(-1, 1, num_points)
        y = np.random.uniform(-1, 1, num_points)
        return x, y

    def is_inside_circle(self, x, y):
        """Проверка, находится ли точка внутри окружности x² + y² ≤ 1"""
        return x ** 2 + y ** 2 <= 1

    def calculate_pi(self, points_inside, total_points):
        """Вычисление приближенного значения π"""
        if total_points == 0:
            return 0
        return 4 * points_inside / total_points

    def calculate_error(self, pi_estimate):
        """Вычисление абсолютной ошибки относительно настоящего π"""
        return abs(pi_estimate - np.pi)

    def run_simulation(self, total_points, batch_size=1000):
        """Запуск симуляции методом Монте-Карло"""
        self.points_inside = 0
        self.total_points = 0
        self.pi_estimates = []
        self.errors = []
        self.points_history = []

        print(f"Запуск симуляции для {total_points} точек...")

        # Генерируем точки партиями для экономии памяти
        batches = total_points // batch_size
        remaining = total_points % batch_size

        for i in tqdm(range(batches + 1)):
            if i < batches:
                num_points = batch_size
            else:
                num_points = remaining
                if num_points == 0:
                    break

            # Генерируем точки
            x, y = self.generate_points(num_points)

            # Проверяем каждую точку
            for j in range(num_points):
                self.total_points += 1
                if self.is_inside_circle(x[j], y[j]):
                    self.points_inside += 1

                # Записываем историю каждые 100 точек для экономии памяти
                if self.total_points % 100 == 0 or self.total_points == total_points:
                    pi_estimate = self.calculate_pi(self.points_inside, self.total_points)
                    error = self.calculate_error(pi_estimate)

                    self.pi_estimates.append(pi_estimate)
                    self.errors.append(error)
                    self.points_history.append(self.total_points)

        final_pi = self.calculate_pi(self.points_inside, self.total_points)
        final_error = self.calculate_error(final_pi)

        print(f"Окончательный результат: π ≈ {final_pi:.8f}")
        print(f"Абсолютная ошибка: {final_error:.8f}")
        print(f"Относительная ошибка: {(final_error / np.pi) * 100:.4f}%")
        print(f"Точек внутри окружности: {self.points_inside}/{self.total_points}")

        return final_pi, final_error

    def plot_results(self):
        """Построение графиков результатов"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # График 1: Приближенное значение π
        ax1.plot(self.points_history, self.pi_estimates, 'b-', alpha=0.7, label='Приближенное π')
        ax1.axhline(y=np.pi, color='r', linestyle='--', label='Истинное π')
        ax1.set_xscale('log')
        ax1.set_xlabel('Количество точек')
        ax1.set_ylabel('Значение π')
        ax1.set_title('Приближение числа π методом Монте-Карло')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Ошибка приближения
        ax2.plot(self.points_history, self.errors, 'r-', alpha=0.7)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Количество точек')
        ax2.set_ylabel('Абсолютная ошибка')
        ax2.set_title('Зависимость ошибки от количества точек')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        """Дополнительный график сходимости"""
        plt.figure(figsize=(10, 6))

        # Ошибка в логарифмическом масштабе
        plt.loglog(self.points_history, self.errors, 'ro-', alpha=0.7, markersize=2)

        # Теоретическая линия сходимости 1/sqrt(N)
        theoretical_error = 1 / np.sqrt(np.array(self.points_history))
        plt.loglog(self.points_history, theoretical_error, 'g--',
                   label='Теоретическая сходимость ~1/√N')

        plt.xlabel('Количество точек (N)')
        plt.ylabel('Абсолютная ошибка')
        plt.title('Сходимость метода Монте-Карло')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# Демонстрация работы
def main():
    # Создаем экземпляр класса
    mc_pi = MonteCarloPi()

    # Запускаем симуляцию
    total_points = 1000000  # 1 миллион точек
    start_time = time.time()

    final_pi, final_error = mc_pi.run_simulation(total_points)

    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.2f} секунд")

    # Строим графики
    mc_pi.plot_results()
    mc_pi.plot_convergence()


if __name__ == "__main__":
    main()