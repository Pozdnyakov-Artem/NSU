from Food import Food
from Fruit import Fruit


class Apple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

    def __str__(self):
        """ Вызов как строки """
        return f'Stack of {self.count} {self.color} apples'

    # @property
    # def eatable(self):
    #     return super().eatable and self._ripe

    # def __call__(self,count:int=1):
    #     """ Вызов как функции """
    #     if count <=self.count:
    #         if self.eatable:
    #             new_count = max(self.count - count, 0)
    #             self.update_count(new_count)
    #     else:
    #         raise ValueError('Нет столько')

    # def __len__(self):
    #     """ Получение длины объекта """
    #     return self.count

    # def __add__(self, num):
    #     """ Сложение с числом """
    #     return self.count + num

    # def __mul__(self, num):
    #     """ Умножение на число """
    #     return self.count * num

    # def __lt__(self, num):
    #     """ Сравнение меньше """
    #     return self.count < num