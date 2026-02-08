class Item:
    def __init__(self, count=3, max_count=16):
        self._count = count
        self._max_count = 16

    def update_count(self, val):
        if val <= self._max_count:
            self._count = val
            return True
        else:
            return False

    # Свойство объекта. Не принимает параметров кроме self, вызывается без круглых скобок
    # Определяется с помощью декоратора property
    @property
    def count(self):
        return self._count

    # Ещё один способ изменить атрибут класса
    @count.setter
    def count(self, val):
        self._count = val
        if val <= self._max_count:
            self._counts = val
        else:
            pass

    @staticmethod
    def static():
        print('I am function')

    @classmethod
    def my_name(cls):
        return cls.__name__

    def __lt__(self, num):
        """ Сравнение меньше """
        return self.count < num

    def __gt__(self, num):
        """ Сравнение больше """
        return self.count > num

    def __eq__(self, num):
        """ Сравнение равно """
        return self.count == num

    def __le__(self, num):
        """ Сравнение меньше или равно """
        return self.count <= num

    def __ge__(self, num):
        """ Сравнение больше или равно """
        return self.count >= num

    def __mul__(self, num):
        """ Умножение на число """
        return self.count * num

    def __add__(self, num):
        """ Сложение с числом """
        return self.count + num

    def __iadd__(self, num):
        """ Сложение с числами """
        return max(self.count + num, self._max_count)

    def __imul__(self, num):
        """ Умножение с числами """
        return max(self.count * num, self._max_count)

    def __isub__(self, num):
        """ Вычитание """
        return max(self.count - num, 0)

    def __len__(self):
        """ Получение длины объекта """
        return self.count


class Fruit(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe


class Food(Item):
    def __init__(self, saturation, **kwargs):
        super().__init__(**kwargs)
        self._saturation = saturation

    @property
    def eatable(self):
        return self._saturation > 0


class Apple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    def color(self):
        return self._color

    @property
    def eatable(self):
        return super().eatable and self._ripe

    def __call__(self):
        """ Вызов как функции """
        if self.eatable:
            new_count = max(self.count - 1, 0)
            self.update_count(new_count)

    def __str__(self):
        """ Вызов как строки """
        return f'Stack of {self.count} {self.color} apples'

    # def __len__(self):
    #     """ Получение длины объекта """
    #     return self.count
    #
    # def __add__(self, num):
    #     """ Сложение с числом """
    #     return self.count + num
    #
    # def __mul__(self, num):
    #     """ Умножение на число """
    #     return self.count * num
    #
    # def __lt__(self, num):
    #     """ Сравнение меньше """
    #     return self.count < num



apple = Apple(False, color='green')
print(apple.count)
print(apple.color)
print(apple.eatable)
print(len(apple))
print(apple + 3)
print(apple * 3)
print(apple < 3)



class Chocolate(Food):
    def __init__(self, ripe, count=1, max_count=50, color='black', saturation=10):
        super().__init__(saturation=saturation, count=count, max_count=max_count)
        self.color = color

class Chicken(Food):
    def __init__(self, ripe, count=1, max_count=50, color='pink', saturation=10):
        super().__init__(saturation=saturation, count=count, max_count=max_count)
        self.color = color