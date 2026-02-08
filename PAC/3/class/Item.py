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

    # @property
    # def color(self):
    #     return self.color
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

    def __eq__(self,item):
        return self.count == Item.count

    def __le__(self, num):
        """ Сравнение меньше или равно """
        return self.count <= num

    def __ge__(self, num):
        """ Сравнение больше или равно """
        return self.count >= num

    def __mul__(self, num):
        """ Умножение на число """
        return min(max(self.count * num,0),self._max_count)

    def __add__(self, num):
        """ Сложение с числом """
        return min(max(self.count + num,0),self._max_count)

    def __iadd__(self, num):
        """ Сложение с числами """
        return min(max(self.count + num, self._max_count),self._max_count)

    def __imul__(self, num):
        """ Умножение с числами """
        return min(max(self.count * num, self._max_count),self._max_count)

    def __isub__(self, num):
        """ Вычитание """
        return min(max(self.count - num, 0),self._max_count)

    def __len__(self):
        """ Получение длины объекта """
        return self.count