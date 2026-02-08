from Item import Item


class Food(Item):
    def __init__(self, saturation, **kwargs):
        super().__init__(**kwargs)
        self._saturation = saturation

    @property
    def eatable(self):
        return self._saturation > 0

    def __call__(self,count:int=1):
        """ Вызов как функции """
        if count <=self.count:
            if self.eatable:
                new_count = max(self.count - count, 0)
                self.update_count(new_count)
        else:
            raise ValueError('Нет столько')