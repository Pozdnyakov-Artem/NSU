from Food import Food

class Chocolate(Food):
    def __init__(self,saturation,count=1,max_count=32,_type='Black'):
        super().__init__(saturation=saturation,count=count,max_count=max_count)
        self._type = _type

    def __str__(self):
        return f'{self._type} chocolates: {self.count}'
