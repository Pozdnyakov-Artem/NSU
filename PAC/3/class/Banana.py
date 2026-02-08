from Food import Food
from Fruit import Fruit

class Banana(Food,Fruit):
    def __init__(self,ripe,count=1,max_count=16,color='yellow',saturation=10):
        super().__init__(ripe=ripe,max_count=max_count,saturation=saturation)
        self._color = color

    @property
    def color(self):
        return self.color