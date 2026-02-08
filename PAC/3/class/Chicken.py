from Food import Food

class Chicken(Food):
    def __init__(self,saturation,count=1,max_count=50):
        super().__init__(count=count,max_count=max_count,saturation=saturation)

    def __str__(self):
        return f'Chicken(: {self.count})'