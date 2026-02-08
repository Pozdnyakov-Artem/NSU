from Item import Item
from Food import Food
from Fruit import Fruit
from Apple import Apple
from Chocolate import Chocolate
from Chicken import Chicken
from Inventory import Inventory


apple = Apple(False, color='green')
invent = Inventory(10)
chocolate = Chocolate(1,2,15)
chicken = Chicken(1,5,6)

invent.add_item(apple,1)
invent.add_item(apple,1)
invent.add_item(chocolate,2)
invent.take_item(2,1)
invent.add_item(chicken,3)
invent.show_items()
invent.take_item(2,1)
invent.show_items()
# apple + 3
print(apple)
# print(apple.count)
# print(apple.color)
# print(apple.eatable)
# print(len(apple))
# print(apple + 3)
# print(apple * 3)
# print(apple < 3)