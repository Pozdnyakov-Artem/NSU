import random

num=random.randint(100,999)
print(num)
print(sum([int(i) for i in str(num)]))
