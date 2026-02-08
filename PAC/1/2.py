import random

num=random.randint(100000,999999)
print(num,sum([int(i) for i in str(num)]),sep='\n')