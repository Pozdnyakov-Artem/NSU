import random

sp=[random.randint(1,100) for i in range(10)]
print(*sp)
sp=list(map(lambda x: x%2,sp))
print(f'eval: {sp.count(0)} \n not eval: {sp.count(1)}')