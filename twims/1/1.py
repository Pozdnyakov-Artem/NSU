import random

count=0
n=10

mas=[random.randint(1,2) for i in range(n)]

print(*mas)
print(float(mas.count(1)/n))