import random
import matplotlib.pyplot as plt
import math

rad = 5
n=55
count=0
dic={}
for i in range(1,n+1):
    x=random.uniform(-rad,rad)
    y=random.uniform(-rad,rad)
    if x**2+y**2<=rad**2:
        count+=1
    if i%10==0:
        dic[i]=(count/i *4)
plt.plot(dic.keys(),dic.values(),)
plt.xlabel('количество точек')
plt.ylabel('приближенное значение')
plt.axhline(math.pi,color='g',linestyle='--',label='значение pi')
plt.show()

print(count/n *4*rad)