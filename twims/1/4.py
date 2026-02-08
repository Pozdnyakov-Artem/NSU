import time
import matplotlib.pyplot as plt

def rand():
    a = 6364136223846793005
    b = 1442695040888963407
    M = 2**64
    seed=int(time.time_ns())
    while True:
        seed=((seed*a+b)%M)/M
        yield seed

rad = 1
n=1000
count=0
dic={}
for i in range(1,n):
    x=next(rand())
    y=next(rand())
    # print(x,y)
    if x**2+y**2<=rad**2:
        count+=1
    dic[i]=count/i *4
plt.plot(dic.keys(),dic.values())
plt.show()

print(count/n *4*rad)