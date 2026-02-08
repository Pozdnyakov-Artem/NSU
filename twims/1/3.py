import time

a = 6364136223846793005
b = 1442695040888963407
M = 2**64
m=[int(time.time_ns())]
n=10

for i in range(1,n):
    m.append(((m[i-1]*a+b) %M))

print(*m)