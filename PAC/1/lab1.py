import random
n=int(input())

arr=[random.randint(0,1000) for i in range(n)]
print(*arr)

for i in range(n):
    swap=False
    for j in range(n-i-1):
        if arr[j]>arr[j+1]:
            arr[j],arr[j+1]=arr[j+1],arr[j]
            swap=True
    if not swap:
        break
print(arr)