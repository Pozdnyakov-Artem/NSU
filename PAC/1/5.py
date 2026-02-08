import math

n=int(input())
ans=[]

for i in range(2,n+1):
    flag=0
    for j in range(2,int(math.sqrt(i))+1):
        if i%j==0:
            #flag=1
            break
    else:# flag==0:
        ans.append(i)
print(*ans)