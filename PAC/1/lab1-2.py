n=int(input())
ans=[]
for i in range(n):
    row=[0]*(i+1)
    row[0],row[-1]=1,1
    for j in range(1,i):
        row[j]=ans[i-1][j]+ans[i-1][j-1]
    ans.append(row)

for i in range(n):
    print(' '.join(map(str,ans[i])).center(n*2))
