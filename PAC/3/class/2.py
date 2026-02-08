st = " 1 2 3 4 odin"

dic={'1':'odin', '2':'dva'}
a=""
for i in st.split():
    if i in dic.keys():
        a+=dic[i]
    # for i,j in dic.items():
# ['1','2']
print(a)
# str = kk
s=st.split()

for n,i in enumerate(s):
    if i in dic.keys():
        s[n]=dic[i]
print(' '.join(s))