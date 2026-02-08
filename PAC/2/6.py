from linecache import getline

file=open("input.txt","r")
mas=[]

for i in file:
    mas.append([i.strip(' ,./?\\"\'*&(){}[]%^$#@!') for i in i.split()])

count=0
for i in mas:
    for j in i:
        count+=len(j)

print(f'rows: {len(mas)}\nwords: {sum(map(len,mas))}\nletters: {count}')
# print(list(lambda x:len(x),mas))