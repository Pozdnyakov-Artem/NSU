

# f = open(r'titanic_with_labels_fixed.csv',"r")
# f=open("try.txt","r")
mas=[]
with open('try.txt', 'r', encoding='utf-8') as f:
    for i in f:
        print(i)
        # if i=="\n":
        #     continue
        # if "РќРµ,СѓРєР°Р·Р°РЅ" in i:
        #     print(i)
        mas.append(i.replace("не,указан","не указан"))
# f.close()
# f=open(r'titanic_with_labels_fixed.csv',"w")
# f=open("try.txt","w")
with open('try.txt', 'w', encoding='utf-8') as f:
    for i in mas:
        # f.write(','.join(i)+'\n')
        f.write(i)
# f.close()
# print(mas)