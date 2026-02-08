import os
dir=input()
# dir="C:\\2 sem\\PAC"
print(dir.replace('\\','/'))
if os.path.isdir(dir.replace('\\','/')):
    for i in os.walk(dir):
        print(i)
else:
    print("Нет такой папки")