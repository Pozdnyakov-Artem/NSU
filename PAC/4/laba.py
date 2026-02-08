import numpy as np
import random
import os

def read_mas(path):
    if os.path.exists(path) and os.path.isfile(path):
        try:
            file = open(path,'r')

            try:
                mas = np.array([i for i in file.readline().split()],dtype=float)

                return mas

            except:
                raise ValueError('В файле не числа')

        except Exception as e:
            raise "Ошибка открытия файла"
    else:
        raise f"Файл {path} не существует"

def fst_met(mas1,mas2,p = 0.5):
    p=1-p
    n=len(mas1)
    mask1 = np.random.random(n) < p
    ans=np.where(mask1,mas1,mas2)
    return ans

def snd_met(mas1,mas2,p = 0.5):
    n=len(mas1)
    k=int(p*n)
    unq_ind = random.sample(range(n),k)

    mas1[unq_ind] = mas2[unq_ind]
    return mas1

def tht_met(mas1,mas2,p = 0.5):
    n = len(mas1)
    random_mask = np.random.choice([0,1], p=[p, 1-p])
    random_mask = random_mask == 0
    conditions = [random_mask, ~random_mask]
    choices = [mas2, mas1]
    result = np.select(conditions, choices)
    return result

arr1 = read_mas(input())
arr2 = read_mas(input())
P = float(input())
if np.shape(arr1) == np.shape(arr2):
    print(fst_met(arr1, arr2,P))
    print(snd_met(arr1,arr2,P))
    print(tht_met(arr1,arr2,P))
else:
    print("матрицы разной длины")