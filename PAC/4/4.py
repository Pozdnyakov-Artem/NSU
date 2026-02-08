import numpy as np

def valid(mat):

    fst = mas[:,1] + mas[:,2]>mas[:,0]
    snd = mas[:,2] + mas[:,0]>mas[:,1]
    tht = mas[:,1] + mas[:,0]>mas[:,2]

    res_bool =  fst * snd * tht

    return mat[res_bool]

n=5
mas = np.random.randint(10,size = (n,3))
print(*mas)
print()
res=valid(mas)

print(*res)
