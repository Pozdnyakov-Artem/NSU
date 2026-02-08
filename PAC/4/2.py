import numpy as np

h=2
w=2

mas=np.random.randint(255,size = (h,w,3))
new_mas=np.array([[[1,1,1],[2,3,2]],
         [[1,1,1],[2,2,2]]])
# print(mas)
print(new_mas.shape)
print()
print(np.unique(new_mas,axis=0))