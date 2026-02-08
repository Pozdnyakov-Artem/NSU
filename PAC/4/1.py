import numpy as np

mas=np.array([1,2,3,4,5,6,7,8,1,2,3,4,5,1,2,3,1,2,1])
print(*mas)

uniq, count = np.unique(mas, return_counts=True)
s=uniq[np.argsort(count)]
s=np.repeat(s,np.sort(count))
print(*s)
