import numpy as np

A = np.array([
    [3, 4, 2],
    [5, 2, 3],
    [4, 3, 2]
])

B = np.array([17, 23, 19])

if np.linalg.det(A) != 0:
    A_ob = np.linalg.inv(A)
    X = A_ob @ B
    print(X)
    print(A @ X)
else:
    print('бесконечное множество решений')