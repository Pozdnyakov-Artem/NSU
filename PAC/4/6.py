import numpy as np

A = np.matrix("1 0 1; 0 1 0; 1 0 1")

AAT = A.T @ A

m,n=A.shape

znach, V = np.linalg.eig(AAT)

ind = np.argsort(znach)[::-1]
V = V[:,ind]
znach = znach[ind]

S_values = np.sqrt(znach)
S = np.diag(S_values)

r = np.linalg.matrix_rank(A)

S_inv = np.diag(1 / S_values[:r])

U = np.zeros((m, m))
U[:, :r] = (A @ V[:, :r]) @ S_inv

print(U @ S @ V.T)