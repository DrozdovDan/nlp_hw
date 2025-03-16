import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import scipy.stats as ss
from scipy.optimize import root_scalar, NonlinearConstraint
from scipy.special import softmax

n, m = map(int, input().split())

a = np.zeros(shape=(n, n))
b = np.zeros(shape=(m, n))
for i in range(n):
  t = list(map(int, input().split()))
  for j in range(n):
    a[i, j] = t[j]

for i in range(m):
  t = list(map(int, input().split()))
  for j in range(n):
    b[i, j] = t[j]

def forward(x):
    l = softmax(b @ np.maximum(a @ x.reshape(-1, 1), 0)).flatten()
    l.sort()
    return abs(l[-1] - l[-2])

x = np.ones(shape=(n, ))

def sum_abs(x):
    return np.sum(np.abs(x))

y = root_scalar(forward, x0=x, constraints=[NonlinearConstraint(sum_abs, 1e-8, np.inf)])

if y.success:
    print('YES')
    print(*y.x)
else:
    print('NO')