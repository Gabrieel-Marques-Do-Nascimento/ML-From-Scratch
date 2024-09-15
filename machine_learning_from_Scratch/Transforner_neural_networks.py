"""

"""
import numpy as np
import math

L, d_k, d_v, = 4, 8, 8
q = np.random.randn(L, d_k)
k = np.random.randn(L, d_k)
v = np.random.randn(L, d_v)

print("   Q\n", q)
print("   K\n", k)
print("   zV\n", v)
mask = np.tril(np.ones( (L, L) ))