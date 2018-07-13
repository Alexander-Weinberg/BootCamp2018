import numpy as np
from numba import njit, prange

@njit
def parallel_sum(A):
    sum = 0.0
    for i in prange(A.shape[0]):
        sum += A[i]

    return sum

a = np.random.random(100)
A = np.random.random(100_000_000)
parallel_sum(a)
B = parallel_sum(A)
