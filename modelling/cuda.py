import numpy as np
import numba
from numba import cuda

a = np.random.randint(0, 2, 2048)
b = np.random.randint(0, 2, 2048)
result = np.zeros(a.size)


def xor(a, b):
    return (a + b) % 2


@numba.njit
def xor_jit(a, b):
    return (a + b) % 2


@cuda.jit
def xor_cuda(a, b, result):
    i = cuda.grid(1)

    if i < a.size:
        result[i] = (a[i] + b[i]) % 2


# %timeit xor(a, b)
# %timeit xor_jit(a, b)
# %timeit xor_cuda(a, b, result)
