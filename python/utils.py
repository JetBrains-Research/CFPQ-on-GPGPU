import time
from functools import wraps

import numpy as np


def time_measure(f):
    @wraps(f)
    def inner(*args, **kwargs):
        time_start = time.time()
        out = f(*args, **kwargs)
        time_stop = time.time()
        return out, time_stop - time_start
    return inner


def mat_hash(mat):
    HASH_BASE_0 = 11
    HASH_BASE_1 = 13
    HASH_MOD = int(1e9 + 7)

    vec_base0 = np.frompyfunc(lambda a, b: (a * HASH_BASE_0 + b) % HASH_MOD, 2, 1)
    vec_base1 = np.frompyfunc(lambda a, b: (a * HASH_BASE_1 + b) % HASH_MOD, 2, 1)
    return vec_base1.reduce(vec_base0.reduce(mat, axis=0))
