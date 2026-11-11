from __future__ import annotations
import numpy as np
from .shapley import shapley_weights, shapley_from_table

try:
    import numba as nb

    @nb.njit(parallel=True, cache=True)
    def _shapley_kernel(v, n, w):
        N = v.shape[0]
        nthreads = nb.get_num_threads()
        acc = np.zeros((nthreads, n))
        chunk = (N + nthreads - 1) // nthreads
        for t in nb.prange(nthreads):
            lo = t * chunk
            hi = min(N, lo + chunk)
            for m in range(lo, hi):
                pc = 0
                x = m
                while x:
                    x &= x - 1
                    pc += 1
                vm = v[m]
                for i in range(n):
                    if not (m >> i) & 1:
                        acc[t, i] += w[pc] * (v[m | (1 << i)] - vm)
        out = np.zeros(n)
        for t in range(nthreads):
            for i in range(n):
                out[i] += acc[t, i]
        return out

    def shapley_from_table_fast(v: np.ndarray, n: int) -> np.ndarray:
        v = np.ascontiguousarray(v, dtype=np.float64)
        if v.ndim != 1 or v.shape[0] != (1 << n):
            raise ValueError("value table must be a 1-D array with 2^n entries")
        return _shapley_kernel(v, n, shapley_weights(n))
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

    def shapley_from_table_fast(v, n):
        return shapley_from_table(v, n)
